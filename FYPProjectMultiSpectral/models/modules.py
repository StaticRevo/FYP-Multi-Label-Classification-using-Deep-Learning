# Third-party imports
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# Local application imports
from config.config import ModuleConfig

# Squeeze and Excitation Module (SE)
class SE(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(SE, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Excitation
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // ModuleConfig.reduction), kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.activation = ModuleConfig.activation(inplace=True) if ModuleConfig.activation.__name__ != "Sigmoid" else ModuleConfig.activation()
        self.fc2 = nn.Conv2d(in_channels=(in_channels // ModuleConfig.reduction), out_channels=in_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

        self.use_dropout = ModuleConfig.dropout_rt is not None and ModuleConfig.dropout_rt > 0
        
        if self.use_dropout:
            self.dropout = nn.Dropout(ModuleConfig.dropout_rt)

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)
        # Excitation
        y = self.fc1(y)
        y = self.activation(y)
        if self.use_dropout:
            y = self.dropout(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

# Efficient Channel Attention Module (ECA)
class ECA(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2,
                              dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)  
        y = self.conv(y.unsqueeze(1)).squeeze(1) 
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1) 
        return x * y

# Drop Path Module (DropPath)
class DropPath(nn.Module):
    def __init__(self, drop_prob=ModuleConfig.dropout_rt):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # Compute a binary mask
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

# Residual Block Module (ResidualBlock)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, 
                               padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=ModuleConfig.dropout_rt)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, 
                               padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, 
                          padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
                nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        self.drop_path = DropPath(ModuleConfig.dropout_rt)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_path(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# Spatial Attention Module (SA)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                              dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
    
# Spectral Attention Module (SA)
class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=(in_channels // ModuleConfig.reduction), bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=(in_channels // ModuleConfig.reduction), out_features=in_channels, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        y = x.view(batch, channels, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

# Dual Attention Module (DA) - Combines Channel Attention and Spatial Attention
class DualAttention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DualAttention, self).__init__()
        self.channel_att = SpectralAttention(in_channels=in_channels) # Spectral Attention
        self.spatial_att = SpatialAttention(kernel_size=kernel_size, stride=stride) # Spatial Attention
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
    
# Depthwise Separable Convolution Module (DepthwiseSeparableConv)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Coordinate Attention Module (CA)
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, 
                               padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, 
                                padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.size()
        
        x_h = F.adaptive_avg_pool2d(x, (h, 1)) # Pool along height
        x_w = F.adaptive_avg_pool2d(x, (1, w)) # Pool along width
        x_w = x_w.permute(0, 1, 3, 2) # Permute x_w so its spatial dimensions match for concatenation
        y = torch.cat([x_h, x_w], dim=2) # Concatenate along the height dimension (dim=2)
        y = self.conv1(y)  
        y = self.bn1(y)
        y = self.act(y)
        # Split features back into height and width parts
        x_h, x_w = torch.split(y, [h, w], dim=2)  # x_h: (n, mid_channels, h, 1), x_w: (n, mid_channels, w, 1)
        a_h = self.conv_h(x_h).sigmoid()  
        a_w = self.conv_w(x_w).sigmoid()  
        a_w = a_w.permute(0, 1, 3, 2)  
        out = x * a_h * a_w  # Multiply the attention maps with the original input
        return out

# Multi-Scale Block Module (MultiScaleBlock)
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias, padding_mode):
        super(MultiScaleBlock, self).__init__()
        self.conv_dil1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=1, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_dil2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=2, dilation=2, groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_dil3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=3, dilation=3, groups=groups, bias=bias, padding_mode=padding_mode)
        self.fuse = nn.Conv2d(in_channels=(out_channels * 3), out_channels=out_channels, kernel_size=1, stride=stride,
                              padding=0, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        dil1 = self.conv_dil1(x)
        dil2 = self.conv_dil2(x)
        dil3 = self.conv_dil3(x)
        out = torch.cat([dil1, dil2, dil3], dim=1)
        out = self.fuse(out)
        return out
