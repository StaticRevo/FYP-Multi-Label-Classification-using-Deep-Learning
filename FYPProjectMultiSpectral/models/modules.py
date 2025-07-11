# Third-party imports
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

# Local application imports
from config.config import ModuleConfig

# -- Bottle Neck Blocks -- 
# Bottlneck Residual Block (as used in ResNet50 adapted from ResNet)
class Bottleneck(nn.Module):
    expansion = ModuleConfig.expansion * 2
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First Conv2D
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second Conv2D
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third Conv2D
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
# Wide Bottleneck Residual Block (as used in WRN-50-2 adapted from WideResNet) - 3 Convolutional Layers (1x1, 3x3, 1x1)
class WideBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, widen_factor=2):
        super(WideBottleneck, self).__init__()
        self.expansion = widen_factor
        wide_channels = out_channels * widen_factor  # Double middle channels for WRN-50-2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, wide_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(wide_channels)

        self.conv3 = nn.Conv2d(wide_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dropout_rt = ModuleConfig.dropout_rt
    
    def drop_path(self, x, drop_prob):
        if drop_prob > 0. and self.training:
            keep_prob = 1 - drop_prob
            mask = torch.bernoulli(torch.ones(x.size(0), 1, 1, 1, device=x.device) * keep_prob)
            x.div_(keep_prob)
            x.mul_(mask)
            
        return x

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out, self.dropout_rt)

        out += identity
        out = self.relu(out)

        return out

# Wide Bottleneck Residual Block with ECA as used in WRN-B4-ECA - 2 Convolutional Layers (1x1, 3x3) with ECA
class WideBottleneckECA(nn.Module):
    expansion = 4  # WideResNet uses a widen factor of 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(WideBottleneckECA, self).__init__()
        final_channels = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, final_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(final_channels)

        self.eca = ECA(final_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First Conv2D
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second Conv2D
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply ECA
        out = self.eca(out)

        # Downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection
        out += identity
        out = self.relu(out)

        return out

# Wide Bottleneck Residual Block with ECA as used in WRN-B4-CBAM - 2 Convolutional Layers (3x3, 3x3)
class WideBasicBlockECA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.3):
        super(WideBasicBlockECA, self).__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
      
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ECA Module
        self.eca = ECA(out_channels)

        # Downsample
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        # First Conv2D
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        # Second Conv2D
        out = self.conv2(out)

        # Apply ECA
        out = self.eca(out)

        # Downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection
        out += identity
        out = self.relu(out)

        return out

# -- Spectral/Spatial Attention Modules --
# Squeeze and Excitation Module (SE)
class SE(nn.Module):
    def __init__(self, in_channels, kernel_size=1):
        super(SE, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1) 

        # Excitation
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // ModuleConfig.reduction), kernel_size=kernel_size, stride=1, padding=0, dilation=1, bias=False)           
        self.activation = ModuleConfig.activation(inplace=True) if ModuleConfig.activation.__name__ != "Sigmoid" else ModuleConfig.activation()
        self.fc2 = nn.Conv2d(in_channels=(in_channels // ModuleConfig.reduction), out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.use_dropout = ModuleConfig.dropout_rt is not None and ModuleConfig.dropout_rt > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(ModuleConfig.dropout_rt)

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x) # Global Average Pooling

        # Excitation
        y = self.fc1(y) 
        y = self.activation(y)
        if self.use_dropout:
            y = self.dropout(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        return x * y # Return the scaled input

# Efficient Channel Attention Module (ECA)
class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()  
        
        t = int(abs((math.log2(in_channels) / gamma) + b / gamma))
        k_size = t if t % 2 else t + 1  # Ensure k_size is odd
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, stride=1,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        y = self.avg_pool(x) # Global Average Pooling
        y = y.view(y.size(0), y.size(1)).unsqueeze(1) # Reshape for 1D convolution
        y = self.conv(y) # 1D Convolution
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # Reshape back
        y = self.sigmoid(y) # Sigmoid activation to generate attention weights
        
        return x * y # Scale the input with attention weights
    
# Drop Path Module (DropPath)
class DropPath(nn.Module):
    def __init__(self, drop_prob=ModuleConfig.drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: 
            return x
        
        keep_prob = 1 - self.drop_prob # Calculate the probability of keeping a path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # Create a shape for the binary mask
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # Generate a random tensor
        random_tensor.floor_()  # Binarize the tensor

        return x.div(keep_prob) * random_tensor # Scale the input with the binary mask

# Spatial Attention Module (SA)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size, stride):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                              dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # Average Pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Max Pooling
        y = torch.cat([avg_out, max_out], dim=1) # Concatate avg_out and max_out
        y = self.conv(y)
        y = self.sigmoid(y)

        return x * y
    
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
        y = self.sigmoid(y).view(batch, channels, 1, 1) # Sigmoid activation and Reshape to match input shape

        return x + y

# Dual Attention Module (DA) - Combines Channel Attention and Spatial Attention
class DualAttention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DualAttention, self).__init__()
        self.channel_att = SpectralAttention(in_channels=in_channels) # Spectral Attention
        self.spatial_att = SpatialAttention(kernel_size=kernel_size, stride=stride) # Spatial Attention
    
    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)

        return out

# Coordinate Attention Module (CA)
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction) # Compute reduced channel size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, 
                               padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, 
                                padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.size() # h, w = 60, 60 
        x_h = F.adaptive_avg_pool2d(x, (h, 1))  # Pool along height
        x_w = F.adaptive_avg_pool2d(x, (1, w))  # Pool along width
        x_w = x_w.permute(0, 1, 3, 2)  # Permute x_w so its spatial dimensions match for concatenation

        y = torch.cat([x_h, x_w], dim=2) # Concatenate along the height dimension 
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split features back into height and width parts
        x_h, x_w = torch.split(y, [h, w], dim=2)  
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_w = a_w.permute(0, 1, 3, 2)

        return x * a_h * a_w
    
# Depthwise Separable Convolution Module (DepthwiseSeparableConv)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, padding_mode):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode)
        self.stride = stride
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.depthwise(x) # Depthwise Convolution
        out = self.pointwise(out) # Pointwise Convolution

        if self.downsample is not None: # If downsample is not None, apply downsample
            x = self.downsample(x)

        if x.shape == out.shape: # If input and output shapes match, add a residual connection
            return x + out

        return out

# Multi-Scale Block Module (MultiScaleBlock)
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'):
        super(MultiScaleBlock, self).__init__()
        self.conv_dil1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, bias=bias, padding_mode=padding_mode)
        self.conv_dil2 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding=2, dilation=2, bias=bias, padding_mode=padding_mode)
        self.conv_dil3 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding=3, dilation=3, bias=bias, padding_mode=padding_mode)
        self.fuse = nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dil1 = self.conv_dil1(x)  # Local features (3x3 receptive field)
        dil2 = self.conv_dil2(x)  # Mid-range features (5x5 receptive field)
        dil3 = self.conv_dil3(x)  # Global features (7x7 receptive field)
        
        out = torch.cat([dil1, dil2, dil3], dim=1)  # Concatenate along channels
        out = self.relu(self.fuse(out))  # Fuse to out_channels
        
        return x + out  # Residual connection
    
# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        reduced_channels = max(16, in_channels // 8) # Compute reduced channel size

        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=reduced_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_out = x * self.channel_att(x)

        # Spatial attention
        avg_out = torch.mean(channel_out, dim=1, keepdim=True) # Average Pooling
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True) # Max Pooling
        spatial_in = torch.cat([avg_out, max_out], dim=1) # Concatenate avg_out and max_out
        spatial_out = self.spatial_att(spatial_in)

        out = channel_out * spatial_out # Combine
        return out + x # Residual Connection
    
