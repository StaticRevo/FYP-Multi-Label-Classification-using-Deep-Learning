# Third-party imports
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math

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
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, 
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

# Positional Encoding Module (PositionalEncoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
 
    def forward(self, x):   
        seq_len = x.size(1) # x: (B, seq_len, d_model)
        return x + self.pe[:, :seq_len, :]
    
# Transformer Module (TransformerModule)
class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=1, dropout=0.1, return_mode="reshape"):
        super(TransformerModule, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.return_mode = return_mode

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2) # Flatten spatial dimensions -> (B, H*W, C)
        tokens = self.positional_encoding(tokens)
        tokens = self.transformer_encoder(tokens)

        if self.return_mode == "reshape":
            tokens = tokens.transpose(1, 2).reshape(B, C, H, W) # Reshape back to 4D
        elif self.return_mode == "pool":
            tokens = tokens.mean(dim=1)  # Aggregate tokens to (B, C)
        else:
            raise ValueError("Unsupported return_mode. Choose 'reshape' or 'pool'.")
        return tokens
    
# ASPP Module (ASPP)
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, dilations, bias, padding_mode):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[0], stride=strides[0],
                               padding=dilations[0], dilation=dilations[0], groups=1, bias=bias, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[1], stride=strides[1],
                               padding=dilations[1], dilation=dilations[1], groups=1, bias=bias, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[2], stride=strides[2],
                               padding=dilations[2], dilation=dilations[2], groups=1, bias=bias, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[3], stride=strides[3],
                               padding=dilations[3], dilation=dilations[3], groups=1, bias=bias, padding_mode=padding_mode)
        self.conv5 = nn.Conv2d(in_channels=(4 * out_channels), out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                               groups=1, bias=bias, padding_mode=padding_mode)
        self.upsample = nn.Upsample(mode="bilinear", align_corners=True)  

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        target_size = x1.shape[2:]  
        x2 = F.interpolate(x2, size=target_size, mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=target_size, mode="bilinear", align_corners=True)
        x4 = F.interpolate(x4, size=target_size, mode="bilinear", align_corners=True)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv5(x)
        return x

# Swin Transformer Block Module (SwinTransformerBlock)
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size):
        super(SwinTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)  
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)  
        self.norm2 = nn.LayerNorm(dim)  
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x):
        B, C, H, W = x.shape  # Extract batch, channels, height, width
        assert C == self.norm1.normalized_shape[0], f"Expected {self.norm1.normalized_shape[0]} channels, got {C}"

        x = x.flatten(2).permute(0, 2, 1)  # Change to (B, H*W, C) for attention
        x = self.norm1(x)  # Normalize along feature dimension
        attn_out, _ = self.attn(x, x, x)
        x = attn_out + x  # Residual connection
        x = self.norm2(x)
        x = self.mlp(x) + x  # Residual connection for MLP
        
        return x.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to (B, C, H, W)

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        reduced_channels = max(16, in_channels // 8)

        # Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_att(x)  
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return x * self.spatial_att(x)
    
# Mixed Depthwise Convolution Module
class MixedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], stride=1, groups=1):
        super(MixedDepthwiseConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, 
                      padding=k//2, groups=groups, bias=False)
            for k in kernel_sizes
        ])
        self.pointwise = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.pointwise(x)