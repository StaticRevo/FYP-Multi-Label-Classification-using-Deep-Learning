# Third-party imports
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# Local application imports
from config.config import ModuleConfig

# Squeeze and Excitation Module (SE)
class SE(nn.Module):
    def __init__(self, in_channels):
        super(SE, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc1 = nn.Conv2d(in_channels, in_channels // ModuleConfig.reduction, kernel_size=1, padding=0)
        self.activation = ModuleConfig.activation(inplace=True) if ModuleConfig.activation.__name__ != "Sigmoid" else ModuleConfig.activation()
        self.fc2 = nn.Conv2d(in_channels // ModuleConfig.reduction, in_channels, kernel_size=1, padding=0)
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

# Convolutional Block Attention Module (CBAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // ModuleConfig.reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // ModuleConfig.reduction, in_channels, kernel_size=1, padding=0)
        
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# Efficient Channel Attention Module (ECA)
class ECA(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)  
        y = self.conv(y.unsqueeze(1)).squeeze(1) 
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1) 
        return x * y

# Spatial Attention Module (SA)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

# Residual Block Module (ResidualBlock)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# Spectral Attention Module (SA)
class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // ModuleConfig.reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // ModuleConfig.reduction, in_channels)
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
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.channel_att = SpectralAttention(in_channels) # Channel Attention
        self.spatial_att = SpatialAttention(kernel_size=7) # Spatial Attention
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

