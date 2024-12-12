# Channel Attention Modules 

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import ModuleConfig

# Squeeze and Excitation Module (SE)
class SE(nn.Module):
    def __init__(self, in_channels, config: ModuleConfig):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // config.reduction, kernel_size=1, padding=0)
        self.activation = config.activation(inplace=True) if config.activation.__name__ != "Sigmoid" else config.activation()
        self.fc2 = nn.Conv2d(in_channels // config.reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.use_dropout = config.dropout_rt is not None and config.dropout_rt > 0
        
        if self.use_dropout:
            self.dropout = nn.Dropout(config.dropout_rt)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.activation(y)
        if self.use_dropout:
            y = self.dropout(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

# Convolutional Block Attention Module (CBAM)
class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
    def forward(self, x):
        pass

# Efficient Channel Attention Module (ECA)
class ECA(nn.Module):
    def __init__(self):
        super(ECA, self).__init__()
    def forward(self, x):
        pass





