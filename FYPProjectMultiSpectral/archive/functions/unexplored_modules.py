# Standard library imports
import math

# Third-party imports
import torch 
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# -- Old Custom Model version Modules --
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
        if in_channels != out_channels or stride != 1: # Trigger downsample if channels or spatial dims change
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
        if self.downsample is not None:
            residual = self.downsample(x)

        if out.shape != residual.shape:  # Ensure shapes match before addition
            raise RuntimeError(f"Shape mismatch: out {out.shape} vs residual {residual.shape}")
        
        # Residual connection
        out += residual
        out = self.relu(out)
        
        return out

# Positional Encoding Module (PositionalEncoding) - adds positional information to the input tokens
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # positional encoding tensor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute sine and cosine for even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)
 
    def forward(self, x):   
        seq_len = x.size(1) 

        return x + self.pe[:, :seq_len, :] # Add positional encoding
     
# Transformer Module (TransformerModule) - applies a transformer encoder to 2D feature maps
class TransformerModule(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=1, dropout=0.1, return_mode="reshape"):
        super(TransformerModule, self).__init__()
        # Single transformer encoder layer
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
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
    
# Atrous Spatial Pyramid Pooling (ASPP) - computes features at multiple scales using parallel dilated convolutions
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
        x1 = self.conv1(x) # Apply convolutions with different dilation rates
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        target_size = x1.shape[2:] # Ensure all feature maps have the same spatial dimensions
        x2 = F.interpolate(x2, size=target_size, mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=target_size, mode="bilinear", align_corners=True)
        x4 = F.interpolate(x4, size=target_size, mode="bilinear", align_corners=True)

        x = torch.cat([x1, x2, x3, x4], dim=1) # Concatenate the feature maps
        out = self.conv5(x) # Apply 1x1 convolution to fuse the features together
        
        if x.shape == out.shape: # If input and output shapes match, add a residual connection
            return x + out
        
        return out
    
# Mixed Depthwise Convolution Module - applies multiple depthwise separable convolutions with different kernel sizes in parallel
class MixedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], stride=1):
        super(MixedDepthwiseConv, self).__init__()
        self.convs = nn.ModuleList([ # Create multiple depthwise separable convolutions
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=stride, padding=k//2, groups=in_channels, bias=False)
            for k in kernel_sizes
        ])
        self.pointwise = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1, bias=False) # Pointwise convolution 

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1) # Apply depthwise separable convolutions in parallel and concatenate
        out = self.pointwise(out) # Apply pointwise convolution

        if x.shape == out.shape: 
            return x + out
        
        return out

