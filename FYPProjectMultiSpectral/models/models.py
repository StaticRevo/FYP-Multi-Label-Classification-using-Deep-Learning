# Third-party imports
from torch import nn
from torchsummary import summary
import timm
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, 
    resnet101, ResNet101_Weights, resnet152, ResNet152_Weights,
    vgg16, VGG16_Weights, vgg19, VGG19_Weights, 
    densenet121, DenseNet121_Weights, 
    efficientnet_b0, EfficientNet_B0_Weights, 
    efficientnet_v2_m, EfficientNet_V2_M_Weights
)

# Local application imports
from config.config import ModelConfig, ModuleConfig
from models.base_model import BaseModel
from models.modules import *

# Custom Model
class CustomModel(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()  # Dummy model for custom architecture to pass to the base model
        super(CustomModel, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # -- Spectral Mixing and Initial Feature Extraction --
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (in_channels->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (64->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer
            SpectralAttention(in_channels=32),  # SpectralAttention Module (32->32)
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling Layer - (120x120 -> 60x60)
        )

        # -- Block 1 --
        block1_downsample = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (32->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            WideBottleneck(in_channels=32, out_channels=24, stride=1, downsample=block1_downsample, widen_factor=2),  # WideBottleneck Block (32->48 (24*2))  
            SpectralAttention(in_channels=48),  # SpectralAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt),  # Dropout Layer
            CoordinateAttention(in_channels=48, reduction=16),  # CoordinateAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )

        # -- Block 2 --
        block2_downsample = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (48->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (48->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=48, out_channels=48, stride=1, downsample=block2_downsample, widen_factor=2),  # WideBottleneck Block (48->96 (48*2)) 
            ECA(in_channels=96),  # ECA Module
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )

        # -- Block 3 --
        block3_downsample = nn.Sequential(
            nn.Conv2d(96, 168, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (96->168)
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (96->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=96, out_channels=84, stride=1, downsample=block3_downsample, widen_factor=2),  # WideBottleneck Block (96->168 (84*2)) 
            SE(in_channels=168, kernel_size=1),  # Squeeze and Excitation Module (168->168)
            nn.Dropout(p=ModuleConfig.dropout_rt * 1.5)  # Dropout Layer
        )

        # -- Skip Connection Adapters --
        self.skip_adapter = nn.Sequential( # Skip Connection from Block 1 to Block Fusion 1
            nn.Conv2d(in_channels=48, out_channels=168, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (48->168)
            nn.AvgPool2d(kernel_size=4, stride=4)  # Average Pooling Layer (60x60 -> 15x15)
        )
        self.skip_adapter_spectral = nn.Sequential(  # Skip Connection from Spectral Mixer to Fusion 2
            nn.Conv2d(in_channels=32, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=4, stride=4)  # Downsample from 60x60 to 15x15
        )
        self.skip_adapter_mid = nn.Sequential( # Skip Connection from Block 2 to Fusion 2
            nn.Conv2d(in_channels=96, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros'),  # (96->232)
            nn.AvgPool2d(kernel_size=2, stride=2)  # (30x30 -> 15x15)
        )
        self.skip_adapter_deep = nn.Sequential( # Skip Connection from Block 3 to Fusion 2
            nn.Conv2d(in_channels=168, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros')  # (168->232)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(336, 2, kernel_size=1, bias=False),  # 336 -> 2
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(928, 4, kernel_size=1, bias=False),  # 928 -> 4 
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

        # -- Block 4 -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(168, 232, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (168->232)
            nn.BatchNorm2d(num_features=232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block4 = nn.Sequential(
            MultiScaleBlock(in_channels=168, out_channels=168, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            WideBottleneck(in_channels=168, out_channels=116, stride=1, downsample=block4_downsample, widen_factor=2),  # WideBottleneck Block (168->232 (116*2))
            CBAM(in_channels=232),  # CBAM Module (Channel+Spatial Attention) (232->232)
            nn.Dropout(p=ModuleConfig.dropout_rt * 2)  # Dropout Layer
        )

        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            ECA(in_channels=232),  # ECA Module (232->232) 
            nn.Conv2d(in_channels=232, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (232->128)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # Adaptive Average Pooling Layer
            nn.Flatten(),  # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt * 2),  # Dropout Layer
            nn.Linear(in_features=128, out_features=num_classes)  # Fully Connected Layer (128->19)
        )
    
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x)  # Spectral mixing (32, 60, 60)
        features_low = self.block1(x)  # Block 1: Low-level features (48, 60, 60)
        features_mid = self.block2(features_low)  # Block 2: Mid-level features (96, 30, 30)
        features_deep = self.block3(features_mid)  # Block 3: Deep features (168, 15, 15)

        # Skip Connection Adapters
        adapted_features_low = self.skip_adapter(features_low)  # (168, 15, 15)
        adapted_features_mid = self.skip_adapter_mid(features_mid)  # (232, 15, 15)
        adapted_features_deep = self.skip_adapter_deep(features_deep)  # (232, 15, 15)
        adapted_features_spectral = self.skip_adapter_spectral(x)  # (232, 15, 15)

        # Fusion 1: Learned convolutional approach
        fused_input = torch.cat([features_deep, adapted_features_low], dim=1)  # (336, 15, 15)
        weights = self.fusion_conv(fused_input)  # (2, 15, 15)
        w_deep, w_low = weights[:, 0:1, :, :], weights[:, 1:2, :, :]  # Split into two masks
        fused_features = (w_deep * features_deep) + (w_low * adapted_features_low)  # (168, 15, 15)

        features_high = self.block4(fused_features)  # Block 4: High-level features (232, 15, 15)

        # Fusion 2: Learned convolutional approach
        fusion_input2 = torch.cat([features_high, adapted_features_mid, adapted_features_deep, adapted_features_spectral], dim=1)  # (928, 15, 15)
        weights2 = self.fusion_conv2(fusion_input2)  # (4, 15, 15)
        w_high, w_mid, w_deep, w_early = weights2[:, 0:1, :, :], weights2[:, 1:2, :, :], weights2[:, 2:3, :, :], weights2[:, 3:4, :, :]  # Split into four masks
        fused_features_high = (w_high * features_high) + (w_mid * adapted_features_mid) + (w_deep * adapted_features_deep) + (w_early * adapted_features_spectral)  # (232, 15, 15)

        out = self.classifier(fused_features_high)  # Classifier (19)
        return out
    
    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# Custom ResNet50 Model
class CustomResNet50(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity() # Dummy model for custom architecture to pass to the base model
        super(CustomResNet50, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) # Conv Layer (120,120) -> (64,60,60)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # MaxPool Layer (64,60,60) -> (64,30,30)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 3, stride=1) # (64,30,30) -> (256,30,30)  ## 3=ResNet50,ResNet101,ResNet152
        self.layer2 = self._make_layer(128, 4, stride=2) # (256,30,30) -> (512,15,15) ## 4=ResNet50,ResNet101, 8=ResNet152
        self.layer3 = self._make_layer(256, 6, stride=2) # (512,15,15) -> (1024,8,8)  ## 6=ResNet50, 23=ResNet101, 36=ResNet152
        self.layer4 = self._make_layer(512, 3, stride=2) # (1024,8,8) -> (2048,4,4) ## 3=ResNet50,ResNet101,ResNet152

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # AdaptiveAvgPool Layer (2048,4,4) -> (2048,1,1)
        self.fc = nn.Linear(512 * ModuleConfig.expansion, num_classes) # Fully Connected Layer (2048,1,1) -> (19)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if model_weights is not None or "None":
            print("Pretrained weights not supported in the custom ResNet50 model")

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * ModuleConfig.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * ModuleConfig.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ModuleConfig.expansion),
            )
        layers = []
        layers.append(WideBottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * ModuleConfig.expansion
        for _ in range(1, blocks):
            layers.append(WideBottleneck(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    # Override forward function for Custom ResNet50 Model
    def forward(self, x):
        x = self.conv1(x) # (19, 120, 120) -> (64, 60, 60)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (64, 60, 60) -> (64, 30, 30)

        x = self.layer1(x) # (64, 30, 30) -> (256, 30, 30)
        x = self.layer2(x) # (256, 30, 30) -> (512, 15, 15)
        x = self.layer3(x)  # (512, 15, 15) -> (1024, 8, 8)
        x = self.layer4(x) # (1024, 8, 8) -> (2048, 4, 4)

        x = self.avgpool(x) # (2048, 4, 4) -> (2048, 1, 1)
        x = torch.flatten(x, 1) # Flatten (2048, 1, 1) -> (2048)
        x = self.fc(x) # (2048) -> (19)

        return x

    # Override optimizer configuration for Custom ResNet50 Model
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor,  patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

# Custom WideResNetB4-ECA Model
class CustomWRNB4ECA(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()
        super(CustomWRNB4ECA, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)

        self.w = 2.0736 # Width scaling factor
        self.K = 1.15 # Widening factor adjustement 

        # Apply width scaling factor and adjustement
        self.in_channels = int(16 * self.K * self.w)  # 16 * 2.0736  * 1.15 = 38
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Define layers with WideBottleneckECA blocks
        self.layer1 = self._make_layer(int(16 * self.K * self.w), 2, stride=1)  # (38, 176, 176) -> (38, 176, 176)
        self.layer2 = self._make_layer(int(32 * self.K * self.w), 2, stride=2)  # (38, 176, 176) -> (76, 88, 88)
        self.layer3 = self._make_layer(int(64 * self.K * self.w), 2, stride=2)  # (76, 88, 88) -> (153, 44, 44)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (153, 44, 44) -> (153, 1, 1)
        self.fc = nn.Linear(int(64 * self.K * self.w), num_classes)  # (153) -> (19)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if model_weights is not None and model_weights != "None":
            print("Pretrained weights not supported in the custom WideResNetB4-ECA model")

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [WideBasicBlockECA(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(WideBasicBlockECA(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # [12, 176, 176] -> [38, 176, 176]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # [38, 176, 176] -> [38, 176, 176]
        x = self.layer2(x)  # [38, 176, 176] -> [76, 88, 88]
        x = self.layer3(x)  # [76, 88, 88] -> [153, 44, 44]
        x = self.avgpool(x)  # [153, 44, 44] -> [153, 1, 1]
        x = torch.flatten(x, 1)  # [153]
        x = self.fc(x)  # [153] -> [19]
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# Custom WideResNetB0 Model
class CustomWRNB0(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()
        super(CustomWRNB0, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)

        self.w = 1.0  # Width scaling factor for B0 (beta^phi = 1.2^0)
        self.K = 1.97  # Widening factor to match parameters from report

        # Apply width scaling factor and adjustment
        self.in_channels = int(16 * self.K * self.w)  # 16 * 1.97 * 1.0 ≈ 32
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Define layers with WideBasicBlockECA blocks
        self.layer1 = self._make_layer(int(16 * self.K * self.w), 1, stride=1)  # (32, 120, 120) -> (32, 120, 120)
        self.layer2 = self._make_layer(int(32 * self.K * self.w), 1, stride=2)  # (32, 120, 120) -> (63, 60, 60)
        self.layer3 = self._make_layer(int(64 * self.K * self.w), 1, stride=2)  # (63, 60, 60) -> (126, 30, 30)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (126, 30, 30) -> (126, 1, 1)
        self.fc = nn.Linear(int(64 * self.K * self.w), num_classes)  # (126) -> (19)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if model_weights is not None and model_weights != "None":
            print("Pretrained weights not supported in the custom WideResNetB0-ECA model")

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [WideBasicBlockECA(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(WideBasicBlockECA(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # [12, 120, 120] -> [32, 120, 120]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # [32, 120, 120] -> [32, 120, 120]
        x = self.layer2(x)  # [32, 120, 120] -> [63, 60, 60]
        x = self.layer3(x)  # [63, 60, 60] -> [126, 30, 30]
        x = self.avgpool(x)  # [126, 30, 30] -> [126, 1, 1]
        x = torch.flatten(x, 1)  # [126]
        x = self.fc(x)  # [126] -> [19]
        return x
    
    # Override optimizer configuration for Custom WRN B0 Model
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor,  patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# -- Custom Model Versions --
# Custom Model Version 1
class CustomModelV1(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        custom_model_1 = nn.Sequential(
            # -- Block 1 --
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=32, out_channels=32, stride=1), # Residual Block (32->32)  
            SpectralAttention(in_channels=32), # SpectralAttention Module

            # -- Block 2 --
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=64, out_channels=64, stride=1), # Residual Block (64->64) 
            ECA(in_channels=64), # ECA Module

            # -- BLock 3 -- 
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=128, out_channels=128, stride=1), # Residual Block (128->128) 
            SE(in_channels=128, kernel_size=1), # Squeeze and Excitation Module

            # -- Block 4 -- 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=256, out_channels=256, stride=1), # Residual Block (256->256)
            DualAttention(in_channels=256, kernel_size=3, stride=1), # DualAttention Module

            # Global Pool and Classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(ModelConfig.dropout),
            nn.Linear(256, num_classes)
        )
        super(CustomModelV1, self).__init__(custom_model_1, num_classes, class_weights, in_channels, main_path)
        
# Custom Model Version 2
class CustomModelV2(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        custom_model_2 = nn.Sequential(
            # Spectral Mixing 
            nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # -- Block 1 --
            DepthwiseSeparableConv(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (16->32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=32, out_channels=32, stride=1), # Residual Block (32->32)  
            SpectralAttention(in_channels=32), # SpectralAttention Module
            CoordinateAttention(in_channels=32, reduction=16),

            # -- Block 2 --
            DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (32->64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MultiScaleBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            ResidualBlock(in_channels=64, out_channels=64, stride=1), # Residual Block (64->64) 
            ECA(in_channels=64), # ECA Module

            # -- BLock 3 -- 
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (64->128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            MultiScaleBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            ResidualBlock(in_channels=128, out_channels=128, stride=1), # Residual Block (128->128) 
            SE(in_channels=128, kernel_size=1), # Squeeze and Excitation Module

            # -- Block 4 -- 
             DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (128->256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels=256, out_channels=256, stride=1), # Residual Block (256->256)
            DualAttention(in_channels=256, kernel_size=3, stride=1), # DualAttention Module

            # Global Pool and Classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(ModelConfig.dropout),
            nn.Linear(256, num_classes)
        )
        super(CustomModelV2, self).__init__(custom_model_2, num_classes, class_weights, in_channels, main_path)

# Custom Model Version 3
class CustomModelV3(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()
        super(CustomModelV3, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # Spectral Mixing & Initial Feature Extraction
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # -- Block 1 --
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (32->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            ResidualBlock(in_channels=64, out_channels=64, stride=1), # Residual Block (64->64)  
            SpectralAttention(in_channels=64), # SpectralAttention Module (64->64)
            CoordinateAttention(in_channels=64, reduction=16), # CoordinateAttention Module (64->64)
        )
        # -- Block 2 --
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (64->128)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            MultiScaleBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=1, 
                            bias=True, padding_mode='zeros'), # MultiScaleBlock (128->128)
            ResidualBlock(in_channels=128, out_channels=128, stride=1), # Residual Block (128->128) 
            ECA(in_channels=128), # ECA Module
        )
        # -- BLock 3 --
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (128->256)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            MultiScaleBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=1, bias=True, padding_mode='zeros'), # MultiScaleBlock (256->256)
            ResidualBlock(in_channels=256, out_channels=256, stride=1), # Residual Block (256->256) 
            SE(in_channels=256, kernel_size=1), # Squeeze and Excitation Module
        )
        self.transformer_block = TransformerModule(d_model=256, nhead=8, num_layers=1, dropout=0.2, return_mode="reshape")
        self.skip_adapter = nn.Conv2d(64, 256, kernel_size=1, bias=False)

        # -- Block 4 -- 
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (256->512)
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            ResidualBlock(in_channels=512, out_channels=512, stride=1), # Residual Block (512->512)
            DualAttention(in_channels=512, kernel_size=7, stride=1), # DualAttention Module (Spectal+Spatial Attention Modules)
        )
        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # Adaptive Average Pooling Layer
            nn.Flatten(), # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
            nn.Linear(in_features=512, out_features=num_classes) # Fully Connected Layer
        )
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x) # Step 1: Spectral mixing
        features_low = self.block1(x) # Step 2: Extract low-level features.
        features_mid = self.block2(features_low) # Step 3: Compute mid-level features.
        features_deep = self.block3(features_mid) # Step 4: Compute deep features.
        features_deep = self.transformer_block(features_deep)  # Apply transformer block for global context
        adapted_features_low = self.skip_adapter(features_low) # Adjust low-level features
    
        # If spatial dimensions differ interpolate adapted_features_low to match features_deep
        adapted_features_low = torch.nn.functional.interpolate(
            adapted_features_low,
            size=(15, 15),  
            mode='bilinear',
            align_corners=False
        )
    
        fused_features = features_deep + adapted_features_low  # Step 5: Fuse features.
        features_high = self.block4(fused_features)      # Step 6: Refine high-level representations.
        out = self.classifier(features_high)             # Step 7: Final classification.
        return out

    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor,  patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# Custom Model Version 4
class CustomModelV4(BaseModel):
     def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
         dummy_model = nn.Identity() # Dummy model for custom architecture to pass to the base model
         super(CustomModelV4, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
     
         # Spectral Mixing and Initial Feature Extraction
         self.spectral_mixer = nn.Sequential(
             nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                       dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (in_channels->64)
             nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
             nn.GELU(), # GELU Activation Function
             nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling Layer - (120x120 -> 60x60)
         )
         # -- Block 1 --
         block1_downsample = nn.Sequential(
             nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (64->128)
             nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
         )
         self.block1 = nn.Sequential(
             DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, 
                                    dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (64->64)
             nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
             nn.GELU(), # GELU Activation Function
             Bottleneck(in_channels=64, out_channels=32, stride=1, downsample=block1_downsample), # Bottleneck Block (64->128 (32*4))  
             SpectralAttention(in_channels=128), # SpectralAttention Module (128->128)
             CoordinateAttention(in_channels=128, reduction=16), # CoordinateAttention Module (128->128)
         )
         # -- Block 2 --
         block2_downsample = nn.Sequential(
             nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (128->256)
             nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
         )
         self.block2 = nn.Sequential(
             DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, 
                                    dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (128->128)
             nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
             nn.GELU(),
             MultiScaleBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=1, 
                             bias=True, padding_mode='zeros'), # MultiScaleBlock (128->128)
             Bottleneck(in_channels=128, out_channels=64, stride=1, downsample=block2_downsample), # Bottleneck Block (128->256 (64*4)) 
             ECA(in_channels=256), # ECA Module
         )
         # -- Block 3 --
         block3_downsample = nn.Sequential(
             nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (256->512)
             nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
         )
         self.block3 = nn.Sequential(
             DepthwiseSeparableConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (256->256)
             nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
             nn.GELU(),
             MultiScaleBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=1, bias=True, padding_mode='zeros'), # MultiScaleBlock (256->256)
             Bottleneck(in_channels=256, out_channels=128, stride=1, downsample=block3_downsample), # Bottleneck Block (256->512 (128*4)) 
             SE(in_channels=512, kernel_size=1), # Squeeze and Excitation Module
         )
         self.skip_adapter = nn.Sequential(
             nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (128->512)
             nn.AvgPool2d(kernel_size=4, stride=4) # Average Pooling Layer (60x60 -> 15x15)
         )
         # -- Block 4 -- 
         block4_downsample = nn.Sequential(
             nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (512->1024)
             nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
         )
         self.block4 = nn.Sequential(
             DepthwiseSeparableConv(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, 
                                    dilation=2, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (512->512)
             nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
             nn.GELU(),
             Bottleneck(in_channels=512, out_channels=256, stride=1, downsample=block4_downsample), # Bottleneck Block (512->1024 (256*4))
             DualAttention(in_channels=1024, kernel_size=7, stride=1), # DualAttention Module (Spectral+Spatial Attention Modules)
         )
         # -- Block 5 -- 
         self.classifier = nn.Sequential(
             nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer
             nn.GELU(),
             nn.AdaptiveAvgPool2d(1), # Adaptive Average Pooling Layer
             nn.Flatten(), # Flatten Layer
             nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
             nn.Linear(in_features=1024, out_features=num_classes) # Fully Connected Layer
         )
     
     # Override forward function for CustomModel
     def forward(self, x):
         x = self.spectral_mixer(x) # Spectral mixing (64, 60, 60)
         features_low = self.block1(x) # Block 1: Low-level features (128, 60, 60)
         features_mid = self.block2(features_low) # Block 2: Mid-level features (256, 30, 30)
         features_deep = self.block3(features_mid) # Block 3: Deep features (512, 15, 15)
         adapted_features_low = self.skip_adapter(features_low) # Adapt low-level features to match deep features (512, 15, 15)
         fused_features = features_deep + adapted_features_low  # Fuse low level and deep features (512, 15, 15)
         features_high = self.block4(fused_features) # Refine high-level representations (1024, 15, 15)
         out = self.classifier(features_high) # Final classification (19)
         return out
     
     # Override optimizer configuration for CustomModel
     def configure_optimizers(self):
         optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor,  patience=ModelConfig.lr_patience)                                                  
         return {
             'optimizer': optimizer,
             'lr_scheduler': {
                 'scheduler': scheduler,
                 'monitor': 'val_loss',
                 'interval': 'epoch',
                 'frequency': 1
             }
         }

# Custom Model Version 5
class CustomModelV5(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity() # Dummy model for custom architecture to pass to the base model
        super(CustomModelV5, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # Spectral Mixing and Initial Feature Extraction
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (in_channels->32)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'), # Convolutional Layer (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer
            nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling Layer - (120x120 -> 60x60)
        )
        # -- Block 1 --
        block1_downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (32->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            WideBottleneck(in_channels=32, out_channels=16, stride=1, downsample=block1_downsample, widen_factor=4), # WideBottleneck Block (32->64 (16*4))  
            SpectralAttention(in_channels=64), # SpectralAttention Module (64->64)
            nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
            CoordinateAttention(in_channels=64, reduction=16), # CoordinateAttention Module (64->64)
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 2 --
        block2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (64->128)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (64->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=64, out_channels=32, stride=1, downsample=block2_downsample, widen_factor=4), # WideBottleneck Block (64->128 (32*4)) 
            ECA(in_channels=128), # ECA Module,
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 3 --
        block3_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (128->256)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (128->128)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=128, out_channels=64, stride=1, downsample=block3_downsample, widen_factor=4), # WideBottleneck Block (128->256 (64*4)) 
            SE(in_channels=256, kernel_size=1), # Squeeze and Excitation Module
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        self.skip_adapter = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (64->256)
            nn.AvgPool2d(kernel_size=4, stride=4) # Average Pooling Layer (60x60 -> 15x15)
        )
        # -- Block 4 -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (256->512)
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, 
                                   dilation=2, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (256->256)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=256, out_channels=128, stride=1, downsample=block4_downsample, widen_factor=4), # WideBottleneck Block (256->512 (128*4))
            CBAM(in_channels=512), # CBAM Module (Channel+Spatial Attention)
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (512->256)
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1), # Adaptive Average Pooling Layer
            nn.Flatten(), # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
            nn.Linear(in_features=256, out_features=num_classes) # Fully Connected Layer (256->19)
        )
    
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x) # Spectral mixing (32, 60, 60)
        features_low = self.block1(x) # Block 1: Low-level features (64, 60, 60)
        features_mid = self.block2(features_low) # Block 2: Mid-level features (128, 30, 30)
        features_deep = self.block3(features_mid) # Block 3: Deep features (256, 15, 15)
        adapted_features_low = self.skip_adapter(features_low) # Adapt low-level features to match deep features (256, 15, 15)
        fused_features = features_deep + adapted_features_low  # Fuse low level and deep features (256, 15, 15)
        features_high = self.block4(fused_features) # Refine high-level representations (512, 15, 15)
        out = self.classifier(features_high) # Final classification (19)
        
        return out
    
    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

# Custom Model Version 6
class CustomModelV6(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity() # Dummy model for custom architecture to pass to the base model
        super(CustomModelV6, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # Spectral Mixing and Initial Feature Extraction
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (in_channels->32)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'), # Convolutional Layer (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer
            nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling Layer - (120x120 -> 60x60)
        )
        # -- Block 1 --
        block1_downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (32->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            WideBottleneck(in_channels=32, out_channels=16, stride=1, downsample=block1_downsample, widen_factor=4), # WideBottleneck Block (32->64 (16*4))  
            SpectralAttention(in_channels=64), # SpectralAttention Module (64->64)
            nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
            CoordinateAttention(in_channels=64, reduction=16), # CoordinateAttention Module (64->64)
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 2 --
        block2_downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (64->128)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (64->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=64, out_channels=32, stride=1, downsample=block2_downsample, widen_factor=4), # WideBottleneck Block (64->128 (32*4)) 
            ECA(in_channels=128), # ECA Module,
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 3 --
        block3_downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (128->256)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (128->128)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=128, out_channels=64, stride=1, downsample=block3_downsample, widen_factor=4), # WideBottleneck Block (128->256 (64*4)) 
            SE(in_channels=256, kernel_size=1), # Squeeze and Excitation Module
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        self.skip_adapter = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (64->256)
            nn.AvgPool2d(kernel_size=4, stride=4) # Average Pooling Layer (60x60 -> 15x15)
        )
        # -- Block 4 -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False, padding_mode='zeros'), # Downsample path (256->512)
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # Batch Normalization for downsample
        )
        self.block4 = nn.Sequential(
            MultiScaleBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            WideBottleneck(in_channels=256, out_channels=128, stride=1, downsample=block4_downsample, widen_factor=4), # WideBottleneck Block (256->512 (128*4))
            CBAM(in_channels=512), # CBAM Module (Channel+Spatial Attention)
            nn.Dropout(p=ModuleConfig.dropout_rt) # Dropout Layer
        )
        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (512->256)
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1), # Adaptive Average Pooling Layer
            nn.Flatten(), # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt), # Dropout Layer
            nn.Linear(in_features=256, out_features=num_classes) # Fully Connected Layer (256->19)
        )
    
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x) # Spectral mixing (32, 60, 60)
        features_low = self.block1(x) # Block 1: Low-level features (64, 60, 60)
        features_mid = self.block2(features_low) # Block 2: Mid-level features (128, 30, 30)
        features_deep = self.block3(features_mid) # Block 3: Deep features (256, 15, 15)
        adapted_features_low = self.skip_adapter(features_low) # Adapt low-level features to match deep features (256, 15, 15)
        fused_features = features_deep + adapted_features_low  # Fuse low level and deep features (256, 15, 15)
        features_high = self.block4(fused_features) # Refine high-level representations (512, 15, 15)
        out = self.classifier(features_high) # Final classification (19)
        
        return out

    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

# Custom Model Version 7
class CustomModelV7(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()  # Dummy model for custom architecture to pass to the base model
        super(CustomModelV7, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # Spectral Mixing and Initial Feature Extraction
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (in_channels->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (64->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling Layer - (120x120 -> 60x60)
        )
        # -- Block 1 --
        block1_downsample = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (32->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            WideBottleneck(in_channels=32, out_channels=24, stride=1, downsample=block1_downsample, widen_factor=2),  # WideBottleneck Block (32->48 (24*2))  
            SpectralAttention(in_channels=48),  # SpectralAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt),  # Dropout Layer
            CoordinateAttention(in_channels=48, reduction=16),  # CoordinateAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )
        # -- Block 2 --
        block2_downsample = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (48->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (48->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=48, out_channels=48, stride=1, downsample=block2_downsample, widen_factor=2),  # WideBottleneck Block (48->96 (48*2)) 
            ECA(in_channels=96),  # ECA Module
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )
        # -- Block 3 --
        block3_downsample = nn.Sequential(
            nn.Conv2d(96, 168, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (96->168)
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (96->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=96, out_channels=84, stride=1, downsample=block3_downsample, widen_factor=2),  # WideBottleneck Block (96->168 (84*2)) 
            SE(in_channels=168, kernel_size=1),  # Squeeze and Excitation Module (168->168)
            nn.Dropout(p=ModuleConfig.dropout_rt * 1.5)  # Dropout Layer
        )
        # -- Skip Connection Adapters --
        self.skip_adapter = nn.Sequential( # Skip Connection from Block 1 to Block 3
            nn.Conv2d(in_channels=48, out_channels=168, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (48->168)
            nn.AvgPool2d(kernel_size=4, stride=4)  # Average Pooling Layer (60x60 -> 15x15)
        )
        self.skip_adapter_mid = nn.Sequential( # Skip Connection from Block 2 to Block 4
            nn.Conv2d(in_channels=96, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros'),  # (96->232)
            nn.AvgPool2d(kernel_size=2, stride=2)  # (30x30 -> 15x15)
        )
        self.skip_adapter_deep = nn.Sequential( # Skip Connection from Block 3 to Block 4
            nn.Conv2d(in_channels=168, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros')  # (168->232)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(336, 2, kernel_size=1, bias=False),  # (336 -> 2)
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )
        # -- Block 4 -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(168, 232, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (168->232)
            nn.BatchNorm2d(num_features=232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block4 = nn.Sequential(
            MultiScaleBlock(in_channels=168, out_channels=168, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            WideBottleneck(in_channels=168, out_channels=116, stride=1, downsample=block4_downsample, widen_factor=2),  # WideBottleneck Block (168->232 (116*2))
            CBAM(in_channels=232),  # CBAM Module (Channel+Spatial Attention) (232->232)
            nn.Dropout(p=ModuleConfig.dropout_rt * 2)  # Dropout Layer
        )
        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=232, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (232->128)
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # Adaptive Average Pooling Layer
            nn.Flatten(),  # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt * 2),  # Dropout Layer
            nn.Linear(in_features=128, out_features=num_classes)  # Fully Connected Layer (128->19)
        )
    
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x)  # Spectral mixing (32, 60, 60)
        features_low = self.block1(x)  # Block 1: Low-level features (48, 60, 60)
        features_mid = self.block2(features_low)  # Block 2: Mid-level features (96, 30, 30)
        features_deep = self.block3(features_mid)  # Block 3: Deep features (168, 15, 15)

        # Skip Connection Adapters
        adapted_features_low = self.skip_adapter(features_low)  # (168, 15, 15)
        adapted_features_mid = self.skip_adapter_mid(features_mid)  # (232, 15, 15)
        adapted_features_deep = self.skip_adapter_deep(features_deep)  # (232, 15, 15)

        # Lightweight attention-guided fusion for features_deep and adapted_features_low
        fused_input = torch.cat([features_deep, adapted_features_low], dim=1)  # (336, 15, 15)
        weights = self.fusion_conv(fused_input)  # (2, 15, 15)
        w_deep, w_low = weights[:, 0:1, :, :], weights[:, 1:2, :, :]  # Split into two masks
        fused_features = (w_deep * features_deep) + (w_low * adapted_features_low)  # (168, 15, 15)

        features_high = self.block4(fused_features)  # Block 4: High-level features (232, 15, 15)

        # Parameter-free attention-guided fusion for features_high, adapted_features_mid, and adapted_features_deep
        mask_high = torch.sigmoid(torch.mean(features_high, dim=1, keepdim=True) + torch.max(features_high, dim=1, keepdim=True)[0])
        mask_mid = torch.sigmoid(torch.mean(adapted_features_mid, dim=1, keepdim=True) + torch.max(adapted_features_mid, dim=1, keepdim=True)[0])
        mask_deep = torch.sigmoid(torch.mean(adapted_features_deep, dim=1, keepdim=True) + torch.max(adapted_features_deep, dim=1, keepdim=True)[0])
        mask_sum = mask_high + mask_mid + mask_deep + 1e-8  # Add small epsilon to avoid division by zero
        mask_high = mask_high / mask_sum
        mask_mid = mask_mid / mask_sum
        mask_deep = mask_deep / mask_sum
        fused_features_high = (mask_high * features_high) + (mask_mid * adapted_features_mid) + (mask_deep * adapted_features_deep)

        out = self.classifier(fused_features_high)  # Classifier (19)
        return out
    
    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

# Custom Model Version 8
class CustomModelV8(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()  # Dummy model for custom architecture to pass to the base model
        super(CustomModelV8, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # -- Spectral Mixing and Initial Feature Extraction --
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (in_channels->64)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (64->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer
            SpectralAttention(in_channels=32),  # SpectralAttention Module (32->32)
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling Layer - (120x120 -> 60x60)
        )

        # -- Block 1 --
        block1_downsample = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (32->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (32->32)
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  # Batch Normalization Layer 
            nn.GELU(),  # GELU Activation Function
            WideBottleneck(in_channels=32, out_channels=24, stride=1, downsample=block1_downsample, widen_factor=2),  # WideBottleneck Block (32->48 (24*2))  
            SpectralAttention(in_channels=48),  # SpectralAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt),  # Dropout Layer
            CoordinateAttention(in_channels=48, reduction=16),  # CoordinateAttention Module (48->48)
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )

        # -- Block 2 --
        block2_downsample = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (48->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (48->48)
            nn.BatchNorm2d(num_features=48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=48, out_channels=48, stride=1, downsample=block2_downsample, widen_factor=2),  # WideBottleneck Block (48->96 (48*2)) 
            ECA(in_channels=96),  # ECA Module
            nn.Dropout(p=ModuleConfig.dropout_rt)  # Dropout Layer
        )

        # -- Block 3 --
        block3_downsample = nn.Sequential(
            nn.Conv2d(96, 168, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (96->168)
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'),  # Depthwise Separable Convolution (96->96)
            nn.BatchNorm2d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            WideBottleneck(in_channels=96, out_channels=84, stride=1, downsample=block3_downsample, widen_factor=2),  # WideBottleneck Block (96->168 (84*2)) 
            SE(in_channels=168, kernel_size=1),  # Squeeze and Excitation Module (168->168)
            nn.Dropout(p=ModuleConfig.dropout_rt * 1.5)  # Dropout Layer
        )

        # -- Skip Connection Adapters --
        self.skip_adapter = nn.Sequential( # Skip Connection from Block 1 to Block Fusion 1
            nn.Conv2d(in_channels=48, out_channels=168, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (48->168)
            nn.AvgPool2d(kernel_size=4, stride=4)  # Average Pooling Layer (60x60 -> 15x15)
        )
        self.skip_adapter_spectral = nn.Sequential(  # Skip Connection from Spectral Mixer to Fusion 2
            nn.Conv2d(in_channels=32, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=4, stride=4)  # Downsample from 60x60 to 15x15
        )
        self.skip_adapter_mid = nn.Sequential( # Skip Connection from Block 2 to Fusion 2
            nn.Conv2d(in_channels=96, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros'),  # (96->232)
            nn.AvgPool2d(kernel_size=2, stride=2)  # (30x30 -> 15x15)
        )
        self.skip_adapter_deep = nn.Sequential( # Skip Connection from Block 3 to Fusion 2
            nn.Conv2d(in_channels=168, out_channels=232, kernel_size=1, stride=1, padding=0, bias=False, padding_mode='zeros')  # (168->232)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(336, 2, kernel_size=1, bias=False),  # 336 -> 2
            nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(928, 4, kernel_size=1, bias=False),  # 928 -> 4 
            nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid()
        )

        # -- Block 4 -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(168, 232, kernel_size=1, stride=1, bias=False, padding_mode='zeros'),  # Downsample path (168->232)
            nn.BatchNorm2d(num_features=232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Batch Normalization for downsample
        )
        self.block4 = nn.Sequential(
            MultiScaleBlock(in_channels=168, out_channels=168, kernel_size=3, stride=1, groups=1, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(num_features=168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            WideBottleneck(in_channels=168, out_channels=116, stride=1, downsample=block4_downsample, widen_factor=2),  # WideBottleneck Block (168->232 (116*2))
            CBAM(in_channels=232),  # CBAM Module (Channel+Spatial Attention) (232->232)
            nn.Dropout(p=ModuleConfig.dropout_rt * 2)  # Dropout Layer
        )

        # -- Block 5 -- 
        self.classifier = nn.Sequential(
            ECA(in_channels=232),  # ECA Module (232->232) 
            nn.Conv2d(in_channels=232, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),  # Convolutional Layer (232->128)
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # Adaptive Average Pooling Layer
            nn.Flatten(),  # Flatten Layer
            nn.Dropout(p=ModuleConfig.dropout_rt * 2),  # Dropout Layer
            nn.Linear(in_features=128, out_features=num_classes)  # Fully Connected Layer (128->19)
        )
    
    # Override forward function for CustomModel
    def forward(self, x):
        x = self.spectral_mixer(x)  # Spectral mixing (32, 60, 60)
        features_low = self.block1(x)  # Block 1: Low-level features (48, 60, 60)
        features_mid = self.block2(features_low)  # Block 2: Mid-level features (96, 30, 30)
        features_deep = self.block3(features_mid)  # Block 3: Deep features (168, 15, 15)

        # Skip Connection Adapters
        adapted_features_low = self.skip_adapter(features_low)  # (168, 15, 15)
        adapted_features_mid = self.skip_adapter_mid(features_mid)  # (232, 15, 15)
        adapted_features_deep = self.skip_adapter_deep(features_deep)  # (232, 15, 15)
        adapted_features_spectral = self.skip_adapter_spectral(x)  # (232, 15, 15)

        # Fusion 1: Learned convolutional approach
        fused_input = torch.cat([features_deep, adapted_features_low], dim=1)  # (336, 15, 15)
        weights = self.fusion_conv(fused_input)  # (2, 15, 15)
        w_deep, w_low = weights[:, 0:1, :, :], weights[:, 1:2, :, :]  # Split into two masks
        fused_features = (w_deep * features_deep) + (w_low * adapted_features_low)  # (168, 15, 15)

        features_high = self.block4(fused_features)  # Block 4: High-level features (232, 15, 15)

        # Fusion 2: Learned convolutional approach
        fusion_input2 = torch.cat([features_high, adapted_features_mid, adapted_features_deep, adapted_features_spectral], dim=1)  # (928, 15, 15)
        weights2 = self.fusion_conv2(fusion_input2)  # (4, 15, 15)
        w_high, w_mid, w_deep, w_early = weights2[:, 0:1, :, :], weights2[:, 1:2, :, :], weights2[:, 2:3, :, :], weights2[:, 3:4, :, :]  # Split into four masks
        fused_features_high = (w_high * features_high) + (w_mid * adapted_features_mid) + (w_deep * adapted_features_deep) + (w_early * adapted_features_spectral)  # (232, 15, 15)

        out = self.classifier(fused_features_high)  # Classifier (19)
        return out
    
    # Override optimizer configuration for CustomModel
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)                
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)                                                  
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# Custom Model Version 9
class CustomModelV9(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        dummy_model = nn.Identity()
        super(CustomModelV9, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
        
        # -- Enhanced Spectral Mixing and Initial Feature Extraction --
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=36, kernel_size=3, stride=1, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(num_features=36),
            SpectralAttention(in_channels=36),  
            nn.MaxPool2d(kernel_size=2, stride=2)  # (120x120 -> 60x60)
        )

        # -- Block 1: Enhanced Low-Level Feature Extraction --
        block1_downsample = nn.Sequential(
            nn.Conv2d(36, 52, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=52)
        )
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=36, out_channels=36, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=36),
            nn.GELU(),
            WideBottleneck(in_channels=36, out_channels=26, stride=1, downsample=block1_downsample, widen_factor=2),
            SpectralAttention(in_channels=52),  
            nn.Dropout(p=ModuleConfig.dropout_rt),
            CoordinateAttention(in_channels=52, reduction=16),  
            nn.Dropout(p=ModuleConfig.dropout_rt)
        )

        # -- Block 2: Improved Mid-Level Features with Spatial Reduction --
        block2_downsample = nn.Sequential(
            nn.Conv2d(52, 104, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=104)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=52, out_channels=52, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=52),
            nn.GELU(),
            WideBottleneck(in_channels=52, out_channels=52, stride=1, downsample=block2_downsample, widen_factor=2),
            ECA(in_channels=104),  
            nn.Dropout(p=ModuleConfig.dropout_rt)
        )

        # -- Block 3: Enhanced Deep Feature Extraction --
        block3_downsample = nn.Sequential(
            nn.Conv2d(104, 172, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=172)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=104, out_channels=104, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=104),
            nn.GELU(),
            WideBottleneck(in_channels=104, out_channels=86, stride=1, downsample=block3_downsample, widen_factor=2),
            SE(in_channels=172, kernel_size=1),  
            nn.Dropout(p=ModuleConfig.dropout_rt * 1.5)
        )

        # -- Improved Skip Connection Adapters --
        # Skip from Block 1 to Fusion 1
        self.skip_adapter = nn.Sequential( 
            nn.Conv2d(in_channels=52, out_channels=172, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(172),  
            nn.AvgPool2d(kernel_size=4, stride=4)  # (60x60 -> 15x15)
        )
        # Skip from Spectral Mixer to Fusion 2
        self.skip_adapter_spectral = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=236, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(236), 
            nn.AvgPool2d(kernel_size=4, stride=4)  # (60x60 -> 15x15)
        )
        # Skip from Block 2 to Fusion 2
        self.skip_adapter_mid = nn.Sequential(
            nn.Conv2d(in_channels=104, out_channels=236, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(236),  
            nn.AvgPool2d(kernel_size=2, stride=2)  # (30x30 -> 15x15)
        )
        # Skip from Block 3 to Fusion 2
        self.skip_adapter_deep = nn.Sequential(
            nn.Conv2d(in_channels=172, out_channels=236, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(236)  
        )
        
        # -- Enhanced Fusion Mechanisms --
        # Fusion 1: Dynamic weighted sum of deep and low features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(344, 2, kernel_size=1, bias=False),  # 344 = 172 + 172
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )
        
        # Fusion 2: Combines all feature streams with dynamic weights
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(944, 4, kernel_size=1, bias=False),  # 944 = 236*4
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )

        # -- Block 4: Advanced High-Level Feature Refinement -- 
        block4_downsample = nn.Sequential(
            nn.Conv2d(172, 236, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=236)
        )
        self.block4 = nn.Sequential(
            MultiScaleBlock(in_channels=172, out_channels=172, kernel_size=3, stride=1, groups=4, bias=False),  
            nn.BatchNorm2d(num_features=172),
            nn.GELU(),  
            WideBottleneck(in_channels=172, out_channels=118, stride=1, downsample=block4_downsample, widen_factor=2),
            CBAM(in_channels=236),  
            nn.Dropout(p=ModuleConfig.dropout_rt * 1.5)
        )

        # -- Optimized Classifier with Balanced Information Flow --
        self.classifier = nn.Sequential(
            ECA(in_channels=236),  
            nn.Conv2d(in_channels=236, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=ModuleConfig.dropout_rt * 2),  
            nn.Linear(in_features=128, out_features=num_classes)
        )
    
    # Improved forward pass with enhanced integration strategy
    def forward(self, x):
        x = self.spectral_mixer(x)  # (36, 60, 60)
        features_low = self.block1(x)  # (52, 60, 60)
        features_mid = self.block2(features_low)  # (104, 30, 30)
        features_deep = self.block3(features_mid)  # (172, 15, 15)

        # Skip Connection Adapters with normalized features
        adapted_features_low = self.skip_adapter(features_low)  # (172, 15, 15)
        adapted_features_mid = self.skip_adapter_mid(features_mid)  # (236, 15, 15)
        adapted_features_deep = self.skip_adapter_deep(features_deep)  # (236, 15, 15)
        adapted_features_spectral = self.skip_adapter_spectral(x)  # (236, 15, 15)

        # Fusion 1: Deep + Low features with learned weights
        fused_input = torch.cat([features_deep, adapted_features_low], dim=1)  # (344, 15, 15)
        weights = self.fusion_conv(fused_input)  # (2, 15, 15)
        w_deep, w_low = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        
        # Improved fusion with learned complementary features
        fused_features = (w_deep * features_deep) + (w_low * adapted_features_low)  # (172, 15, 15)

        # Block 4: High-level processing
        features_high = self.block4(fused_features)  # (236, 15, 15)

        # Fusion 2: All feature streams with improved weighting
        fusion_input2 = torch.cat([features_high, adapted_features_mid, adapted_features_deep, adapted_features_spectral], dim=1)  # (944, 15, 15)
        weights2 = self.fusion_conv2(fusion_input2)  # (4, 15, 15)
        w_high, w_mid, w_deep, w_early = weights2[:, 0:1, :, :], weights2[:, 1:2, :, :], weights2[:, 2:3, :, :], weights2[:, 3:4, :, :]
        
        # Dynamic weighted fusion with rescaling to maintain signal strength
        fusion_weights_sum = w_high + w_mid + w_deep + w_early + 1e-5  # Avoid division by zero
        normalized_w_high = w_high / fusion_weights_sum
        normalized_w_mid = w_mid / fusion_weights_sum
        normalized_w_deep = w_deep / fusion_weights_sum
        normalized_w_early = w_early / fusion_weights_sum
        
        # Final feature fusion with normalized weights 
        fused_features_high = (normalized_w_high * features_high) + \
                            (normalized_w_mid * adapted_features_mid) + \
                            (normalized_w_deep * adapted_features_deep) + \
                            (normalized_w_early * adapted_features_spectral)  # (236, 15, 15)

        # Classification
        out = self.classifier(fused_features_high)  # (num_classes)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=ModelConfig.learning_rate, 
            weight_decay=ModelConfig.weight_decay,
            eps=1e-8  # For numerical stability
        )    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=ModelConfig.lr_factor, 
            patience=ModelConfig.lr_patience,
            min_lr=1e-7  # Prevent learning rate from becoming too small
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
# -- State-Of-The-Art Models (adapted from torch) --
# ResNet18 Model
class ResNet18(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None

        resnet_model = resnet18(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify first convolution layer to accept multiple channels
            original_conv1 = resnet_model.conv1
            resnet_model.conv1 = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )
            nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        super(ResNet18, self).__init__(resnet_model, num_classes, class_weights, in_channels, main_path)

# ResNet50 Model
class ResNet50(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        resnet_model = resnet50(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify first convolution layer to accept multiple channels
            original_conv1 = resnet_model.conv1
            resnet_model.conv1 = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )
            nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        super(ResNet50, self).__init__(resnet_model, num_classes, class_weights, in_channels, main_path)

class ResNet101(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        resnet_model = resnet101(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify first convolution layer to accept multiple channels
            original_conv1 = resnet_model.conv1
            resnet_model.conv1 = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )
            nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        super(ResNet101, self).__init__(resnet_model, num_classes, class_weights, in_channels, main_path)

class ResNet152(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        resnet_model = resnet152(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify first convolution layer to accept multiple channels
            original_conv1 = resnet_model.conv1
            resnet_model.conv1 = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias,
            )
            nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        super(ResNet152, self).__init__(resnet_model, num_classes, class_weights, in_channels, main_path)

# VGG16 Model
class VGG16(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        vgg_model = vgg16(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify the first convolutional layer
            original_conv1 = vgg_model.features[0] 
            vgg_model.features[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=(original_conv1.bias is not None)  
            )
            nn.init.kaiming_normal_(vgg_model.features[0].weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer in the classifier
        vgg_model.classifier[6] = nn.Linear(
            in_features=vgg_model.classifier[6].in_features,
            out_features=num_classes,
        )

        super(VGG16, self).__init__(vgg_model, num_classes, class_weights, in_channels, main_path)

# VGG19 Model
class VGG19(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        vgg_model = vgg19(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify the first convolutional layer
            original_conv1 = vgg_model.features[0]  # Access the first Conv2d layer
            vgg_model.features[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=(original_conv1.bias is not None)  
            )
            nn.init.kaiming_normal_(vgg_model.features[0].weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer in the classifier
        vgg_model.classifier[6] = nn.Linear(
            in_features=vgg_model.classifier[6].in_features,
            out_features=num_classes,
        )

        super(VGG19, self).__init__(vgg_model, num_classes, class_weights, in_channels, main_path)

# EfficientNetB0 Model
class EfficientNetB0(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == 'EfficientNetB0_Weights.DEFAULT':
            model_weights = EfficientNet_B0_Weights.DEFAULT
        elif model_weights == "None":
            model_weights = None
        efficientnet_model = efficientnet_b0(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify the first convolutional layer
            original_conv1 = efficientnet_model.features[0][0] 
            efficientnet_model.features[0][0] = nn.Conv2d(
                in_channels=in_channels,  # Adjust for custom input channels
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
            nn.init.kaiming_normal_(efficientnet_model.features[0][0].weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        efficientnet_model.classifier[1] = nn.Linear(
            efficientnet_model.classifier[1].in_features, num_classes
        )

        super(EfficientNetB0, self).__init__(efficientnet_model, num_classes, class_weights, in_channels, main_path)

# EfficientNetV2 Model
class EfficientNetV2(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        efficientnet_model = efficientnet_v2_m(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify the first convolutional layer
            original_conv1 = efficientnet_model.features[0][0]  
            efficientnet_model.features[0][0] = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
            nn.init.kaiming_normal_(efficientnet_model.features[0][0].weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        efficientnet_model.classifier[1] = nn.Linear(
            efficientnet_model.classifier[1].in_features, num_classes
        )

        super(EfficientNetV2, self).__init__(efficientnet_model, num_classes, class_weights, in_channels, main_path)

# DenseNet121 Model
class DenseNet121(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights == "None":
            model_weights = None
        densenet_model = densenet121(weights=model_weights)

        if in_channels == 3:
            pass
        else:
            # Modify the first convolutional layer to accept custom number of input channels
            original_conv1 = densenet_model.features[0] 
            densenet_model.features[0] = nn.Conv2d(
                in_channels=in_channels,  
                out_channels=original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
            nn.init.kaiming_normal_(densenet_model.features[0].weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        densenet_model.classifier = nn.Linear(
            densenet_model.classifier.in_features, num_classes
        )

        super(DenseNet121, self).__init__(densenet_model, num_classes, class_weights, in_channels, main_path)   

# Swin Transformer Model
class SwinTransformer(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Swin Transformer model with the specified input size
        swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(SwinTransformer, self).__init__(swin_model, num_classes, class_weights, in_channels, main_path)

# Vision Transformer Model
class VitTransformer(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights, main_path):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Vision Transformer model with the specified input size
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(VitTransformer, self).__init__(vit_model, num_classes, class_weights, in_channels, main_path)
