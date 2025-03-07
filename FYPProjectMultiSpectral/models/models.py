# Third-party imports
from torch import nn
from torchsummary import summary
import timm
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, 
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
        dummy_model = nn.Identity() # Dummy model for custom architecture to pass to the base model
        super(CustomModel, self).__init__(dummy_model, num_classes, class_weights, in_channels, main_path)
    
        # Spectral Mixing and Initial Feature Extraction
        self.spectral_mixer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (in_channels->32)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling Layer - (120x120 -> 60x60)
        )
        # -- Block 1 --
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (32->64)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), # Batch Normalization Layer 
            nn.GELU(), # GELU Activation Function
            ResidualBlock(in_channels=128, out_channels=128, stride=1), # Residual Block (64->64)  
            SpectralAttention(in_channels=128), # SpectralAttention Module (64->64)
            CoordinateAttention(in_channels=128, reduction=16), # CoordinateAttention Module (64->64)
        )
        # -- Block 2 --
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, 
                                   dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (64->128)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            MultiScaleBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, groups=1, 
                            bias=True, padding_mode='zeros'), # MultiScaleBlock (128->128)
            ResidualBlock(in_channels=256, out_channels=256, stride=1), # Residual Block (128->128) 
            ECA(in_channels=256), # ECA Module
        )
        # -- BLock 3 --
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, 
                                  dilation=1, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (128->256)
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            MultiScaleBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, groups=1, bias=True, padding_mode='zeros'), # MultiScaleBlock (256->256)
            ResidualBlock(in_channels=512, out_channels=512, stride=1), # Residual Block (256->256) 
            SE(in_channels=512, kernel_size=1), # Squeeze and Excitation Module
        )
        self.skip_adapter = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'), # Convolutional Layer (64->256)
            nn.AvgPool2d(kernel_size=4, stride=4) # Average Pooling Layer (60x60 -> 15x15)
        )

        # -- Block 4 -- 
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=2, 
                                   dilation=2, bias=False, padding_mode='zeros'), # Depthwise Separable Convolution (256->512)
            nn.BatchNorm2d(num_features=1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            ResidualBlock(in_channels=1024, out_channels=1024, stride=1), # Residual Block (512->512)
            DualAttention(in_channels=1024, kernel_size=7, stride=1), # DualAttention Module (Spectal+Spatial Attention Modules)
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

        # Match spatial dimensions
        adapted_features_low = self.skip_adapter(features_low) # Adapt low-level features to match deep features (512, 60, 60)

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
