from torch import nn
from config.config import DatasetConfig, ModelConfig, ModuleConfig
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from models.base_model import BaseModel
from models.modules import *
from torchsummary import summary
import timm

# Custom Model
class CustomModel(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        custom_model = nn.Sequential(
            # -- Block 1 --
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Residual Block (64->64)
            ResidualBlock(in_channels=64, out_channels=64, stride=1),

            # -- Block 2 --
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Residual Block (128->128) and SE Module
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
            SE(in_channels=128, config=ModuleConfig),

            # -- BLock 3 -- 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual Block (128->256) and ChannelAttention Module
            ResidualBlock(in_channels=256, out_channels=256, stride=1),
            ChannelAttention(in_channels=256, reduction_ratio=16),

            # -- Block 4 -- 
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Residual Block (512->512) and ECA Module
            ResidualBlock(in_channels=512, out_channels=512, stride=1),

            # Global Pool and Classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(ModelConfig.dropout),
            nn.Linear(512, num_classes)
        )

        super(CustomModel, self).__init__(custom_model, num_classes, class_weights, in_channels)
        
# ResNet18 Model
class BigEarthNetResNet18ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
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

        super(BigEarthNetResNet18ModelTIF, self).__init__(resnet_model, num_classes, class_weights, in_channels)

# ResNet50 Model
class BigEarthNetResNet50ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        resnet_model = resnet50(weights=model_weights)

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

        super(BigEarthNetResNet50ModelTIF, self).__init__(resnet_model, num_classes, class_weights, in_channels)

# VGG16 Model
class BigEarthNetVGG16ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        vgg_model = vgg16(weights=model_weights)

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

        super(BigEarthNetVGG16ModelTIF, self).__init__(vgg_model, num_classes, class_weights, in_channels )

# VGG19 Model
class BigEarthNetVGG19ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        vgg_model = vgg19(weights=model_weights)

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

        super(BigEarthNetVGG19ModelTIF, self).__init__(vgg_model, num_classes, class_weights, in_channels)

# EfficientNetB0 Model
class BigEarthNetEfficientNetB0ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        if model_weights == 'EfficientNetB0_Weights.DEFAULT':
            model_weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet_model = efficientnet_b0(weights=model_weights)

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

        super(BigEarthNetEfficientNetB0ModelTIF, self).__init__(efficientnet_model, num_classes, class_weights, in_channels)

# EfficientNetV2 Model
class BigEarthNetEfficientNetV2MModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        efficientnet_model = efficientnet_v2_m(weights=model_weights)

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

        super(BigEarthNetEfficientNetV2MModelTIF, self).__init__(efficientnet_model, num_classes, class_weights, in_channels)

# DenseNet121 Model
class BigEarthNetDenseNet121ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        densenet_model = densenet121(weights=model_weights)

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

        super(BigEarthNetDenseNet121ModelTIF, self).__init__(densenet_model, num_classes, class_weights, in_channels)   

# Swin Transformer Model
class BigEarthNetSwinTransformerModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Swin Transformer model with the specified input size
        swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(BigEarthNetSwinTransformerModelTIF, self).__init__(swin_model, num_classes, class_weights, in_channels)

# Vision Transformer Model
class BigEarthNetVitTransformerModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Vision Transformer model with the specified input size
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(BigEarthNetVitTransformerModelTIF, self).__init__(vit_model, num_classes, class_weights, in_channels)
