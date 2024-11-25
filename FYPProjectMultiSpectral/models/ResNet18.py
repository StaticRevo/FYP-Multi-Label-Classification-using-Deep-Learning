from torch import nn
import torchvision.models as models
from BaseModel import BaseModel

class BigEarthNetResNet18ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes):
        # Load the ResNet-18 model
        resnet_model = models.resnet18(weights=None)

        # Modify first convolution layer to accept 3 channels
        original_conv1 = resnet_model.conv1
        resnet_model.conv1 = nn.Conv2d(
            in_channels=3,  
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )
        nn.init.kaiming_normal_(resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final fully connected layer
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

        # Call the parent class constructor with the modified model
        super(BigEarthNetResNet18ModelTIF, self).__init__(resnet_model, num_classes, class_weights)
