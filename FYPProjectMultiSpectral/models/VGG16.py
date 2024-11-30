from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from models.BaseModel import BaseModel

class BigEarthNetVGG16ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels):
        # Load the pretrained VGG16 model
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)

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

        # Call the parent class constructor with the modified model
        super(BigEarthNetVGG16ModelTIF, self).__init__(vgg_model, num_classes, class_weights)
