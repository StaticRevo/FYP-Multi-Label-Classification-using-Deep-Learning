from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from models.BaseModel import BaseModel

class BigEarthNetEfficientNetB0ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        efficientnet_model = efficientnet_b0(weights=model_weights)

        # Modify the first convolutional layer
        original_conv1 = efficientnet_model.features[0][0]  # Access the first Conv2d layer
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

        super(BigEarthNetEfficientNetB0ModelTIF, self).__init__(efficientnet_model, num_classes, class_weights)
