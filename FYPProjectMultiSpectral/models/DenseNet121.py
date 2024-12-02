from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from models.BaseModel import BaseModel

class BigEarthNetDenseNet121ModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        # Load the DenseNet-121 model
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

        super(BigEarthNetDenseNet121ModelTIF, self).__init__(densenet_model, num_classes, class_weights)
