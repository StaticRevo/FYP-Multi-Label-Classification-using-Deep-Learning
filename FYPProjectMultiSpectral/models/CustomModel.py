from torch import nn
import torchvision.models as models
from models.BaseModel import BaseModel
from torchsummary import summary
import torch

class CustomModel(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        custom_model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 30 * 30, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        super(CustomModel, self).__init__(custom_model, num_classes, class_weights)


