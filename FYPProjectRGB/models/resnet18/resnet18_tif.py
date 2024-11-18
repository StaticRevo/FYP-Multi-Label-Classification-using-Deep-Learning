# Core Library modules
import os  # Operating system interactions, such as reading and writing files.
import numpy as np  # Array operations and computations.

import torch  # Core PyTorch library for tensor computations.
import torch.nn as nn  # Neural network module for defining layers and architectures.
from torch.nn import functional as F  # Functional module for defining functions and loss functions.
import torch.optim as optim  # Optimizer module for training models (SGD, Adam, etc.).
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for PyTorch.
import torchvision.models as models  # Pretrained models for transfer learning.
from torchvision.models import ResNet18_Weights  # Import ResNet18_Weights for pretrained weights.
from torchsummary import summary  # Model summary.
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
)  # Classification metrics.
from torchviz import make_dot  # Model visualization.
import pytorch_lightning as pl  # Training management.

# Custom modules
from config import DatasetConfig  # Import the dataclasses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BigEarthNetResNet18ModelTIF(pl.LightningModule):
    def __init__(self):
        super(BigEarthNetResNet18ModelTIF, self).__init__()
        # Load the ResNet-18 model
        self.model = models.resnet18(weights=None)

        # Modify the first convolutional layer to accept 13 input channels
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=13,  # Update to match the number of bands in the dataset
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize weights for the new channels (copy pretrained weights for 3 channels and random for the rest)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Modify the final layer to output 19 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, DatasetConfig.num_classes)
        # Addition of a sigmoid activation function for muylti-label classification
        self.sigmoid = nn.Sigmoid()
        # Passing the model to the GPU
        self.model.to(device)

        # Accuracy metrics
        self.train_acc = MultilabelAccuracy(num_labels=DatasetConfig.num_classes)
        self.val_acc = MultilabelAccuracy(num_labels=DatasetConfig.num_classes)
        self.test_acc = MultilabelAccuracy(num_labels=DatasetConfig.num_classes)

        # Recall metrics
        self.train_recall = MultilabelRecall(num_labels=DatasetConfig.num_classes)
        self.val_recall = MultilabelRecall(num_labels=DatasetConfig.num_classes)
        self.test_recall = MultilabelRecall(num_labels=DatasetConfig.num_classes)

        # Precision metrics
        self.train_precision = MultilabelPrecision(num_labels=DatasetConfig.num_classes)
        self.val_precision = MultilabelPrecision(num_labels=DatasetConfig.num_classes)
        self.test_precision = MultilabelPrecision(num_labels=DatasetConfig.num_classes)

        # F1 Score metrics
        self.train_f1 = MultilabelF1Score(num_labels=DatasetConfig.num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=DatasetConfig.num_classes)
        self.test_f1 = MultilabelF1Score(num_labels=DatasetConfig.num_classes)

        #torch.summary(self.model, (DatasetConfig.band_channels, ModelConfig.img_size, ModelConfig.img_size))

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
    
    def cross_entropy_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.train_acc(logits, y)
        recall = self.train_recall(logits, y)
        f1 = self.train_f1(logits, y)
        precision = self.train_precision(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.val_acc(logits, y)
        recall = self.val_recall(logits, y)
        f1 = self.val_f1(logits, y)
        precision = self.val_precision(logits, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = self.test_acc(logits, y)
        recall = self.test_recall(logits, y)
        f1 = self.test_f1(logits, y)
        precision = self.test_precision(logits, y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_epoch=True, prog_bar=True)
        self.log('test_precision', precision, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('train_recall_epoch', self.train_recall.compute())
        self.log('train_f1_epoch', self.train_f1.compute())
        self.log('train_precision_epoch', self.train_precision.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute())
        self.log('val_recall_epoch', self.val_recall.compute())
        self.log('val_f1_epoch', self.val_f1.compute())
        self.log('val_precision_epoch', self.val_precision.compute())
    
    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.test_acc.compute())
        self.log('test_recall_epoch', self.test_recall.compute())
        self.log('test_f1_epoch', self.test_f1.compute())
        self.log('test_precision_epoch', self.test_precision.compute())