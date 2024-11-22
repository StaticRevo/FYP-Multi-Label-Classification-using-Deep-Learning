# Core Library modules
import os  # Operating system interactions, such as reading and writing files.
import numpy as np  # Array operations and computations.

import torch  # Core PyTorch library for tensor computations.
import torch.nn as nn  # Neural network module for defining layers and architectures.
from torch.nn import functional as F  # Functional module for defining functions and loss functions.
import torch.optim as optim  # Optimizer module for training models (SGD, Adam, etc.).
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for PyTorch.
import torchvision.models as models  # Pretrained models for transfer learning.
from torchvision.models import VGG16_Weights  # Import VGG16_Weights for pretrained weights.
from torchsummary import summary  # Model summary.
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
)  # Classification metrics.
from torchviz import make_dot  # Model visualization.
import pytorch_lightning as pl  # Training management.

# Custom modules
from FYPProjectRGB.preprocessing.config import DatasetConfig  # Import the dataclasses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BigEarthNetVGG16Model(pl.LightningModule):
    def __init__(self, weights=None):
        super(BigEarthNetVGG16Model, self).__init__()
        # Load the VGG-16 model
        self.model = models.vgg16(weights=weights)
        # Modify the final layer to output 19 classes
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, DatasetConfig.num_classes)
        # Addition of a sigmoid activation function for multi-label classification
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

model = BigEarthNetVGG16Model()
summary(model.model, input_size=(DatasetConfig.band_channels, DatasetConfig.img_size, DatasetConfig.img_size))

# Visualize the model architecture
sample_input = torch.randn(1, DatasetConfig.band_channels, DatasetConfig.img_size, DatasetConfig.img_size).to(device)
output = model(sample_input)  # Pass the sample input through the model
dot = make_dot(output, params=dict(model.named_parameters()))  # Visualize the computational graph
output_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\models\vgg16'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'model_architecture')
dot.format = 'png'
dot.render(output_path)

print(f"Model architecture saved to {output_path}.png")