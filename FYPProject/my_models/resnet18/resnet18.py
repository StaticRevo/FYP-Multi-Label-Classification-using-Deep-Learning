# Core Library modules
import os  # Operating system interactions, such as reading and writing files.
import shutil  # High-level file operations like copying and moving files.
import random  # Random number generation for various tasks.
import textwrap  # Formatting text into paragraphs of a specified width.
import warnings  # Warning control context manager.
import zipfile  # Work with ZIP archives.
import platform  # Access to underlying platform’s identifying data.
import itertools  # Functions creating iterators for efficient looping.
from dataclasses import dataclass  # Class decorator for adding special methods to classes.

# PyTorch and Deep Learning Libaries
import torch  # Core PyTorch library for tensor computations.
import torch.nn as nn  # Neural network module for defining layers and architectures.
from torch.nn import functional as F  # Functional module for defining functions and loss functions.
import torch.optim as optim  # Optimizer module for training models (SGD, Adam, etc.).
from torch.utils.data import Dataset, DataLoader, Subset, random_split  # Data handling and batching
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for PyTorch.
import torchvision  # PyTorch's computer vision library.
from torchvision import datasets, transforms  # Image datasets and transformations.
import torchvision.datasets as datasets  # Specific datasets for vision tasks.
import torchvision.transforms as transforms  # Transformations for image preprocessing.
from torchvision.utils import make_grid  # Grid for displaying images.
import torchvision.models as models  # Pretrained models for transfer learning.
from torchvision.datasets import MNIST, EuroSAT  # Standard datasets.
import torchvision.transforms.functional as TF  # Functional transformations.
from torchvision.models import ResNet18_Weights  # ResNet-18 model with pretrained weights.
from torchsummary import summary  # Model summary.
import torchmetrics  # Model evaluation metrics.
from torchmetrics import MeanMetric, Accuracy  # Accuracy metrics.
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
)  # Classification metrics.
from torchviz import make_dot  # Model visualization.
from torchvision.ops import sigmoid_focal_loss  # Focal loss for class imbalance.
from torchcam.methods import GradCAM  # Grad-CAM for model interpretability.
from torchcam.utils import overlay_mask  # Overlay mask for visualizations.
import pytorch_lightning as pl  # Training management.
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback  # Callbacks.
from pytorch_lightning.loggers import TensorBoardLogger  # Logger for TensorBoard.

# Geospatial Data Processing Libraries
import rasterio  # Reading and writing geospatial raster data.
from rasterio.warp import calculate_default_transform, reproject  # Reprojection and transformation.
from rasterio.enums import Resampling  # Resampling for raster resizing.
from rasterio.plot import show  # Visualization of raster data.

# Data Manipulation, Analysis and Visualization Libraries
import pandas as pd  # Data analysis and manipulation.
import numpy as np  # Array operations and computations.
from sklearn.metrics import confusion_matrix, accuracy_score  # Evaluation metrics.
import matplotlib.pyplot as plt  # Static and interactive plotting.
import seaborn as sns  # High-level interface for statistical graphics.

# Utility Libraries
from tqdm import tqdm  # Progress bar for loops.
from PIL import Image  # Image handling and manipulation.
import ast  # Parsing Python code.
import requests  # HTTP requests.
import zstandard as zstd  # Compression and decompression.
from collections import Counter  # Counting hashable objects.
import certifi  # Certificates for HTTPS.
import ssl  # Secure connections.
import urllib.request  # URL handling.
import kaggle  # Kaggle API for datasets.
from IPython.display import Image  # Display images in notebooks.
from pathlib import Path # File system path handling.
from typing import Dict, List, Tuple  # Type hints.
import sys  # System-specific parameters and functions.
import time # Time access and conversions.

# Custom Libraries

@dataclass
class DatasetConfig:
    dataset_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%'
    combined_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImages'
    metadata_path: str =r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\metadata_50_percent.csv'
    metadata_csv = pd.read_csv(metadata_path)
    img_size: int = 120
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    num_classes: int = 19
    band_channels: int = 3 #13
    valid_pct: float = 0.1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BigEarthNetSubsetModel(pl.LightningModule):
    def __init__(self):
        super(BigEarthNetSubsetModel, self).__init__()
        # Load the ResNet-18 model
        self.model = models.resnet18(weights=None)
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
    

model = BigEarthNetSubsetModel()
summary(model.model, input_size=(DatasetConfig.band_channels, DatasetConfig.img_size, DatasetConfig.img_size))



# Create a sample input tensor
sample_input = torch.randn(1, DatasetConfig.band_channels, DatasetConfig.img_size, DatasetConfig.img_size).to(device)

# Pass the sample input through the model
output = model(sample_input)

# Visualize the computational graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Specify the directory path where you want to save the image
output_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\models\resnet18'

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the visualization as an image file within the specified directory
output_path = os.path.join(output_dir, 'model_architecture')
dot.format = 'png'
dot.render(output_path)

print(f"Model architecture saved to {output_path}.png")