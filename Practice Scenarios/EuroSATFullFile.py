# Core Python Libraries
import os  # Operating system interactions, such as reading and writing files.
import shutil  # High-level file operations like copying and moving files.
import random  # Random number generation for various tasks.
import textwrap  # Formatting text into paragraphs of a specified width.
import warnings  # Warning control context manager.
import zipfile  # Work with ZIP archives.
import platform  # Access to underlying platform’s identifying data.
import itertools  # Functions creating iterators for efficient looping.
from dataclasses import dataclass  # Class decorator for adding special methods to classes.

# PyTorch-related Libraries (Deep Learning)
import torch  # Core PyTorch library for tensor computations.
import torch.nn as nn  # Neural network module for defining layers and architectures.
import torch.optim as optim  # Optimizer module for training models (SGD, Adam, etc.).
from torch.utils.data import Dataset, DataLoader, Subset, random_split  # Dataset and DataLoader for managing and batching data.
import torchvision # PyTorch's computer vision library.
from torchvision import datasets, transforms  # Datasets and transformations for image processing.
import torchvision.datasets as datasets  # Datasets for computer vision tasks.
import torchvision.transforms as transforms  # Transformations for image preprocessing.
from torchvision.utils import make_grid  # Make grid for displaying images.
import torchvision.models as models  # Pretrained models for transfer learning.
import torchvision.transforms.functional as TF  # Functional transformations for image preprocessing.
import torchsummary # PyTorch model summary for Keras-like model summary.
from torchvision.ops import sigmoid_focal_loss  # Focal loss for handling class imbalance in object detection.
from torchmetrics import MeanMetric  # Intersection over Union (IoU) metric for object detection.
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy  # Multilabel classification metrics.

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Geospatial Data Processing Libraries
import rasterio  # Library for reading and writing geospatial raster data.
from rasterio.warp import calculate_default_transform, reproject  # Reprojection and transformation functions.
from rasterio.enums import Resampling  # Resampling methods used for resizing raster data.
from rasterio.plot import show  # Visualization of raster data.

# Data Manipulation and Analysis Libraries
import pandas as pd  # Data analysis and manipulation library for DataFrames and CSVs.
import numpy as np  # Numpy for array operations and numerical computations.
from sklearn.metrics import confusion_matrix, accuracy_score  # Evaluation metrics for classification models.

# Visualization Libraries
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations.
import seaborn as sns  # High-level interface for drawing attractive statistical graphics.

# Utilities
from tqdm import tqdm  # Progress bar for loops and processes.
from PIL import Image  # Image handling, opening, manipulating, and saving.
import ast  # Abstract Syntax Trees for parsing Python code.
import requests  # HTTP library for sending requests.
import zstandard as zstd  # Zstandard compression for fast compression and decompression.
from collections import Counter # Counter for counting hashable objects.
import certifi  # Certificates for verifying HTTPS requests.
import ssl  # Secure Sockets Layer for secure connections.
import urllib.request  # URL handling for requests.
import kaggle # Kaggle API for downloading datasets.
import zipfile # Work with ZIP archives.
import timm # PyTorch Image Models for transfer learning.


# Set seed for reproducibility
# Setting a seed ensures that the results are consistent and reproducible each time the code is run.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check if GPU is enabled
# PyTorch allows for the use of GPU to speed up training. Here we check if a GPU is available and set the device accordingly.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    # If a GPU is available, print the name of the GPU
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class EuroSAT(Dataset):
    # Initialize the EuroSAT dataset class
    def __init__(self, dataset, transform=None):
        self.dataset = dataset  # Store the dataset
        self.transform = transform  # Store the optional transformation function

    def __getitem__(self, index):
        image, label = self.dataset[index]  # Get the image and label from the dataset at the specified index

        if self.transform:
            image = self.transform(image)  # Apply the transformation if provided

        return image, label  # Return the image and label

    def __len__(self):
        return len(self.dataset)  # Return the length of the dataset

input_size = 224
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Define the directory where the dataset images are stored
data_dir = r'D:\Datasets\2019EuroSAT\Eurosat-Dataset-RGB'

# Create an ImageFolder dataset instance for the specified directory
dataset = datasets.ImageFolder(data_dir)

# Extract and print the class names from the dataset
class_names = dataset.classes
print(f"Class names: {class_names}")

# Print the total number of classes in the dataset
print(f"Total number of classes: {len(class_names)}")

# Define the proportion of the dataset to be used for training
train_size = 0.8

# Generate a list of indices for all data points in the dataset
indices = list(range(len(dataset)))

# Calculate the split index for training and testing data
split = int(train_size * len(dataset))

# Shuffle the indices to randomize the dataset split
np.random.shuffle(indices)

# Split the indices into training and testing indices
train_indices, test_indices = indices[:split], indices[split:]

# Define transformations for training and testing data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create subsets of the dataset for training and testing using the indices
train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

# Apply transformations to the subsets
train_data.dataset.transform = train_transform
test_data.dataset.transform = test_transform

# Define the batch size and the number of workers for data loading
batch_size = 16
num_workers = 2

# Create DataLoader instances for training and testing data
# Shuffle training data to ensure randomness during training
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# Do not shuffle testing data; it's used for evaluation
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)



# Load the pre-trained ResNet-50 model
# The 'pretrained=True' argument loads a model that has been pre-trained on ImageNet
model = models.resnet50(pretrained=True)

# Replace the final fully connected layer (classifier) of the model
# model.fc.in_features gives the number of input features to the final layer
# len(class_names) is the number of output classes in our specific task
# We create a new linear layer with the same input features but with an output size equal to the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

# Move the model to the specified device (CPU or GPU)
# 'device' should be a torch device object (e.g., torch.device('cuda') or torch.device('cpu'))
model = model.to(device)

# Print a summary of the model's architecture
# The summary function shows the model's layers, output shapes, and number of parameters
# (3, 224, 224) indicates the input shape of the images (3 channels, 224x224 pixels)
torchsummary.summary(model, (3, 224, 224))



# Define your custom model class
class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        enet_out_size = 1280
        # Make classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

model = CustomModel(num_classes=len(class_names))

# Define the loss function
# CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
# It is useful for classification problems.
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
# SGD stands for Stochastic Gradient Descent. It updates the parameters
# of the model based on the gradients computed during backpropagation.
# Here, we use the parameters of the model and set the learning rate (lr) to 0.001.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(model, dataloader, criterion, optimizer, device):
    # Set the model to training mode
    model.train()

    # Initialize running totals for loss and correct predictions
    running_loss = 0.0
    running_corrects = 0

    # Loop over data batches in the dataloader
    for inputs, labels in tqdm(dataloader):
        # Move inputs and labels to the specified device (e.g., GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients to avoid accumulation
        optimizer.zero_grad()

        # Forward pass: compute outputs by passing inputs to the model
        outputs = model(inputs)

        # Compute the loss between model outputs and actual labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters based on the computed gradients
        optimizer.step()

        # Get the predicted class with the highest score
        _, preds = torch.max(outputs, 1)

        # Update the running loss (scaled by the batch size)
        running_loss += loss.item() * inputs.size(0)

        # Update the count of correct predictions
        running_corrects += torch.sum(preds == labels.data)

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    # Return the average loss and accuracy for the epoch
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track loss, correct predictions, and store all predictions and labels
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over the data in the dataloader
        for inputs, labels in tqdm(dataloader):
            # Move inputs and labels to the specified device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform forward pass to get outputs from the model
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Get the predicted class by finding the index with the highest score
            _, preds = torch.max(outputs, 1)

            # Update the running loss and correct predictions count
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Store predictions and labels for later use (e.g., for metrics computation)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy over the entire dataset
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    # Return the computed metrics
    return epoch_loss, epoch_acc, all_labels, all_preds

# Initialize the best model weights to the current state of the model
best_model_wts = model.state_dict()

# Set the best loss to infinity initially so that any loss will be lower
best_loss = float('inf')

# Set the number of epochs to train the model
num_epochs = 10

# Loop over the dataset multiple times (each loop is one epoch)
for epoch in range(num_epochs):
    # Print the current epoch number
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Train the model and get the training loss and accuracy
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

    # Print the training loss and accuracy for the current epoch
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    # Evaluate the model on the validation set and get the loss, accuracy, all labels, and all predictions
    val_loss, val_acc, all_labels, all_preds = evaluate(model, test_loader, criterion, device)

    # Print the validation loss and accuracy for the current epoch
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # If the current validation loss is lower than the best loss seen so far
    if val_loss < best_loss:
        # Update the best loss to the current validation loss
        best_loss = val_loss
        # Save the model weights as the best model weights
        best_model_wts = model.state_dict()

# Load the best model weights into the model
model.load_state_dict(best_model_wts)

# Define the directory where the model will be saved
model_dir = "./drive/My Drive/Colab Notebooks/models/"

# Create the directory if it does not exist
os.makedirs(model_dir, exist_ok=True)

# Define the file path for saving the best model weights
model_file = os.path.join(model_dir, 'best_model.pth')

# Save the best model weights to the file
torch.save(model.state_dict(), model_file)

# Print a message indicating that the model has been saved
print(f'Model saved to {model_file}')

# Load a pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer to match the number of classes in your dataset
# The original ResNet-50 has 1000 output features for 1000 classes (ImageNet dataset)
# We replace it with a new linear layer that has 'len(class_names)' output features (number of classes in your dataset)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

# Load the trained model weights from a file
# 'model_file' is the path to the file containing the model weights
model.load_state_dict(torch.load(model_file))

# Move the model to the appropriate device (CPU or GPU)
# 'device' is a variable specifying whether to use CPU or GPU (e.g., device = torch.device('cuda') or device = torch.device('cpu'))
model = model.to(device)

# Set the model to evaluation mode
# This is important because it changes the behavior of some layers, like dropout and batch normalization, which should behave differently during training and evaluation
model.eval()


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # If normalization is set to True, normalize the confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Print a message indicating whether the confusion matrix is normalized or not
    print("Confusion matrix, without normalization" if not normalize else "Normalized confusion matrix")

    # Create a new figure with a specified size
    plt.figure(figsize=(10, 10))
    # Display the confusion matrix as an image
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # Set the title of the plot
    plt.title(title)
    # Add a color bar to the side of the plot
    plt.colorbar()
    # Set tick marks at each class index
    tick_marks = np.arange(len(classes))
    # Label the x-axis ticks with the class names, rotated 90 degrees
    plt.xticks(tick_marks, classes, rotation=90)
    # Label the y-axis ticks with the class names
    plt.yticks(tick_marks, classes)

    # Format the values in the confusion matrix as float with 2 decimals if normalized, otherwise as integers
    fmt = '.2f' if normalize else 'd'
    # Determine a threshold to change text color for better readability
    thresh = cm.max() / 2.
    # Iterate through each cell in the confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Place the text in the middle of each cell
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Adjust the layout for better fit
    plt.tight_layout()
    # Label the y-axis as 'True label'
    plt.ylabel('True label')
    # Label the x-axis as 'Predicted label'
    plt.xlabel('Predicted label')

# Calculate and plot the confusion matrix
# `all_labels` contains the true labels and `all_preds` contains the predicted labels
cm = confusion_matrix(all_labels, all_preds)
# `class_names` contains the list of class names
plot_confusion_matrix(cm, class_names, title='Confusion Matrix')

# Calculate the overall accuracy of the model
accuracy = accuracy_score(all_labels, all_preds)
# Print the overall accuracy
print(f'Overall accuracy: {accuracy:.4f}')


# Function to predict the class of an input image
def predict_image(image, model, class_names, device):
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():  # Disable gradient computation for faster inference
        # Move the image to the specified device (CPU or GPU) and make predictions
        outputs = model(image.to(device))
        # Get the predicted class by finding the index with the highest score
        _, preds = torch.max(outputs, 1)
    # Return the name of the predicted class
    return class_names[preds.item()]

# Predict on a single sample image
image_path = r'D:\Datasets\eurosat\2750\Forest\Forest_10.jpg'
image = Image.open(image_path)  # Open the image file
input_image = test_transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
pred_class = predict_image(input_image, model, class_names, device)  # Predict the class of the image
actual_class ='Forest '

# Display the image along with the predicted class
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title(f"Predicted class: {pred_class}\nActual class: {actual_class}")
plt.show()

# List of actual class names for the sample images
actual_classes = [
    "Industrial",
    "Highway",
    "Residential",
    "River"
]

# Predict on multiple sample images
sample_image_paths = [
    r'D:\Datasets\eurosat\2750\Industrial\Industrial_20.jpg',
    r'D:\Datasets\eurosat\2750\Highway\Highway_15.jpg',
    r'D:\Datasets\eurosat\2750\Residential\Residential_13.jpg',
    r'D:\Datasets\eurosat\2750\River\River_12.jpg'
]

for image_path, actual_class in zip(sample_image_paths, actual_classes):
    image = Image.open(image_path)  # Open the image file
    input_image = test_transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    pred_class = predict_image(input_image, model, class_names, device)  # Predict the class of the image

    # Display the image along with the predicted and actual class
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Predicted class: {pred_class}\nActual class: {actual_class}")
    plt.show()
