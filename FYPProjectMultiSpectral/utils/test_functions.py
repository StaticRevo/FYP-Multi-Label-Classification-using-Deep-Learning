import ast
import pandas as pd
import os
from PIL import Image
import torch
import rasterio
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm
from utils.helper_functions import get_labels_for_image, display_image

def collect_predictions_and_plot_confusion_matrix(model, data_module, DatasetConfig):
    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    # Add progress bar using tqdm
    for batch in tqdm(data_module.test_dataloader(), desc="Processing Batches"):
        inputs, labels = batch
        inputs = inputs.to(model.device)  # Move inputs to the same device as the model
        labels = labels.to(model.device)  # Move labels to the same device as the model
        preds = model(inputs).sigmoid() > 0.5  # Apply sigmoid and threshold at 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute multilabel confusion matrix
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    print("Multilabel Confusion Matrix:")
    print(mcm)

    # Aggregate confusion matrices into a single 19x19 matrix
    cm = np.zeros((DatasetConfig.num_classes, DatasetConfig.num_classes), dtype=int)
    for i in range(DatasetConfig.num_classes):
        cm[i, 0] = mcm[i, 0, 0]  # True Negative
        cm[i, 1] = mcm[i, 0, 1]  # False Positive
        cm[i, 2] = mcm[i, 1, 0]  # False Negative
        cm[i, 3] = mcm[i, 1, 1]  # True Positive

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DatasetConfig.class_labels, yticklabels=DatasetConfig.class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

