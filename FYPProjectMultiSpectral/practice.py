import numpy as np
import matplotlib.pyplot as plt
from config.config import DatasetConfig, clean_and_parse_labels
from utils.test_functions import *
from utils.helper_functions import *
from models.models import *
from dataloader import BigEarthNetTIFDataModule
import pandas as pd
import torch
import os
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import decode_target
from config.config import DatasetConfig, clean_and_parse_labels, calculate_class_weights
from transformations.normalisation import BandNormalisation

metadata_path = r'C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\metadata_5_percent.csv'
metadata_csv = pd.read_csv(metadata_path)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

# Convert class_labels to a list
class_labels = list(class_labels)
class_weights, class_weights_array = calculate_class_weights(metadata_csv)
class_weights = class_weights_array

# Load the trained model checkpoint

checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_rgb_bands_5%_BigEarthNet\final.ckpt'
model = BigEarthNetResNet18ModelTIF.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=19, in_channels=3, model_weights='ResNet18_Weights.DEFAULT')
model.eval()

# Predict and display a random image
dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\CombinedImages'

print(get_sigmoid_outputs(model, dataset_dir, metadata_csv, bands=DatasetConfig.rgb_bands))

# Load the saved file
data = np.load('test_predictions_ResNet18_5%_BigEarthNet.npz')

# Retrieve predictions and labels from the file
all_preds = data['all_preds']
all_labels = data['all_labels']

print((all_preds[0]))
print((all_labels[0]))

# Check the shapes or contents if needed
print(f"Predictions shape: {all_preds.shape}")
print(f"Labels shape: {all_labels.shape}")

# # # Set print options to display the full arrays
# np.set_printoptions(threshold=np.inf)

# class_labels = DatasetConfig.class_labels

# class_labels = list(class_labels)

# plot_per_label_confusion_matrices_grid(all_labels, all_preds, class_names=class_labels)

# scores = compute_aggregated_metrics(all_labels, all_preds)
# print(scores)

# cooccur_matrix = plot_cooccurrence_matrix(all_labels, all_preds, class_names=class_labels)

# # # Plot confusion matrix
# plot_confusion_matrix(all_preds, all_labels, DatasetConfig)
# plot_normalized_confusion_matrix(all_preds, all_labels, DatasetConfig)

# predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.7, bands=DatasetConfig.rgb_bands)
    




