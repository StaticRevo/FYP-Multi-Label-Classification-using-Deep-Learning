import numpy as np
import matplotlib.pyplot as plt
from config.config import DatasetConfig, clean_and_parse_labels
from utils.test_functions import *
from models.CustomModel import CustomModel
from models.ResNet18 import BigEarthNetResNet18ModelTIF
from models.ResNet50 import BigEarthNetResNet50ModelTIF
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

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\metadata_1_percent.csv'
metadata_csv = pd.read_csv(metadata_path)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

# Convert class_labels to a list
class_labels = list(class_labels)

# Load the saved file
data = np.load('test_predictions.npz')

# Retrieve predictions and labels from the file
all_preds = data['all_preds']
all_labels = data['all_labels']

# Check the shapes or contents if needed
print(f"Predictions shape: {all_preds.shape}")
print(f"Labels shape: {all_labels.shape}")

print(all_labels)
print(all_preds)

# # Plot confusion matrix
# plot_confusion_matrix(all_preds, all_labels, DatasetConfig)
# plot_normalized_confusion_matrix(all_preds, all_labels, DatasetConfig)

# # Load the trained model checkpoint
# checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\ResNet18-ResNet18_Weights.DEFAULT-epoch=01-val_acc=0.93.ckpt'
# model = BigEarthNetResNet18ModelTIF.load_from_checkpoint(checkpoint_path, class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3, model_weights='ResNet18_Weights.DEFAULT')

# # Predict and display a random image
# dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages'
# predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.55, class_labels=class_labels)
    
# predict_and_display_multiple_images(model, dataset_dir, metadata_csv, threshold=0.55, class_labels=class_labels, num_images=5)

# # Select an image for GradCAM
# random_image_file = random.choice(metadata_csv[metadata_csv['split'] == 'test']['patch_id'].apply(lambda x: f"{x}.tif").tolist())
# print(f"Selected image for GradCAM: {random_image_file}")
# image_path = os.path.join(dataset_dir, random_image_file)
# print(f"Selected image for GradCAM: {random_image_file}")

# # Display GradCAM heatmap
# selected_layer = 'model.layer4'

# print(model)
# display_gradcam_heatmap(model, image_path, class_labels, selected_layer, threshold=0.55)