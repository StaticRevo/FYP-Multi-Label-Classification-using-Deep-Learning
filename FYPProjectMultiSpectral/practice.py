import numpy as np
import matplotlib.pyplot as plt
from config.config import DatasetConfig, clean_and_parse_labels
from utils.test_functions import *
from utils.helper_functions import *
from models.models import *
from dataloader_tif import BigEarthNetTIFDataModule
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
from utils.gradcam import GradCAM
from transformations.normalisation import BandNormalisation

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5_percent.csv'
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
checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\custom_model-None-all_bands-0.5%_BigEarthNetepoch=00-val_loss=6.04.ckpt'
model = CustomModel.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=19, in_channels=12, model_weights='None')

# Predict and display a random image
dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages'

# Load the saved file
data = np.load('test_predictions.npz')

# Retrieve predictions and labels from the file
all_preds = data['all_preds']
all_labels = data['all_labels']

# Check the shapes or contents if needed
print(f"Predictions shape: {all_preds.shape}")
print(f"Labels shape: {all_labels.shape}")

# # Set print options to display the full arrays
# np.set_printoptions(threshold=np.inf)

# # Plot confusion matrix
# plot_confusion_matrix(all_preds, all_labels, DatasetConfig)
# plot_normalized_confusion_matrix(all_preds, all_labels, DatasetConfig)

# predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.7, bands=DatasetConfig.all_bands)
    
# Select an image for GradCAM
random_image_file = random.choice(metadata_csv[metadata_csv['split'] == 'test']['patch_id'].apply(lambda x: f"{x}.tif").tolist())
image_path = os.path.join(dataset_dir, random_image_file)
print()
print(f"Selected image for GradCAM: {random_image_file}")

test_transforms = transforms.Compose([
    transforms.CenterCrop(120),
    transforms.ToTensor(),
    BandNormalisation(
        mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.all_bands],
        std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.all_bands]
    )
])

# Selecting layers from the model to generate activations
image_to_heatmaps = nn.Sequential(*list(model.custom_model.children())[:-4])





