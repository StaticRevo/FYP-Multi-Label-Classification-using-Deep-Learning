import numpy as np
import matplotlib.pyplot as plt
from config.config import DatasetConfig, clean_and_parse_labels
from utils.test_functions import collect_predictions_and_plot_confusion_matrix, predict_and_display_random_image
from models.CustomModel import CustomModel
import pandas as pd

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5%_BigEarthNet.csv'
metadata_csv = pd.read_csv(metadata_path)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

# Load the saved file
data = np.load('test_predictions.npz')

# Retrieve predictions and labels from the file
all_preds = data['all_preds']
all_labels = data['all_labels']

# Check the shapes or contents if needed
print(f"Predictions shape: {all_preds.shape}")
print(f"Labels shape: {all_labels.shape}")

# Plot confusion matrix
collect_predictions_and_plot_confusion_matrix(all_preds, all_labels, DatasetConfig)

# Load the trained model checkpoint
checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\custom_model-custom_model_Weights.DEFAULT-epoch=00-val_acc=0.94.ckpt'
model = CustomModel.load_from_checkpoint(checkpoint_path, class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3, weights='DEFAULT')

# Predict and display a random image
dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages'
predict_and_display_random_image(model, dataset_dir, metadata_csv, bands=[2, 3, 4])
    
