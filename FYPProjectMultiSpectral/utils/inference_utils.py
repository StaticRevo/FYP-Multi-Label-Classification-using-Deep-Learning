# Standard library imports
import os
import math
import random

# Third-party imports
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, f1_score, 
    hamming_loss, accuracy_score, multilabel_confusion_matrix
)
import rasterio

# Local application imports
from config.config import DatasetConfig
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap

def predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.6, bands=DatasetConfig.rgb_bands):
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict
          
    test_metadata = metadata_csv[metadata_csv['split'] == 'test']
    
    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }
    
    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]
    
    # Select a random image from the test split
    test_image_files = test_metadata['patch_id'].apply(lambda x: f"{x}.tif").tolist()
    random_image_file = random.choice(test_image_files)
    image_path = os.path.join(dataset_dir, random_image_file)
    print(f"Selected random image file: {random_image_file}")

    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read().astype(np.float32)
    
        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)
    
        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]
    
    # Stack the bands into an RGB image
    rgb = np.dstack((red, green, blue))
    
    # Read the specified bands for model input
    input_bands = np.stack([all_bands[band_indices[band]] for band in bands], axis=0)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_bands).unsqueeze(0).float()
    input_tensor = input_tensor.to(model.device)

    model.eval() # Predict labels
    with torch.no_grad():
        output = model(input_tensor)

    sigmoid_outputs = output.sigmoid()
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()

    print("Sigmoid Outputs:", sigmoid_outputs)

    
    image_id = os.path.splitext(random_image_file)[0] # Get true labels from the CSV
    row = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
    true_labels = row if isinstance(row, list) else [row]

    # Convert true labels to indices
    true_labels_indices = [class_labels_dict[label] for label in true_labels]

    # Convert numeric predicted labels to text
    predicted_labels_indices = [idx for idx, value in enumerate(predicted_labels) if value == 1]

    print(f"Image ID: {image_id}")
    print(f"True Labels (Indices): {true_labels_indices}")
    print(f"Predicted Labels (Indices): {predicted_labels_indices}")

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(
        f"Image ID: {image_id}\n"
        f"True Labels: {true_labels_indices}\n"
        f"Predicted Labels: {predicted_labels_indices}"
    )
    plt.axis('off')
    plt.show()

def predict_and_display_user_selected_image(model, image_path, metadata_csv, threshold=0.6, bands=DatasetConfig.all_bands):
    # Create dictionaries for mapping between labels and indices
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict

    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }

    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]

    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read().astype(np.float32)

        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)

        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]

    # Stack the bands into an RGB image
    rgb = np.dstack((red, green, blue))

    # Read the specified bands for model input
    input_bands = np.stack([all_bands[band_indices[band]] for band in bands], axis=0)

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_bands).unsqueeze(0).float()
    input_tensor = input_tensor.to(model.device)

    # Predict labels
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    sigmoid_outputs = output.sigmoid()
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()

    print("Sigmoid Outputs:", sigmoid_outputs)

    # Get true labels from the CSV
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    row = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
    true_labels = row if isinstance(row, list) else [row]

    # Convert true labels to indices
    true_labels_indices = [class_labels_dict[label] for label in true_labels]

    # Convert numeric predicted labels to text
    predicted_labels_indices = [idx for idx, value in enumerate(predicted_labels) if value == 1]

    # Print results
    print(f"Image ID: {image_id}")
    print(f"True Labels (Indices): {true_labels_indices}")
    print(f"Predicted Labels (Indices): {predicted_labels_indices}")

    # Display the image, true labels, and predicted labels
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(
        f"Image ID: {image_id}\n"
        f"True Labels: {true_labels_indices}\n"
        f"Predicted Labels: {predicted_labels_indices}"
    )
    plt.axis('off')
    plt.show()

def predict_batch(model, dataloader, threshold=0.6, bands=DatasetConfig.all_bands):
    model.eval()
    all_preds = []
    all_true_labels = []
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs)
            sigmoid_outputs = outputs.sigmoid()
            preds = (sigmoid_outputs > threshold).cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_true_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_true_labels)

def display_batch_predictions(model, dataloader, in_channels, threshold=0.6, bands=DatasetConfig.all_bands, num_images=10):
    all_preds, all_true_labels = predict_batch(model, dataloader, threshold, bands)
    
    dataset_size = len(dataloader.dataset)
    num_images = min(num_images, dataset_size)  
    
    # Randomly select unique indices
    random_indices = random.sample(range(dataset_size), num_images)
    
    if in_channels == 12:
        rgb_channels = [3, 2, 1]
    else:
        rgb_channels  = [2, 1, 0]

    
    for i in random_indices:
        pred = all_preds[i]
        true = all_true_labels[i]
        
        # Convert numeric predicted labels to text
        predicted_labels_indices = [idx for idx, value in enumerate(pred) if value == 1]
        true_labels_indices = [idx for idx, value in enumerate(true) if value == 1]
    
        # Get the image and select RGB bands for visualization
        image_tensor = dataloader.dataset[i][0]  # Assuming the first element is the image
        img = image_tensor[rgb_channels , :, :] 

        # Normalize each channel
        img_cpu = img.detach().cpu().numpy()
        red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
        green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
        blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)

        # Stack into an RGB image
        rgb_image = np.stack([red, green, blue], axis=-1)

        # Convert to PIL Image
        image_rgb = Image.fromarray((rgb_image * 255).astype(np.uint8))
    
        # Display the image, true labels, and predicted labels
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(
            f"Image {i}\n"
            f"True Labels: {true_labels_indices}\n"
            f"Predicted Labels: {predicted_labels_indices}"
        )
        plt.axis('off')
        plt.show()

def get_labels_for_image(image_path, model, transform, patch_to_labels):
    # Load and preprocess the image
    with rasterio.open(image_path) as src:
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        image = transform(image).unsqueeze(0).to(model.device)  # Add batch dimension and move to device

    # Get the predicted labels
    model.eval()
    with torch.no_grad():
        preds = model(image).sigmoid() > 0.5  # Apply sigmoid and threshold at 0.5
        preds = preds.cpu().numpy().astype(int).flatten()

    # Get the true labels
    patch_id = os.path.basename(image_path).split('.')[0]
    true_labels = patch_to_labels[patch_id]

    return preds, true_labels, image