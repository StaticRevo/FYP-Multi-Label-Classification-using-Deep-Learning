import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import logging
import rasterio

def predict_and_display_user_selected_image(model, image_path, metadata_csv, threshold=0.6, bands=DatasetConfig.all_bands):
    # Create dictionaries for mapping between labels and indices
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict

    band_indices = { # Map band names to indices
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }

    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]

    with rasterio.open(image_path) as src:
        all_bands = src.read().astype(np.float32)  # Read all bands

        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)

        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]

    rgb = np.dstack((red, green, blue)) # Stack the bands into an RGB image

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
        all_bands = src.read().astype(np.float32) # Read all bands
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True) # Normalize each band to the range 0-1
    
        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]
    
    rgb = np.dstack((red, green, blue)) # Stack the bands into an RGB image
    
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

    # Get true labels from the CSV
    image_id = os.path.splitext(random_image_file)[0]
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

def get_labels_for_image(image_path, model, transform, patch_to_labels):
    with rasterio.open(image_path) as src: # Load and preprocess the image
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

def display_image(image_path):
    with rasterio.open(image_path) as src:
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        plt.imshow(image)
        plt.title("Image with Bands 2, 3, and 4")
        plt.show()

def display_image_and_labels(image_path, model, transform, patch_to_labels):
    # Display the image
    display_image(image_path)

    # Get predicted and true labels
    preds, true_labels, _ = get_labels_for_image(image_path, model, transform, patch_to_labels)
    print(f"Predicted Labels: {preds}")
    print(f"True Labels: {true_labels}")

def display_rgb_image_from_tiff(tiff_file_path):
    with rasterio.open(tiff_file_path) as src:
        # Read the red, green, and blue bands
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Normalize each band to the range 0-1
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        blue = blue.astype(np.float32)
        
        red /= np.max(red)
        green /= np.max(green)
        blue /= np.max(blue)
        
        # Stack the bands into an RGB image
        rgb = np.dstack((red, green, blue))
        
        # Display the RGB image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()