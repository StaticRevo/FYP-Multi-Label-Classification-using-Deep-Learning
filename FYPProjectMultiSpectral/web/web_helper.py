# Standard library imports
import os
import sys

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import json
import uuid

# Third-party imports
from flask import url_for
import rasterio
import torch
import numpy as np
from PIL import Image
import pandas as pd

# Local application imports
from utils.model_utils import get_model_class
from config.config import DatasetConfig, calculate_class_weights
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap 
from utils.data_utils import get_band_indices
from transformations.transforms import TransformsConfig

EXPERIMENTS_DIR = DatasetConfig.experiment_path
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

class_weights, class_weights_array = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))
CLASS_WEIGHTS = class_weights_array

# --- Helper Functions ---
# Load the model from the experiment folder
def load_model_from_experiment(experiment_name):
    # Construct the checkpoint path from the experiment folder.
    checkpoint_path = os.path.join(EXPERIMENTS_DIR, experiment_name, "checkpoints", "final.ckpt")
    
    # Parse the experiment folder name to extract model details.
    parsed = parse_experiment_folder(experiment_name)
    model_name = parsed["model"]
    bands_str = parsed["bands"]
    model_weights = parsed["weights"]
    
    if bands_str.lower() == "all_bands":
        in_channels = len(DatasetConfig.all_bands)
    elif bands_str.lower() == "rgb_bands":
        in_channels = len(DatasetConfig.rgb_bands)
    elif bands_str.lower() == "rgb_nir_bands":
        in_channels = len(DatasetConfig.rgb_nir_bands)
    elif bands_str.lower() == "rgb_swir_bands":
        in_channels = len(DatasetConfig.rgb_swir_bands)
    elif bands_str.lower() == "rgb_nir_swir_bands":
        in_channels = len(DatasetConfig.rgb_nir_swir_bands)
    else:
        in_channels = 3  # Fallback to 3 channels

    main_path = os.path.dirname(checkpoint_path)
    model_class, _ = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Model class for {model_name} not found!")
    
    # Load the model using the checkpoint from the experiment.
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        class_weights=CLASS_WEIGHTS,
        num_classes=DatasetConfig.num_classes,
        in_channels=in_channels,
        model_weights=model_weights,  
        main_path=main_path
    )
    model.eval()
    print(f"Model from experiment {experiment_name} loaded successfully.")
    return model

# Load the experiment metrics from the results folder
def load_experiment_metrics(experiment_name):
    experiment_path = os.path.join(EXPERIMENTS_DIR, experiment_name)
    results_path = os.path.join(experiment_path, "results")
    metrics = {}
    metrics_file = os.path.join(results_path, "best_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            metrics = {"error": f"Error loading metrics: {e}"}
    return metrics

# Preprocess a TIFF image for the model
def preprocess_tiff_image(file_path, selected_bands=DatasetConfig.all_bands):
    transforms_pipeline = TransformsConfig.test_transforms
    normalisation = TransformsConfig.normalisations
    selected_band_indices = get_band_indices(selected_bands, DatasetConfig.all_bands)
    
    try:
        with rasterio.open(file_path) as src:
            image = src.read()  # Shape -> (channels, height, width)
            image = image[selected_band_indices, :, :]  # Select only the desired bands
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Returning a zero tensor.")
        image = torch.zeros((len(selected_band_indices), DatasetConfig.image_height, DatasetConfig.image_width), dtype=torch.float32)
        return image.unsqueeze(0)
    
    image = torch.tensor(image, dtype=torch.float32)
    
    # Apply the transforms and normalisation 
    image = transforms_pipeline(image)
    image = normalisation(image)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
        
    return image

# Create an RGB visualization from a TIFF image
def create_rgb_visualization(file_path, selected_bands=None):
    with rasterio.open(file_path) as src:
        image = src.read()
    if selected_bands is None:
        if image.shape[0] >= 4:
            selected_bands = [3, 2, 1]
        else:
            selected_bands = [0, 1, 2]

    # Normalize each channel [0..1]
    red = image[selected_bands[0]].astype(np.float32)
    green = image[selected_bands[1]].astype(np.float32)
    blue = image[selected_bands[2]].astype(np.float32)
    red = (red - red.min()) / (red.max() - red.min() + 1e-8)
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-8)

    # Stack the channels and convert to uint8
    rgb_image = np.stack([red, green, blue], axis=-1)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # Generate a unique filename for each visualization
    out_filename = f"result_image_{uuid.uuid4().hex}.png"
    out_path = os.path.join(STATIC_FOLDER, out_filename)
    Image.fromarray(rgb_image).save(out_path)
    return url_for('static', filename=out_filename)

# Predict the image using the model
def predict_image_for_model(model, image_tensor):
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    predictions = []
    for idx, prob in enumerate(probs):
        if prob > 0.5:
            label = DatasetConfig.reversed_class_labels_dict.get(idx, f"Class_{idx}")
            predictions.append({"label": label, "probability": prob})
    return predictions

# Generate a Grad-CAM visualization for a single image
def generate_gradcam_for_single_image(model, img_tensor, class_labels, model_name, in_channels, predicted_indices=None):
    gradcam_results = {}

    # Determine target layer based on model_name
    if model_name == 'ResNet18':
        target_layer = model.model.layer3[-1].conv2
    elif model_name == 'ResNet50':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'VGG16':
        target_layer = model.model.features[28]
    elif model_name == 'VGG19':
        target_layer = model.model.features[34]
    elif model_name == 'EfficientNetB0':
        target_layer = model.model.features[8][0]
    elif model_name == 'EfficientNet_v2':
        target_layer = model.modelfeatures[7][4].block[3]
    elif model_name == 'Swin-Transformer':
        target_layer = model.model.stages[3].blocks[-1].norm1
    elif model_name == 'Vit-Transformer':
        target_layer = model.model.layers[-1].attention
    elif model_name == 'custom_model':
        target_layer = model.model[25]
    elif model_name == 'DenstNet121':
        target_layer = model.model.features.norm5
    else:
        print(f"Grad-CAM not implemented for model {model_name}. Skipping visualization.")
        return gradcam_results

    # Ensure img_tensor is batched
    if img_tensor.dim() == 3:
        input_tensor = img_tensor.unsqueeze(0).to(model.device)  # (1, C, H, W)
    elif img_tensor.dim() == 4:
        input_tensor = img_tensor.to(model.device)
    else:
        raise ValueError(f"img_tensor must be 3D or 4D, got {img_tensor.dim()}D.")

    # If predicted_indices are not provided, compute them with a 0.5 threshold.
    if predicted_indices is None:
        output = model(input_tensor)  # (1, num_classes)
        threshold = 0.5
        predicted_indices = torch.where(output[0] > threshold)[0].tolist()

    # For each predicted class, compute a GradCAM heatmap.
    heatmaps = {}
    for idx in predicted_indices:
        grad_cam = GradCAM(model, target_layer)
        input_clone = input_tensor.clone()
        model.zero_grad()
        _ = model(input_clone)  # (Optional: re-run forward pass)
        cam, _ = grad_cam.generate_heatmap(input_clone, target_class=idx)
        heatmap_norm = np.linalg.norm(cam)
        heatmaps[class_labels[idx]] = cam

    # Convert input tensor to a PIL image for visualization.
    img = input_tensor.squeeze()  # Remove batch dimension
    rgb_channels = [3, 2, 1] if in_channels == 12 else [2, 1, 0]
    img = img[rgb_channels, :, :]

    # Normalize each channel.
    img_cpu = img.detach().cpu().numpy()
    red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
    green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
    blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)
    base_img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Save each overlay to disk and record its URL.
    for class_name, heatmap in heatmaps.items():
        overlay = overlay_heatmap(base_img, heatmap, alpha=0.5)
        filename = f"gradcam_{model.__class__.__name__}_{class_name}.png"
        out_path = os.path.join(STATIC_FOLDER, filename)
        overlay.save(out_path)
        gradcam_results[class_name] = url_for('static', filename=filename)

    return gradcam_results

# Fetch the actual labels from the metadata CSV
def fetch_actual_labels(patch_id):
    import ast
    metadata_df = pd.read_csv(DatasetConfig.metadata_path)
    row = metadata_df.loc[metadata_df['patch_id'] == patch_id]
    if row.empty:
        return []
    labels_str = row.iloc[0]['labels']
    if isinstance(labels_str, str):
        try:
            # Clean the string similar to the dataset class logic
            cleaned_labels = labels_str.replace(" '", ", '").replace("[", "[").replace("]", "]")
            labels = ast.literal_eval(cleaned_labels)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing labels for patch_id {patch_id}: {e}")
            labels = []
    else:
        labels = labels_str
    return labels

# Parse the experiment folder name to extract details
def parse_experiment_folder(folder_name):
    parts = folder_name.split('_')
    if len(parts) == 7:
        model = parts[0]
        weights = parts[1]
        bands = parts[2] + "_" + parts[3]
        dataset = parts[4] + "_" + parts[5]
        epochs = parts[6]
    else:
        if any(char.isdigit() for char in parts[-1]):
            if any(char.isdigit() for char in parts[-2]):
                epochs = parts[-2] + "_" + parts[-1]
                dataset = parts[-4] + "_" + parts[-3]
                remaining = parts[:-4]
            else:
                epochs = parts[-1]
                dataset = parts[-3] + "_" + parts[-2]
                remaining = parts[:-3]
        else:
            # Fallback if last part doesn't contain digits.
            epochs = parts[-1]
            dataset = parts[-3] + "_" + parts[-2]
            remaining = parts[:-3]

        if "None" in remaining:
            w_index = remaining.index("None")
            weights = remaining[w_index]
            model = "_".join(remaining[:w_index])  
            bands = "_".join(remaining[w_index+1:])  
        else:
            # If no "None" found, fallback to defaults:
            model = remaining[0]
            weights = ""
            bands = "_".join(remaining[1:])
    return {"model": model, "weights": weights, "bands": bands, "dataset": dataset, "epochs": epochs}

# Save a tensor as an image and return the URL
def save_tensor_as_image(tensor, in_channels=12):
    if tensor.dim() == 4: # If the tensor has a batch dimension, remove it
        tensor = tensor.squeeze(0)

    # Choose channels
    if in_channels == 12:
        rgb_channels = [3, 2, 1]
    else:
        rgb_channels = [2, 1, 0]

    tensor = tensor[rgb_channels, :, :]

    # Normalize each channel [0..1]
    arr = tensor.detach().cpu().numpy()
    red = (arr[0] - arr[0].min()) / (arr[0].max() - arr[0].min() + 1e-8)
    green = (arr[1] - arr[1].min()) / (arr[1].max() - arr[1].min() + 1e-8)
    blue = (arr[2] - arr[2].min()) / (arr[2].max() - arr[2].min() + 1e-8)
    rgb = np.stack([red, green, blue], axis=-1)
    pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

    # Save to static folder
    filename = f"original_img_{uuid.uuid4().hex}.png"
    out_path = os.path.join(STATIC_FOLDER, filename)
    pil_img.save(out_path)
    return url_for('static', filename=filename)

# Get the number of channels and the band names based on the selected bands
def get_channels_and_bands(selected_bands: str):
    selected_bands_lower = selected_bands.lower()
    
    if selected_bands_lower == "all_bands":
        return len(DatasetConfig.all_bands), DatasetConfig.all_bands
    elif selected_bands_lower == "rgb_bands":
        return len(DatasetConfig.rgb_bands), DatasetConfig.rgb_bands
    elif selected_bands_lower == "rgb_nir_bands":
        return len(DatasetConfig.rgb_nir_bands), DatasetConfig.rgb_nir_bands
    elif selected_bands_lower == "rgb_swir_bands":
        return len(DatasetConfig.rgb_swir_bands), DatasetConfig.rgb_swir_bands
    elif selected_bands_lower == "rgb_nir_swir_bands":
        return len(DatasetConfig.rgb_nir_swir_bands), DatasetConfig.rgb_nir_swir_bands
    else:
        return 3, DatasetConfig.rgb_bands  # Default fallback
    
# Validate the number of channels in an image
def validate_image_channels(file_path: str, expected_channels: int):
    try:
        with rasterio.open(file_path) as src:
            actual_channels = src.count  # number of bands
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")

    if actual_channels != expected_channels:
        raise ValueError(
            f"Invalid image: Expected {expected_channels} channels, but found {actual_channels}. "
            "Please upload an image with the correct number of bands."
        )

# Validate the number of channels in an image
def validate_image_channels(file_path, expected_channels):
    with rasterio.open(file_path) as src:
        actual_channels = src.count  # Number of bands in the TIFF
    if actual_channels != expected_channels:
        raise ValueError(f"Invalid image: Expected {expected_channels} channels but got {actual_channels}.")
