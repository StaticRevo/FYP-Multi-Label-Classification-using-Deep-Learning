# Standard library imports
import os
import sys
import time

# Directory set-up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import json
import uuid
import ast

# Third-party imports
from flask import url_for, flash, render_template
import rasterio
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.cm as cm
from scipy.ndimage import zoom, gaussian_filter
from PIL import ImageEnhance
from rasterio.transform import from_bounds 

# Local application imports
from utils.model_utils import get_model_class
from config.config import DatasetConfig, calculate_class_weights
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap 
from utils.data_utils import get_band_indices
from transformations.transforms import TransformsConfig
from utils.test_utils import get_target_layer

EXPERIMENTS_DIR = DatasetConfig.experiment_path
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

CLASS_WEIGHTS = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))

# --- Helper Functions ---
# Load the model from the experiment folder
def load_model_from_experiment(experiment_name):
    checkpoint_path = os.path.join(EXPERIMENTS_DIR, experiment_name, "checkpoints", "last.ckpt") 
    
    # Parse the experiment folder name to extract model details.
    parsed = parse_experiment_folder(experiment_name)
    model_name = parsed["model"]
    bands_str = parsed["bands"]
    model_weights = parsed["weights"]
    
    if bands_str.lower() == "all_bands":
        in_channels = len(DatasetConfig.all_bands)
    elif bands_str.lower() == "all_imp_bands":
        in_channels = len(DatasetConfig.all_imp_bands)
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

# Get the band indices based on the selected bands
def get_band_indices_web(band_names, all_band_names):
    indices = []
    for band in band_names:
        try:
            band_int = int(band)
            idx = band_int - 1  # convert from 1-indexed to 0-indexed
            if idx < 0 or idx >= len(all_band_names):
                raise ValueError(f"Band index {band_int} is out of range. Valid indices: 1 to {len(all_band_names)}.")
            indices.append(idx)
        except ValueError:
            try:
                idx = all_band_names.index(band)
                indices.append(idx)
            except ValueError:
                raise ValueError(f"Band '{band}' not found in the available bands: {all_band_names}.")
    return indices

# Preprocess a TIFF image for the model
def preprocess_tiff_image(file_path, selected_bands=DatasetConfig.all_bands):
    transforms_pipeline = TransformsConfig.test_transforms
    normalisation = TransformsConfig.normalisations
    selected_band_indices = get_band_indices_web(selected_bands, DatasetConfig.all_bands)
    
    try:
        with rasterio.open(file_path) as src:
            image = src.read()  # Shape (channels, height, width)
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

    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        return gradcam_results
    
    # Ensure img_tensor is batched
    if img_tensor.dim() == 3:
        input_tensor = img_tensor.unsqueeze(0).to(model.device)  
    elif img_tensor.dim() == 4:
        input_tensor = img_tensor.to(model.device)
    else:
        raise ValueError(f"img_tensor must be 3D or 4D, got {img_tensor.dim()}D.")

    # If predicted_indices are not provided, compute them with a 0.5 threshold
    if predicted_indices is None:
        output = model(input_tensor) 
        threshold = 0.5
        predicted_indices = torch.where(output[0] > threshold)[0].tolist()

    # For each predicted class compute a GradCAM heatmap
    heatmaps = {}
    for idx in predicted_indices:
        grad_cam = GradCAM(model, target_layer)
        input_clone = input_tensor.clone()
        model.zero_grad()
        _ = model(input_clone)  
        cam, _ = grad_cam.generate_heatmap(input_clone, target_class=idx)
        heatmap_norm = np.linalg.norm(cam)
        heatmaps[class_labels[idx]] = cam

    # Convert input tensor to a PIL image for visualization
    img = input_tensor.squeeze()  # Remove batch dimension
    rgb_channels = [3, 2, 1] if in_channels == 12 else [2, 1, 0]
    img = img[rgb_channels, :, :]

    # Normalize each channel
    img_cpu = img.detach().cpu().numpy()
    red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
    green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
    blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)
    base_img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Save each overlay to disk and record its URL
    for class_name, heatmap in heatmaps.items():
        overlay = overlay_heatmap(base_img, heatmap, alpha=0.5)
        unique_hash = uuid.uuid4().hex  # generate a unique hash code
        filename = f"gradcam_{model.__class__.__name__}_{class_name}_{unique_hash}.png"
        out_path = os.path.join(STATIC_FOLDER, filename)
        overlay.save(out_path)
        gradcam_results[class_name] = url_for('static', filename=filename)

    return gradcam_results

CATEGORY_GROUPS = {
    "Urban & Industrial": {
        "classes": [7, 18],
        "colour": (255, 0, 0)  # red
    },
    "Agricultural & Managed Lands": {
        "classes": [0, 1, 5, 10, 15, 16],
        "colour": (255, 165, 0)  # orange
    },
    "Forest & Woodland": {
        "classes": [3, 6, 12, 17],
        "colour": (34, 139, 34)  # forest green
    },
    "Natural Vegetation (Non-Forest)": {
        "classes": [13, 14],
        "colour": (128, 128, 0)  # olive
    },
    "Water Bodies": {
        "classes": [8, 11],
        "colour": (0, 0, 255)  # blue
    },
    "Wetlands": {
        "classes": [4, 9],
        "colour": (255, 0, 255)  # magenta
    },
    "Coastal & Transitional": {
        "classes": [2],
        "colour": (255, 255, 0)  # yellow
    }
}
# Generate a colour-coded Grad-CAM visualization
def generate_colourcoded_gradcam(model, img_tensor, class_labels, model_name, in_channels, predicted_indices=None):
    gradcam_results = {}

    target_layer = get_target_layer(model, model_name)
    if target_layer is None:
        return gradcam_results

    # Ensure input tensor is 4D
    if img_tensor.dim() == 3:
        input_tensor = img_tensor.unsqueeze(0).to(model.device)
    elif img_tensor.dim() == 4:
        input_tensor = img_tensor.to(model.device)
    else:
        raise ValueError(f"img_tensor must be 3D or 4D, got {img_tensor.dim()}D.")

    # Compute predicted indices if none provided
    if predicted_indices is None:
        with torch.no_grad():
            output = model(input_tensor)
        threshold = 0.5
        predicted_indices = torch.where(output[0] > threshold)[0].tolist()

    # Convert input tensor to base RGB
    img = input_tensor.squeeze()
    rgb_channels = [3, 2, 1] if in_channels == 12 else [2, 1, 0]
    img = img[rgb_channels, :, :]
    img_cpu = img.detach().cpu().numpy()

    # Normalize each channel to [0..1]
    red   = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
    green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
    blue  = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Slight gamma correction to brighten the RGB image
    rgb_image = np.power(rgb_image, 0.8)
    rgb_image = np.clip(rgb_image, 0, 1)

    target_height, target_width = rgb_image.shape[:2]
    grad_cam = GradCAM(model, target_layer)

    # Prepare a map for each high-level category
    category_cam_map = {
        cat_name: np.zeros((target_height, target_width), dtype=np.float32)
        for cat_name in CATEGORY_GROUPS
    }
    # Generate and accumulate Grad-CAM for every predicted class 
    for idx in predicted_indices:
        class_name = class_labels[idx]

        # Identify which high-level category this class belongs to
        cat_for_class = None
        for cat_name, cat_info in CATEGORY_GROUPS.items():
            if idx in cat_info["classes"]:
                cat_for_class = cat_name
                break
        if cat_for_class is None:
            continue

        # Generate the Grad-CAM
        cam, _ = grad_cam.generate_heatmap(input_tensor.clone(), target_class=idx)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to match the overlay size
        cam_resized = zoom(cam, (target_height / cam.shape[0], target_width / cam.shape[1]), order=3)

        # Use lower sigma for crisper boundaries
        cam_resized = gaussian_filter(cam_resized, sigma=1)
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

        # Accumulate into that category’s map
        category_cam_map[cat_for_class] += cam_resized

    cat_names = list(CATEGORY_GROUPS.keys())
    stacked_maps = []
    cat_colours = []

    for cat_name in cat_names:
        stacked_maps.append(category_cam_map[cat_name])
        (r, g, b) = CATEGORY_GROUPS[cat_name]["colour"]
        cat_colours.append((r / 255.0, g / 255.0, b / 255.0))

    stacked_maps = np.stack(stacked_maps, axis=-1)  # shape: (H, W, #categories)
    cat_colours = np.array(cat_colours)  # shape: (#categories, 3)

    # Hard max across categories per pixel
    argmax_map = np.argmax(stacked_maps, axis=-1)  # shape: (H, W)
    max_vals = np.max(stacked_maps, axis=-1)       

    # Create overlay array
    overlay_array = np.zeros((target_height, target_width, 3), dtype=np.float32)

    # Threshold for ignoring low-activation pixels
    activation_threshold = 0.2

    for i in range(target_height):
        for j in range(target_width):
            best_cat_idx = argmax_map[i, j]
            activation = max_vals[i, j]
            if activation >= activation_threshold:
                overlay_array[i, j] = cat_colours[best_cat_idx] * activation

    # Alpha-blend with the base image
    overlay_alpha = 0.6  
    final_image = rgb_image * (1 - overlay_alpha) + overlay_array * overlay_alpha
    final_image = np.clip(final_image, 0, 1)

    final_img = Image.fromarray((final_image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Color(final_img)
    final_img = enhancer.enhance(1.2)  

    # Save
    unique_hash = uuid.uuid4().hex
    filename = f"gradcam_colourcoded_{model.__class__.__name__}_{unique_hash}.png"
    out_path = os.path.join(STATIC_FOLDER, filename)
    final_img.save(out_path)

    # Build category legend
    category_legend = {}
    for cat_name, cat_info in CATEGORY_GROUPS.items():
        r, g, b = cat_info["colour"]
        category_legend[cat_name] = f"rgb({r},{g},{b})"

    gradcam_results["combined"] = {
        "url": url_for('static', filename=filename),
        "legend": category_legend
    }

    return gradcam_results

# Fetch the actual labels from the metadata CSV
def fetch_actual_labels(patch_id):
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
    elif selected_bands_lower == "all_imp_bands":
        return len(DatasetConfig.all_imp_bands), DatasetConfig.all_imp_bands
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
def validate_image_channels(file_path: str, expected_channels: int) -> int:
    try:
        with rasterio.open(file_path) as src:
            actual_channels = src.count  # number of bands in the image
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")

    if actual_channels < expected_channels:
        raise ValueError(
            f"Invalid image: Expected at least {expected_channels} channels, but found {actual_channels}."
        )
    return actual_channels

# Get the list of experiments from the experiments directory
def get_experiments_list():
    experiments = []
    if os.path.exists(EXPERIMENTS_DIR):
        for d in os.listdir(EXPERIMENTS_DIR):
            full_path = os.path.join(EXPERIMENTS_DIR, d)
            if os.path.isdir(full_path):
                experiments.append(d)
    return experiments

# Process the prediction for a single image
def process_prediction(file_path, filename, bands, selected_experiment):
    try:
        rgb_url = create_rgb_visualization(file_path)
    except Exception as e:
        flash("Error creating RGB visualization: " + str(e), "error")
        experiments = get_experiments_list()
        return render_template('upload.html', experiments=experiments)
    
    input_tensor = preprocess_tiff_image(file_path, selected_bands=bands)
    model_instance = load_model_from_experiment(selected_experiment)
    input_tensor = input_tensor.to(next(model_instance.parameters()).device)
    
    preds = predict_image_for_model(model_instance, input_tensor)
    with torch.no_grad():
        output = model_instance(input_tensor)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    predicted_indices = [idx for idx, prob in enumerate(probs) if prob > 0.5]

    class_probs = {DatasetConfig.class_labels[i]: float(prob) for i, prob in enumerate(probs)}

    experiment_details = parse_experiment_folder(selected_experiment)
    model_name = experiment_details["model"]
    
    # Generate Grad-CAM visualization (one per label)
    gradcam = generate_gradcam_for_single_image(
        model_instance, input_tensor,
        class_labels=DatasetConfig.class_labels,
        model_name=model_name,
        in_channels=len(bands),
        predicted_indices=predicted_indices
    )
    # Generate colour-coded Grad-CAM visualization
    gradcam_colourcoded = generate_colourcoded_gradcam(
        model_instance, input_tensor,
        class_labels=DatasetConfig.class_labels,
        model_name=model_name,
        in_channels=len(bands),
        predicted_indices=predicted_indices
    )

    patch_id = os.path.splitext(filename)[0]  # Fetch actual labels from metadata 
    actual_labels = fetch_actual_labels(patch_id)
    
    experiment_details = parse_experiment_folder(selected_experiment)
    
    return render_template('result.html',
                           filename=filename,
                           predictions={selected_experiment: preds},
                           probabilities=class_probs,
                           actual_labels=actual_labels,
                           rgb_url=rgb_url,
                           gradcam=gradcam,
                           gradcam_colourcoded=gradcam_colourcoded,
                           multiple_models=False,
                           experiment_details=experiment_details)

# Convert an image to a TIFF file
def convert_image_to_tiff(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".png", ".jpeg", ".jpg"]:
        try:
            img = Image.open(file_path).convert("RGB")
            # Append a random UUID so each file is unique
            unique_suffix = uuid.uuid4().hex
            base = file_path.rsplit('.', 1)[0]
            new_file_path = f"{base}_{unique_suffix}.tif"
            img.save(new_file_path, format="TIFF")
            return new_file_path
        except Exception as e:
            raise ValueError(f"Error converting image to TIFF: {e}")
    return file_path

# Crop an image based on geographical coordinates
def crop_image_from_coords(app, image_path, top_left_lat, top_left_lng, bottom_right_lat, bottom_right_lng, out_width=120, out_height=120):
    with rasterio.open(image_path) as src:
        transform = src.transform
        left, bottom = rasterio.transform.xy(transform, src.height, 0)
        right, top = rasterio.transform.xy(transform, 0, src.width)

        # Determine the window based on the provided coordinates
        window = rasterio.windows.from_bounds(top_left_lng, bottom_right_lat, bottom_right_lng, top_left_lat, transform=transform)

        # Read the data within the window
        cropped_data = src.read(window=window)

        # Calculate the transform for the cropped image
        out_transform = from_bounds(top_left_lng, bottom_right_lat, bottom_right_lng, top_left_lat, out_width, out_height)

        # Create a temporary file to save the cropped image
        temp_filename = f"cropped_{time.time()}.tif"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        # Write the cropped data to the temporary file
        with rasterio.open(temp_filepath, 'w', driver='GTiff',
                           height=out_height, width=out_width,
                           count=cropped_data.shape[0], dtype=cropped_data.dtype,
                           crs=src.crs, transform=out_transform) as dst:
            dst.write(cropped_data)

        return temp_filepath
