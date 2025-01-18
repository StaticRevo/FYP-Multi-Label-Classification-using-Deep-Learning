import torch
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import random
from config.config import DatasetConfig
from models.models import *
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Helper functions
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize(tensors, *, mean, std):
    for c in range(DatasetConfig.band_channels):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)

def encode_label(label: list, num_classes=DatasetConfig.num_classes):
    target = torch.zeros(num_classes)
    for l in label:
        if l in DatasetConfig.class_labels_dict:
            target[DatasetConfig.class_labels_dict[l]] = 1.0
    return target

def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.4,
    cls_labels: dict = None,
):
    result = []
    for i, x in enumerate(target):
        if x >= threshold:
            if text_labels:
                result.append(cls_labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)


def get_band_indices(band_names, all_band_names):
    return [all_band_names.index(band) for band in band_names]

# Function to derive bands based on selected_bands
def get_bands(selected_bands):
    band_options = {
        'all_bands': ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        'rgb_bands': ["B04", "B03", "B02"],
        'rgb_nir_bands': ["B04", "B03", "B02", "B08"],
        'rgb_swir_bands': ["B04", "B03", "B02", "B11", "B12"],
        'rgb_nir_swir_bands': ["B04", "B03", "B02", "B08", "B11", "B12"]
    }
    return band_options.get(selected_bands, [])

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

# Function to save TensorBoard graphs as images
def save_tensorboard_graphs(log_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get the list of tags (metrics) in the logs
    tags = event_acc.Tags().get('scalars', [])

    # Filter tags: include tags with '_epoch' or starting with 'val_', exclude 'class' in name
    filtered_tags = [
        tag for tag in tags 
        if ('_epoch' in tag or tag.startswith('val_')) and 'class' not in tag
    ]

    # Iterate over each tag and plot the graph
    for tag in filtered_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Plot the graph
        plt.figure()
        plt.plot(steps, values, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel(tag)
        plt.title(tag.replace('_', ' ').capitalize())
        plt.grid(True)

        # Save the graph as an image
        sanitized_tag = tag.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{sanitized_tag}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Graphs saved to {output_dir}")

def extract_number(string):
    number_str = string.split('%')[0]
    try:
        number = float(number_str)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        raise ValueError(f"Cannot extract a number from the string: {string}")

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

def calculate_band_stats(root_dir, num_bands):
    band_means = np.zeros(num_bands)
    band_stds = np.zeros(num_bands)
    pixel_counts = np.zeros(num_bands)

    # Get the total number of files for the progress bar
    total_files = sum(os.path.isfile(os.path.join(root_dir, file)) for file in os.listdir(root_dir))

    # Iterate through each file in the root directory with a progress bar
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if os.path.isfile(file_path) and file_path.endswith('.tif'):
                with rasterio.open(file_path) as src:
                    for band in range(1, num_bands + 1):
                        band_data = src.read(band).astype(np.float32)
                        band_means[band - 1] += band_data.sum()
                        band_stds[band - 1] += (band_data ** 2).sum()
                        pixel_counts[band - 1] += band_data.size
                pbar.update(1)

    # Calculate means and standard deviations
    band_means /= pixel_counts
    band_stds = np.sqrt(band_stds / pixel_counts - band_means ** 2)

    return band_means, band_stds

def convert_tif_to_jpg(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(tif_path, root_dir)
                jpg_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.jpg')
                
                os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
                
                with Image.open(tif_path) as img:
                    img.convert('RGB').save(jpg_path, 'JPEG')

##### Trainer helper functions

# Function to initialize paths for saving results
def initialize_paths(model_name, weights, selected_bands, selected_dataset, epochs):
    experiment_path = DatasetConfig.experiment_path
    main_path = fr'{experiment_path}\{model_name}_{weights}_{selected_bands}_{selected_dataset}_{epochs}epochs'
    if os.path.exists(main_path):
        increment = 1
        new_main_path = f"{main_path}_{increment}"
        while os.path.exists(new_main_path):
            increment += 1
            new_main_path = f"{main_path}_{increment}"
        main_path = new_main_path
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    return main_path


def get_dataset_info(selected_dataset):
    dataset_num = extract_number(selected_dataset)
    dataset_dir = DatasetConfig.dataset_paths[str(dataset_num)]
    metadata_path = DatasetConfig.metadata_paths[str(dataset_num)]
    metadata_csv = pd.read_csv(metadata_path)
    return dataset_dir, metadata_path, metadata_csv

def get_model_class(model_name):
    model_mapping = {
        'custom_model': (CustomModel, 'custom_model'),
        'ResNet18': (ResNet18, 'resnet18'),
        'ResNet50': (ResNet50, 'resnet50'),
        'VGG16': (VGG16, 'vgg16'),
        'VGG19': (VGG19, 'vgg19'),
        'DenseNet121': (DenseNet121, 'densenet121'),
        'EfficientNetB0': (EfficientNetB0, 'efficientnetb0'),
        'EfficientNet_v2': (EfficientNetV2, 'efficientnet_v2'),
        'Vit-Transformer': (VitTransformer, 'vit_transformer'),
        'Swin-Transformer': (SwinTransformer, 'swin_transformer')
    }
    return model_mapping.get(model_name, (None, None))