# Standard library imports
import os
import random

# Third-party imports
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Local application imports
from config.config import DatasetConfig
from models.models import *

def extract_number(string):
    number_str = string.split('%')[0]
    try:
        number = float(number_str)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        raise ValueError(f"Cannot extract a number from the string: {string}")
    
def get_dataset_info(selected_dataset):
    dataset_num = extract_number(selected_dataset)
    dataset_dir = DatasetConfig.dataset_paths[str(dataset_num)]
    metadata_path = DatasetConfig.metadata_paths[str(dataset_num)]
    metadata_csv = pd.read_csv(metadata_path)
    return dataset_dir, metadata_path, metadata_csv

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

def denormalize(tensors, *, mean, std):
    for c in range(DatasetConfig.band_channels):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)