import os
import rasterio
import numpy as np
from tqdm import tqdm

def calculate_band_stats(root_dir, num_bands):
    band_means = np.zeros(num_bands)
    band_stds = np.zeros(num_bands)
    pixel_counts = np.zeros(num_bands)

    # Get the total number of folders for the progress bar
    total_folders = sum(os.path.isdir(os.path.join(root_dir, folder)) for folder in os.listdir(root_dir))

    # Iterate through each folder in the root directory with a progress bar
    with tqdm(total=total_folders, desc="Processing folders", unit="folder") as pbar:
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                # Iterate through each band file in the folder
                for band in range(1, num_bands + 1):
                    band_file = os.path.join(folder_path, f"{folder}_B{band:02d}.tif")
                    if os.path.exists(band_file):
                        with rasterio.open(band_file) as src:
                            band_data = src.read(1).astype(np.float32)
                            band_means[band - 1] += band_data.sum()
                            band_stds[band - 1] += (band_data ** 2).sum()
                            pixel_counts[band - 1] += band_data.size
                pbar.update(1)

    # Calculate means and standard deviations
    band_means /= pixel_counts
    band_stds = np.sqrt(band_stds / pixel_counts - band_means ** 2)

    return band_means, band_stds

# Example usage
root_dir = r'C:\Users\isaac\Desktop\BigEarthTests\OnePBigEarthNetCopySubsets'
num_bands = 12  # Number of bands in your dataset
band_means, band_stds = calculate_band_stats(root_dir, num_bands)

# Print the results
for i in range(num_bands):
    print(f"Band {i+1} - Mean: {band_means[i]}, Std: {band_stds[i]}")