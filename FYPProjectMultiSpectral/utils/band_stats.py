import os
import rasterio
import numpy as np
from tqdm import tqdm

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

# Example usage
root_dir = r'C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\CombinedImages'
num_bands = 12  
band_means, band_stds = calculate_band_stats(root_dir, num_bands)

# Print the results
for i in range(num_bands):
    print(f"Band {i+1} - Mean: {band_means[i]}, Std: {band_stds[i]}")