import os
import numpy as np
import rasterio
from tqdm import tqdm


# Compute mean and standard deviation for each band in multi-spectral images
def compute_band_statistics(image_dir):
    band_sums = None
    band_squared_sums = None
    total_pixels = 0

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")] 
    
    for fname in tqdm(image_files, desc="Processing images"):
        fpath = os.path.join(image_dir, fname)
        with rasterio.open(fpath) as src:
            img = src.read().astype(np.float32)  # Read image data as a NumPy array (shape: 12 bands, H, W)

            # Initialize sums on the first iteration
            if band_sums is None:
                num_bands = img.shape[0]
                band_sums = np.zeros(num_bands, dtype=np.float64)
                band_squared_sums = np.zeros(num_bands, dtype=np.float64)

            # Reshape image to (12, H*W) for easier computation
            reshaped = img.reshape(num_bands, -1)  
            band_sums += reshaped.sum(axis=1)
            band_squared_sums += (reshaped ** 2).sum(axis=1)
            total_pixels += reshaped.shape[1]

    # Calculate mean and standard deviation for each band
    means = band_sums / total_pixels
    stds = np.sqrt(band_squared_sums / total_pixels - means ** 2)

    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
                  'B08', 'B8A', 'B09', 'B11', 'B12']

    stats = {
        "mean": {band: float(means[i]) for i, band in enumerate(band_names)},
        "std": {band: float(stds[i]) for i, band in enumerate(band_names)}
    }

    return stats


image_dir = r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\CombinedImages"
band_stats = compute_band_statistics(image_dir)
print("Mean per band:", band_stats["mean"])
print("Std per band:", band_stats["std"])
