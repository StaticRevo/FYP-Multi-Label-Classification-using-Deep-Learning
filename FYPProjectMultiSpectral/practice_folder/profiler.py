import rasterio
import numpy as np
import matplotlib.pyplot as plt

def extract_bands(input_path, output_path):
    # Open the input TIFF file
    with rasterio.open(input_path) as src:
        # Read bands 2, 3, and 4 (rasterio uses 1-based indexing)
        band2 = src.read(2)
        band3 = src.read(3)
        band4 = src.read(4)
        
        # Stack the selected bands into a single array with shape (3, height, width)
        stacked_bands = np.stack([band2, band3, band4])
        
        # Copy the metadata and update the count for the new bands
        profile = src.profile.copy()
        profile.update(count=3)
        
        # Write the new TIFF file with the extracted bands
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(stacked_bands)

#extract_bands(r"C:\Users\isaac\Desktop\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_37_67.tif", r"C:\Users\isaac\Desktop\rgb.tif")

# Open the TIFF file and read the RGB bands
with rasterio.open(r"C:\Users\isaac\Desktop\rgb.tif") as src:
    rgb = src.read([1, 2, 3])
    # Transpose from (bands, height, width) to (height, width, bands)
    rgb_transposed = np.transpose(rgb, (1, 2, 0))

# Normalize the image: scale each channel to [0, 1]
rgb_normalized = np.zeros_like(rgb_transposed, dtype=np.float32)
for i in range(3):
    band = rgb_transposed[:, :, i]
    band_min, band_max = band.min(), band.max()
    # Avoid division by zero in case band_max equals band_min
    if band_max > band_min:
        rgb_normalized[:, :, i] = (band - band_min) / (band_max - band_min)
    else:
        rgb_normalized[:, :, i] = band

# Display the normalized image using matplotlib
plt.imshow(rgb_normalized)
plt.title("RGB TIFF Image")
plt.axis('off')  # Hide axis ticks
plt.show()