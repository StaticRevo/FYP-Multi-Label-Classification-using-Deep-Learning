import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformations.transforms import TransformsConfig
from config.config import DatasetConfig
import torch
from FYPProjectMultiSpectral.dataset import BigEarthNetDatasetTIF

root_dir = DatasetConfig.dataset_path
metadata_csv = DatasetConfig.metadata_csv
train_df = metadata_csv[metadata_csv['split'] == 'train']
transforms = TransformsConfig.train_transforms

train_dataset = BigEarthNetDatasetTIF(df=train_df, root_dir=root_dir, transforms=None, selected_bands=DatasetConfig.all_imp_bands)

image, label = train_dataset[5]
print(f"Original Image shape: {image.shape}, Label: {label}")

# Convert the tensor image back to a numpy array
image_np = image.numpy()

train_dataset2 = BigEarthNetDatasetTIF(df=train_df, root_dir=root_dir, transforms=transforms, selected_bands=DatasetConfig.all_imp_bands)

image2, label = train_dataset2[5]
print(f"Transformed Image shape: {image2.shape}, Label: {label}")

# Convert the tensor image back to a numpy array
image2_np = image2.numpy()

# Combine the first three bands to form the RGB image
original_rgb = np.stack([image_np[0], image_np[1], image_np[2]], axis=-1)
transformed_rgb = np.stack([image2_np[0], image2_np[1], image2_np[2]], axis=-1)

# Normalize the RGB values to [0, 1] for display if needed
if original_rgb.max() > 1:
    original_rgb = (original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min())
if transformed_rgb.max() > 1:
    transformed_rgb = (transformed_rgb - transformed_rgb.min()) / (transformed_rgb.max() - transformed_rgb.min())

# Display the original and transformed RGB images using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Display original RGB image
ax = axes[0]
ax.imshow(original_rgb)
ax.set_title('Original RGB Image')
ax.axis('off')

# Display transformed RGB image
ax = axes[1]
ax.imshow(transformed_rgb)
ax.set_title('Transformed RGB Image')
ax.axis('off')

plt.tight_layout()
plt.show()

# Display the original and transformed images using matplotlib for all bands
num_bands = image_np.shape[0]
fig, axes = plt.subplots(2, num_bands, figsize=(20, 10))

for i in range(num_bands):
    # Display original image bands
    ax = axes[0, i]
    ax.imshow(image_np[i, :, :], cmap='gray')
    ax.set_title(f'Original Band {i + 1}')
    ax.axis('off')

    # Display transformed image bands
    ax = axes[1, i]
    ax.imshow(image2_np[i, :, :], cmap='gray')
    ax.set_title(f'Transformed Band {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Display band numbers of the original image
band_numbers = list(range(1, num_bands + 1))
print(f"Band numbers in the original image: {band_numbers}")
