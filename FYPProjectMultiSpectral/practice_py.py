import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformations.transforms import TransformsConfig
from config.config import DatasetConfig
import torch

def display_rgb_image(tiff_file_path):
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
        
    return rgb

def apply_transforms_and_display(tiff_file_path, transforms):
    # Load original image
    rgb_image = display_rgb_image(tiff_file_path)

    # Convert numpy array to PIL Image for transformations
    pil_img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Apply the transformations
    transformed_image = transforms(pil_img)

    # Convert transformed image back to numpy array for visualization
    if isinstance(transformed_image, torch.Tensor):
        transformed_image = transformed_image.permute(1, 2, 0).numpy()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display the transformed image
    axes[1].imshow(transformed_image)
    axes[1].set_title("Transformed Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# Path to your .tif file
tiff_file_path = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF\S2B_MSIL2A_20180506T105029_N9999_R051_T31UER_79_36.tif'

# Apply and display the train transforms
apply_transforms_and_display(tiff_file_path, TransformsConfig.train_transforms)