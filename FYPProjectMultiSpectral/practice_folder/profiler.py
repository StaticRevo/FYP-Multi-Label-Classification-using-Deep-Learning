import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

def display_rgb_image(tiff_file_path):
    with rasterio.open(tiff_file_path) as src:
        # Read the red, green, and blue bands
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Normalize each band to the range 0-1
        red = red.astype(np.float32) / np.max(red)
        green = green.astype(np.float32) / np.max(green)
        blue = blue.astype(np.float32) / np.max(blue)
        
        # Stack the bands into an RGB image
        rgb = np.dstack((red, green, blue))
    return rgb

def apply_transforms_and_display(tiff_file_path):
    # Load original image
    rgb_image = display_rgb_image(tiff_file_path)

    # Convert numpy array to PIL Image for transformations
    pil_img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Convert PIL Image to torch.Tensor
    tensor_img = transforms.ToTensor()(pil_img)

    # Convert tensor to uint8
    tensor_img = (tensor_img * 255).byte()

    # Define the transformations and their names
    transformations = [
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ]
    
    transformation_names = [
        "RandomRotation(30)",
        "RandomResizedCrop (224)",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomResizedCrop (256, 256)",
        "RandomAffine",
        "RandomErasing"
    ]
    
    # Create a figure with subplots (dynamic rows and columns)
    n_transforms = len(transformations)
    cols = 3
    rows = -(-n_transforms // cols) + 1  # Add 1 for the original image
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Display the original image
    axes = axes.flatten()  # Flatten axes for easy indexing
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Apply each transformation and display the result
    for i, (transform, name) in enumerate(zip(transformations, transformation_names), start=1):
        transformed_image = transform(tensor_img)
        transformed_image = transforms.ToPILImage()(transformed_image)
        axes[i].imshow(transformed_image)
        axes[i].set_title(name)
        axes[i].axis('off')

    # Hide any unused subplots
    for j in range(len(transformations) + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Path to your .tif file
tiff_file_path = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF\S2B_MSIL2A_20180506T105029_N9999_R051_T31UER_79_36.tif'
apply_transforms_and_display(tiff_file_path)