import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
# Removed PIL import as per your requirement
from transformations.transforms import TransformsConfig
from config.config import DatasetConfig
import torch
from dataset import BigEarthNetDatasetTIF
import pandas as pd
from dataloader import BigEarthNetTIFDataModule


def get_rgb_from_tensor(image_tensor, in_channels):
    image_cpu = image_tensor.detach().cpu().numpy()

    if in_channels == 12:
        # Select specific bands for RGB visualization (example indices)
        red = image_cpu[3]
        green = image_cpu[2]
        blue = image_cpu[1]
    else:
        # Default to first three channels
        red = image_cpu[0]
        green = image_cpu[1]
        blue = image_cpu[2]

    # Normalize each channel to [0, 1] for visualization
    red = (red - red.min()) / (red.max() - red.min() + 1e-8)
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-8)

    # Stack into an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    return rgb_image


def main():
    # Load configuration and dataset
    bands = DatasetConfig.all_bands
    dataset_dir = DatasetConfig.dataset_paths['0.5']
    metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['0.5'])
    data_module = BigEarthNetTIFDataModule(
        bands=bands,
        dataset_dir=dataset_dir,
        metadata_csv=metadata_csv
    )
    data_module.setup(stage=None)

    # Get the transformation pipeline
    transform_pipeline = TransformsConfig.train_transforms

    # Load one batch of test data
    train_loader = data_module.train_dataloader()
    example_batch = next(iter(train_loader))
    example_imgs, example_lbls = example_batch

    in_channels = len(bands)

    # Select the first image in the batch
    original_img_tensor = example_imgs[0].clone()  # Shape: (C, H, W)

    # Convert original tensor to RGB for visualization
    original_rgb = get_rgb_from_tensor(original_img_tensor, in_channels)

    # Prepare list to store images and their titles
    images_to_show = [("Original", original_rgb)]

    # Extract individual transforms from the pipeline
    if isinstance(transform_pipeline, transforms.Compose):
        transform_list = transform_pipeline.transforms
    else:
        transform_list = [transform_pipeline]

    # Apply each transform to the original image and store results
    for transform in transform_list:
        try:
            # Some transforms expect batch dimensions, ensure proper shape
            transformed_img = transform(original_img_tensor.unsqueeze(0)).squeeze(0)
            transformed_rgb = get_rgb_from_tensor(transformed_img, in_channels)
            images_to_show.append( (transform.__class__.__name__, transformed_rgb) )
        except Exception as e:
            print(f"Transform {transform.__class__.__name__} failed: {e}")
            # If transform fails, append the original image with an error note
            images_to_show.append( (f"{transform.__class__.__name__} (Failed)", original_rgb) )

    # Determine grid size for plotting
    num_images = len(images_to_show)
    cols = min(num_images, 4)  # Limit to 4 columns for better visibility
    rows = (num_images + cols - 1) // cols

    # Create a figure to display images
    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, (title, rgb_image) in enumerate(images_to_show):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
