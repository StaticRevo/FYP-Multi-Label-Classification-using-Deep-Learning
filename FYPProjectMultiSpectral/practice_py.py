import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformations.transforms import TransformsConfig
from config.config import DatasetConfig
import torch
from dataset_tif import BigEarthNetDatasetTIF
from transformations.transforms import TransformsConfig

# Path to your .tif file
tiff_file_path = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF\S2B_MSIL2A_20180506T105029_N9999_R051_T31UER_79_36.tif'
root_dir = DatasetConfig.combined_path
metadata_csv = DatasetConfig.metadata_csv
train_df = metadata_csv[metadata_csv['split'] == 'train']
transforms = TransformsConfig.train_transforms

train_dataset = BigEarthNetDatasetTIF(df=train_df, root_dir=root_dir, transforms=None, selected_bands=DatasetConfig.rgb_nir_bands)

image, label = train_dataset[5]
print(f"Image shape: {image.shape}, Label: {label}")

print()

print(image)

print()

train_dataset2 = BigEarthNetDatasetTIF(df=train_df, root_dir=root_dir, transforms=transforms, selected_bands=DatasetConfig.rgb_nir_bands)

image, label = train_dataset2[5]
print(f"Image shape: {image.shape}, Label: {label}")

print()

print(image)

