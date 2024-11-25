import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from glob import glob
from pathlib import Path
import ast
import pandas as pd

from config.config import DatasetConfig
from utils.helper_functions import encode_label, get_band_indices


class BigEarthNetDatasetTIF(Dataset):
    def __init__(self, *, df, root_dir, transforms=None, is_test=False, selected_bands=None):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_test = is_test
        self.selected_bands = selected_bands if selected_bands is not None else DatasetConfig.rgb_bands

        self.image_paths = list(Path(root_dir).rglob("*.tif"))
        self.metadata = pd.read_csv(DatasetConfig.metadata_path)
        self.patch_to_labels = dict(zip(self.metadata['patch_id'], self.metadata['labels']))
        self.image_paths = list(Path(root_dir).rglob("*.tif"))

        self.selected_band_indices = get_band_indices(self.selected_bands, DatasetConfig.rgb_bands)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with rasterio.open(image_path) as src:
            image = src.read()  
            image = image[self.selected_band_indices, :, :]
        
        image = torch.tensor(image, dtype=torch.float32)

        if self.transforms:
            image = self.transforms(image)

        label = self.get_label(image_path)

        return image, label

    def get_label(self, img_path):
        img_path = Path(img_path) 
        patch_id = img_path.stem
        labels = self.patch_to_labels.get(patch_id, None)

        if labels is None:
            return torch.zeros(DatasetConfig.num_classes)  
    
        if isinstance(labels, str):
            labels = ast.literal_eval(labels) 
    
        encoded = encode_label(labels)
        return encoded
    