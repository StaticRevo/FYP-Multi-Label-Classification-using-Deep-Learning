# Core Library modules
from dataclasses import dataclass  # Class decorator for adding special methods to classes.

# PyTorch and Deep Learning Libaries
import torch  # Core PyTorch library for tensor computations.
import torch.nn as nn  # Neural network module for defining layers and architectures.
from torch.utils.data import Dataset  # Data handling and batching
from torchvision import transforms  # Image datasets and transformations.
from PIL import Image  # Image handling and manipulation.
import pandas as pd  # Data analysis and manipulation.
import ast  # Parsing Python code.
from pathlib import Path  # File system path handling.
from config import DatasetConfig  # Import the dataclasses
from utils.helper_functions import encode_label  # Helper function for encoding labels.

from config import ModelConfig, DatasetConfig

class BigEarthNetSubset(Dataset):
    def __init__(self, *, df, root_dir, transforms=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
        self.is_test = is_test

        self.image_paths = list(Path(root_dir).rglob("*.jpg"))
        self.metadata = pd.read_csv(DatasetConfig.metadata_path)
        self.patch_to_labels = dict(zip(self.metadata['patch_id'], self.metadata['labels']))
        self.image_paths = list(Path(root_dir).rglob("*.jpg"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        
        label = self.get_label(image_path)  

        return image, label
    
    def get_label(self, img_path):
        img_path = Path(img_path) 
        patch_id = img_path.stem
        labels = self.patch_to_labels.get(patch_id, None)

        if labels is None:
            return torch.zeros(DatasetConfig.num_classes)  
    
        # Convert the labels string to an actual list if needed
        if isinstance(labels, str):
            labels = ast.literal_eval(labels) 
    
        encoded = encode_label(labels)
        return encoded
    

train_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'train']

train_dataset = BigEarthNetSubset(df=train_df, root_dir=DatasetConfig.combined_rgb_path, transforms=ModelConfig.train_transforms)
print(f"Dataset length: {len(train_dataset)}")
print()
print(train_dataset)
print()
image, label = train_dataset[0]
print()
print(image) # should be a tensor