from dataclasses import dataclass
import pandas as pd
from torchvision import transforms
import ast
import numpy as np  

metadata_path: str =r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\metadata_50_percent.csv'
metadata_csv = pd.read_csv(metadata_path)

if isinstance(metadata_csv['labels'].iloc[0], str):
    metadata_csv['labels'] = metadata_csv['labels'].apply(ast.literal_eval)

class_labels = metadata_csv['labels'].explode().unique()

# Calculate class weights
label_counts = metadata_csv['labels'].explode().value_counts()
total_counts = label_counts.sum()
class_weights = {label: total_counts / count for label, count in label_counts.items()}
class_weights_array = np.array([class_weights[label] for label in class_labels])

# Description: Configuration file for the project
@dataclass
class DatasetConfig:
    dataset_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%'
    combined_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF'
    combined_rgb_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImagesJPG'
    metadata_path: str =r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\metadata_50_percent.csv'
    unwanted_metadata_file: str = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    metadata_csv = pd.read_csv(metadata_path)
    unwanted_metadata = pd.read_parquet(unwanted_metadata_file)
    img_size: int = 120
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    num_classes: int = 19
    band_channels: int = 3 #13
    valid_pct: float = 0.1
    class_labels_dict = {label: idx for idx, label in enumerate(class_labels)}
    reversed_class_labels_dict = {idx: label for label, idx in class_labels_dict.items()}
    class_weights = class_weights_array

@dataclass
class ModelConfig:
    batch_size: int = 32
    num_epochs: int = 10
    model_name: str = 'resnet18'
    num_workers: int = 2 #os.cpu_count() // 2

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])

