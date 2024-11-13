from dataclasses import dataclass
import pandas as pd
from torchvision import transforms

# Description: Configuration file for the project
@dataclass
class DatasetConfig:
    dataset_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%'
    combined_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImagesJPG'
    metadata_path: str =r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\metadata_50_percent.csv'
    metadata_csv = pd.read_csv(metadata_path)
    img_size: int = 120
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    num_classes: int = 19
    band_channels: int = 3 #13
    valid_pct: float = 0.1

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
