from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config.config import DatasetConfig, ModelConfig
from dataset_tif import BigEarthNetDatasetTIF
from dataclasses import dataclass

@dataclass
class TransformsConfig:
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=DatasetConfig.img_mean, std=DatasetConfig.img_std)
    ])
