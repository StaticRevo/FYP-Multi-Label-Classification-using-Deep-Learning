from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config.config import DatasetConfig, ModelConfig
from dataset import BigEarthNetDatasetTIF
from dataclasses import dataclass
from .normalisation import BandNormalisation

@dataclass
class TransformsConfig:
    train_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(size=(120, 120), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((120, 120))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((120, 120))
    ])

    normalisations = transforms.Compose([
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.all_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.all_bands]
        )
    ])

