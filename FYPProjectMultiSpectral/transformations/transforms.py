from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config.config import DatasetConfig, ModelConfig
from dataset_tif import BigEarthNetDatasetTIF
from dataclasses import dataclass
from .normalisation import BandNormalisation

@dataclass
class TransformsConfig:
    train_transforms = transforms.Compose([
        transforms.RandomCrop(120),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.rgb_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.rgb_bands]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop(120),
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.rgb_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.rgb_bands]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(120),
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.rgb_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.rgb_bands]
        )
    ])
