from torchvision import transforms
from config.config import DatasetConfig
from dataclasses import dataclass
from .normalisation import BandNormalisation

@dataclass
class TransformsConfig:
    # Training transforms
    train_transforms = transforms.Compose([
        # Random flips and rotations for orientation invariance
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomErasing(p=0.2)
    ])

    # Validation transforms 
    val_transforms = transforms.Compose([
        transforms.Resize((120, 120))
    ])

    # Test transforms 
    test_transforms = transforms.Compose([
        transforms.Resize((120, 120))
    ])

    # Normalizations (applied after spatial transforms)
    normalisations = transforms.Compose([
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.all_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.all_bands]
        )
    ])
