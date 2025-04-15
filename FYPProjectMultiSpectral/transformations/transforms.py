# Third-party imports
from torchvision import transforms

# Local application imports
from config.config import DatasetConfig
from dataclasses import dataclass
from .normalisation import BandNormalisation

 # Training transforms Config
@dataclass
class TransformsConfig:
    train_transforms = transforms.Compose([
        transforms.Resize((120, 120)), # Resize to 120x120
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.RandomVerticalFlip(), # Random vertical flip
        transforms.RandomRotation(degrees=(0, 180)), # Random rotation
        transforms.RandomErasing(p=0.2) # Random erasing
    ])

    # Validation transforms 
    val_transforms = transforms.Compose([
        transforms.Resize((120, 120)) # Resize to 120x120
    ])

    # Test transforms 
    test_transforms = transforms.Compose([
        transforms.Resize((120, 120)) # Resize to 120x120
    ])

    # Normalizations (applied after spatial transforms)
    normalisations = transforms.Compose([
        BandNormalisation(
            mean=[DatasetConfig.band_stats["mean"][band] for band in DatasetConfig.all_bands],
            std=[DatasetConfig.band_stats["std"][band] for band in DatasetConfig.all_bands]
        )
    ])
