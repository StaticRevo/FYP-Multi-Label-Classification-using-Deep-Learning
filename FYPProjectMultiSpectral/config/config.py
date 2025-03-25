# -- Configuration file for the project --

# Standard library imports
from dataclasses import dataclass, field

# Third-party imports
import pandas as pd
import torch
import torch.nn as nn

# Local application imports
from .config_utils import *

# -- Dataset Configuration --
@dataclass
class DatasetConfig:
    metadata_path = r"C:\Users\isaac\Desktop\BigEarthTests\100%_BigEarthNet\metadata_100_percent.csv"
    dataset_paths = {
        "0.5": r"C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages",
        "1": r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages",
        "5": r"C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\CombinedImages",
        "10": r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\CombinedImages",
        "50": r"C:\Users\isaac\Desktop\BigEarthTests\50%_BigEarthNet\CombinedImages",
        "100": r"D:\100%_BigEarthNet"
    }
    metadata_paths = {
        "0.5": r"C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5_percent.csv",
        "1": r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\metadata_1_percent.csv",
        "5": r"C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\metadata_5_percent.csv",
        "10": r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\metadata_10_percent.csv",
        "50": r"C:\Users\isaac\Desktop\BigEarthTests\50%_BigEarthNet\metadata_50_percent.csv",
        "100": r"C:\Users\isaac\Desktop\BigEarthTests\100%_BigEarthNet\metadata_100_percent.csv"
    }
    unwanted_metadata_file: str = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    unwanted_metadata_csv = pd.read_parquet(unwanted_metadata_file)

    experiment_path = r'C:\Users\isaac\Desktop\experiments'
    
    class_labels = calculate_class_labels(pd.read_csv(metadata_path))
    class_labels = class_labels
    class_labels_dict = {label: idx for idx, label in enumerate(class_labels)}
    reversed_class_labels_dict = {idx: label for label, idx in class_labels_dict.items()}

    num_classes: int = 19
    band_channels: int = 12
    valid_pct: float = 0.1
    img_size: int = 120
    image_height: int = 120
    image_width: int = 120

    rgb_bands = ["B04", "B03", "B02"]
    rgb_nir_bands = ["B04", "B03", "B02", "B08"]
    rgb_swir_bands = ["B04", "B03", "B02", "B11", "B12"]
    rgb_nir_swir_bands = ["B04", "B03", "B02", "B08", "B11", "B12"]
    all_imp_bands = [ "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    band_stats = {
        "mean": {
            "B01": 359.93681858037576,
            "B02": 437.7795146920668,
            "B03": 626.9061237185847,
            "B04": 605.0589129818594,
            "B05": 971.6512098450492,
            "B06": 1821.9817358749056,
            "B07": 2108.096240315571,
            "B08": 2256.3215618504346,
            "B8A": 2310.6351913265307,
            "B09": 2311.6085833217353,
            "B11": 1608.6865167942176,
            "B12": 1017.1259618291762
        },
        "std": {
            "B01": 583.5085769396974,
            "B02": 648.4384481402268,
            "B03": 639.2669907766995,
            "B04": 717.5748664544205,
            "B05": 761.8971822905785,
            "B06": 1090.758232889144,
            "B07": 1256.5524552734478,
            "B08": 1349.2050493390414,
            "B8A": 1287.1124261320342,
            "B09": 1297.654379610044,
            "B11": 1057.3350765979644,
            "B12": 802.1790763840752
        }
    }

# -- Model Configuration --
@dataclass
class ModelConfig:
    num_epochs: int = 100
    batch_size: int = 256
    num_workers: int = 8 
    learning_rate: float = 0.001
    lr_factor: float = 0.5
    lr_patience: int = 4 # ReduceLROnPlateau Patience
    patience: int = 10 # Early Stopping Patience
    momentum: float = 0.9
    weight_decay: float = 1e-3
    dropout: float = 0.5
    loss_fn: str = "CombinedFocalLossWithPosWeight"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
       
# -- Module Configuration --
@dataclass
class ModuleConfig:
    reduction: int = 16
    ratio: int = 8
    kernel_size: int = 3
    dropout_rt: float = 0.2
    activation: type = nn.ReLU

    # Loss Function Configuration
    focal_alpha: float = 0.5
    focal_gamma: float = 3.0

    # Dropout Configuration
    drop_prob: float = 0.1

    # Bottle Neck Blocks
    expansion: int = 2




