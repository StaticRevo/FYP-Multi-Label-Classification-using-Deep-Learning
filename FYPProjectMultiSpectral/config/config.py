# -- Configuration file for the project --

# Standard library imports
from dataclasses import dataclass, field
import json
import os
import math

# Third-party imports
import pandas as pd
import torch
import torch.nn as nn

# Local application imports
from .config_utils import *

# -- Dataset Configuration --
@dataclass
class DatasetConfig:
    # -- Paths --
    base_path = os.path.join(os.path.dirname(__file__), "metadata")
    metadata_path = r'C:\Users\isaac\Desktop\BigEarthTests\100%_BigEarthNet\metadata_100_percent.csv'
    experiment_path = r'C:\Users\isaac\Desktop\experiments'
    subset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets'

    dataset_paths = {
        "0.5": r"C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages",
        "1": r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages",
        "5": r"C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\CombinedImages",
        "10": r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet\CombinedImages",
        "50": r"C:\Users\isaac\Desktop\BigEarthTests\50%_BigEarthNet\CombinedImages",
        "100": r"D:\100%_BigEarthNet"
    }
    metadata_paths = {
        "0.5": os.path.join(base_path, "metadata_0.5_percent.csv"),
        "1": os.path.join(base_path, "metadata_1_percent.csv"),
        "5": os.path.join(base_path, "metadata_5_percent.csv"),
        "10": os.path.join(base_path, "metadata_10_percent.csv"),
        "50": os.path.join(base_path, "metadata_50_percent.csv"),
    }

    # --- Metadata Filtering ---
    original_metdata_file: str = os.path.join(base_path, "metadata.parquet")
    unwanted_metadata_file: str = os.path.join(base_path, "metadata_for_patches_with_snow_cloud_or_shadow.parquet")
    unwanted_metadata_csv = pd.read_parquet(unwanted_metadata_file)

    # --- Class Labels ---
    class_labels = calculate_class_labels(pd.read_csv(metadata_path))
    class_labels = class_labels
    class_labels_dict = {label: idx for idx, label in enumerate(class_labels)}
    reversed_class_labels_dict = {idx: label for label, idx in class_labels_dict.items()}

     # --- Dataset Specifications ---
    num_classes: int = 19
    band_channels: int = 12
    img_size: int = 120
    image_height: int = 120
    image_width: int = 120

    # --- Band Sets ---
    rgb_bands = ["B04", "B03", "B02"]
    rgb_nir_bands = ["B04", "B03", "B02", "B08"]
    rgb_swir_bands = ["B04", "B03", "B02", "B11", "B12"]
    rgb_nir_swir_bands = ["B04", "B03", "B02", "B08", "B11", "B12"]
    all_imp_bands = [ "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    
    # --- Band Stats ---
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
    # --- Training Hyperparameters ---
    num_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-3
    
    # --- Learning Rate Scheduling ---
    lr_factor: float = 0.5
    lr_patience: int = 4 

    # --- Early Stopping ---
    patience: int = 10 

    # --- Architecture/Training Settings ---
    dropout: float = 0.5
    loss_fn: str = "CombinedFocalLossWithPosWeight"

    # --- Runtime ---
    num_workers: int = math.ceil(os.cpu_count() * 2 / 3)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
       
# -- Module Configuration --
@dataclass
class ModuleConfig:
    # --- Architectural Configs ---
    reduction: int = 16
    ratio: int = 8
    kernel_size: int = 3
    activation: type = nn.ReLU

    # --- Dropout Settings ---
    drop_prob: float = 0.1
    dropout_rt: float = 0.1

    # --- Bottleneck Design ---
    expansion: int = 2

    # -- Loss Function Settings ---
    focal_alpha: float = 0.5
    focal_gamma: float = 3.0


    




