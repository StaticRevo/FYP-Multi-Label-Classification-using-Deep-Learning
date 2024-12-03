from dataclasses import dataclass, field
import pandas as pd
from torchvision import transforms
import ast
import numpy as np  
import os

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5%_BigEarthNet.csv'
metadata_csv = pd.read_csv(metadata_path)

# Function to clean and parse labels
def clean_and_parse_labels(label_string):
    cleaned_labels = label_string.replace(" '", ", '").replace("[", "[").replace("]", "]")
    return ast.literal_eval(cleaned_labels)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

label_counts = metadata_csv['labels'].explode().value_counts()
total_counts = label_counts.sum()
class_weights = {label: total_counts / count for label, count in label_counts.items()}
class_weights_array = np.array([class_weights[label] for label in class_labels])


# Description: Configuration file for the project
@dataclass
class DatasetConfig:
    dataset_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages'
    metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5%_BigEarthNet.csv'
    unwanted_metadata_file: str = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    metadata_csv = pd.read_csv(metadata_path)
    unwanted_metadata_csv = pd.read_parquet(unwanted_metadata_file)
    img_size: int = 120
    img_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    img_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_classes: int = 19
    band_channels: int = 12
    valid_pct: float = 0.1
    class_labels = class_labels
    class_labels_dict = {label: idx for idx, label in enumerate(class_labels)}
    reversed_class_labels_dict = {idx: label for label, idx in class_labels_dict.items()}
    class_weights = class_weights_array

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

    def update_path(self, dataset_percentage):
        base_path = r'C:\Users\isaac\Desktop\BigEarthTests'
        self.dataset_path = os.path.join(base_path, f'{dataset_percentage}\CombinedImages')
        self.metadata_path = os.path.join(base_path, f'{dataset_percentage}\metadata_{dataset_percentage}.csv')
        
@dataclass
class ModelConfig:
    batch_size: int = 32
    num_epochs: int = 10
    model_name: str = 'resnet18'
    num_workers: int = os.cpu_count() // 2
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_step_size: int = 7
    lr_gamma: float = 0.1
    patience: int = 5

    model_names: list = field(default_factory=lambda: [
        'resnet18', 
        'resnet34', 
        'resnet50', 
        'resnet101', 
        'resnet152', 
        'densenet121', 
        'densenet169', 
        'densenet201', 
        'densenet161',
        'efficientnet-b0',
        'vgg16',
        'vgg19'
    ])


    
# if isinstance(metadata_csv['labels'].iloc[0], str):
#     metadata_csv['labels'] = metadata_csv['labels'].apply(ast.literal_eval)

# class_labels = metadata_csv['labels'].explode().unique()

# # Calculate class weights
# label_counts = metadata_csv['labels'].explode().value_counts()
# total_counts = label_counts.sum()
# class_weights = {label: total_counts / count for label, count in label_counts.items()}
# class_weights_array = np.array([class_weights[label] for label in class_labels])
