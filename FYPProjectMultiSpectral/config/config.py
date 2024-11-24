import os
import pandas as pd
import numpy as np
import ast
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    # Paths and dataset configuration
    dataset_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%'
    combined_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF'
    combined_rgb_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImagesJPG'
    metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\metadata_50_percent.csv'
    unwanted_metadata_file: str = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    img_size: int = 120
    img_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    img_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    num_classes: int = 19
    valid_pct: float = 0.1
    band_stats: dict = field(default_factory=lambda: {
        "mean": {
            "B02": 445.769,
            "B03": 626.906,
            "B04": 605.059,
            "B05": 971.651,
            "B06": 1821.982,
            "B07": 2108.096,
            "B08": 2256.322,
            "B8A": 2310.635,
            "B11": 1608.687,
            "B12": 1017.126
        },
        "std": {
            "B02": 648.438,
            "B03": 639.267,
            "B04": 717.575,
            "B05": 761.897,
            "B06": 1090.758,
            "B07": 1256.552,
            "B08": 1349.205,
            "B8A": 1287.112,
            "B11": 1057.335,
            "B12": 802.179
        }
    })

    # Fields populated during initialization
    metadata_csv: pd.DataFrame = field(init=False)
    unwanted_metadata: pd.DataFrame = field(init=False)
    class_weights: np.ndarray = field(init=False)
    class_labels_dict: dict = field(init=False)
    reversed_class_labels_dict: dict = field(init=False)

    def __post_init__(self):
        # Load metadata
        self.metadata_csv = pd.read_csv(self.metadata_path)
        self.unwanted_metadata = pd.read_parquet(self.unwanted_metadata_file)

        # Process labels if they are strings
        if isinstance(self.metadata_csv['labels'].iloc[0], str):
            self.metadata_csv['labels'] = self.metadata_csv['labels'].apply(ast.literal_eval)

        # Calculate class weights
        label_counts = self.metadata_csv['labels'].explode().value_counts()
        total_counts = label_counts.sum()
        self.class_weights = np.array([total_counts / count for count in label_counts])

        # Create label mappings
        self.class_labels_dict = {label: idx for idx, label in enumerate(label_counts.index)}
        self.reversed_class_labels_dict = {idx: label for label, idx in self.class_labels_dict.items()}


@dataclass
class ModelConfig:
    # Model and training configuration
    batch_size: int = 32
    num_epochs: int = 10
    model_name: str = 'resnet18'
    num_workers: int = os.cpu_count() // 2
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_step_size: int = 7
    lr_gamma: float = 0.1
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


# Example Usage
if __name__ == "__main__":
    # Initialize dataset config
    dataset_config = DatasetConfig()
    
    # Print debug information
    print(f"Number of metadata records: {dataset_config.metadata_csv.shape[0]}")
    print(f"Number of unwanted records: {dataset_config.unwanted_metadata.shape[0]}")
    print(f"Total records: {dataset_config.metadata_csv.shape[0] + dataset_config.unwanted_metadata.shape[0]}")

    # Print class weights and labels
    for label, weight in zip(dataset_config.class_labels_dict.keys(), dataset_config.class_weights):
        print(f"Class: {label}, Weight: {weight}")

    print(dataset_config.class_weights)
