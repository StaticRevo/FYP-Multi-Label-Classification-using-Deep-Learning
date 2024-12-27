# Standard library imports
import json
import os
import time
import subprocess
import sys

# Third-party imports
import pandas as pd
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from dataloader import BigEarthNetTIFDataModule
from utils.helper_functions import save_tensorboard_graphs, extract_number, set_random_seeds
from utils.visualisation import *
from models.models import *
from callbacks import BestMetricsCallback

# Training the model
def main():
    class_labels = DatasetConfig.class_labels
    print(f"Class labels: {class_labels}")
    metadata_path = DatasetConfig.metadata_paths['1']
    metadata_csv = pd.read_csv(metadata_path)   
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    model_name = 'ResNet18'
    weights = 'ResNet18_Weights.DEFAULT'
    selected_bands = 'all_bands'
    selected_dataset = '1%_BigEarthNet'
    best_acc_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\epoch=09-val_acc=0.82.ckpt'
    best_loss_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\epoch=09-val_loss=0.56.ckpt'
    last_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\final.ckpt'
    in_channels = 12
    class_weights = class_weights
    metadata_path = metadata_path
    dataset_dir = DatasetConfig.dataset_paths['1']
    bands = DatasetConfig.all_bands

    # Run test
    args = [
            'python', 
            'FYPProjectMultiSpectral\\tester.py', 
            model_name, 
            weights, 
            selected_bands, 
            selected_dataset, 
            best_acc_checkpoint_path, 
            best_loss_checkpoint_path, 
            last_checkpoint_path,
            str(in_channels),
            json.dumps(class_weights.tolist()),
            metadata_path, 
            dataset_dir, 
            json.dumps(bands)
    ]

    

    # Print the arguments
    print("Arguments to subprocess.run:")
    for arg in args:
        print(arg)

    # Run the subprocess
    subprocess.run(args)

if __name__ == "__main__":
    main()