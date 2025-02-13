# Standard library imports
import os
import random

# Third-party imports
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Local application imports
from config.config import DatasetConfig, ModelConfig
from models.models import *

# Function to initialize paths for saving results
def initialize_paths(model_name, weights, selected_bands, selected_dataset, epochs):
    experiment_path = DatasetConfig.experiment_path
    main_path = os.path.join(experiment_path, f"{model_name}_{weights}_{selected_bands}_{selected_dataset}_{epochs}epochs")
    if os.path.exists(main_path):
        increment = 1
        new_main_path = f"{main_path}_{increment}"
        while os.path.exists(new_main_path):
            increment += 1
            new_main_path = f"{main_path}_{increment}"
        main_path = new_main_path
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    return main_path

def save_hyperparameters(model_config, experiment_main_path):
    # Ensure the experiment directory exists
    os.makedirs(experiment_main_path, exist_ok=True)
    file_path = os.path.join(experiment_main_path, "hyperparameters.txt")
    
    with open(file_path, "w") as f:
        f.write("Model Hyperparameters\n")
        f.write("Training Settings:\n")
        f.write(f"  num_epochs    : {ModelConfig.num_epochs}\n")
        f.write(f"  batch_size    : {ModelConfig.batch_size}\n")
        f.write(f"  learning_rate : {ModelConfig.learning_rate}\n")
        f.write(f"  momentum      : {ModelConfig.momentum}\n")
        f.write(f"  weight_decay  : {ModelConfig.weight_decay}\n")
        f.write("\n")
        f.write("Learning Rate Scheduler:\n")
        f.write(f"  lr_step_size  : {ModelConfig.lr_step_size}\n")
        f.write(f"  lr_factor     : {ModelConfig.lr_factor}\n")
        f.write(f"  lr_patience   : {ModelConfig.lr_patience}\n")
        f.write("\n")
        f.write("Additional Training Controls:\n")
        f.write(f"  patience      : {ModelConfig.patience}\n")
        f.write(f"  dropout       : {ModelConfig.dropout}\n")
        f.write(f"  device        : {ModelConfig.device}\n")
    
    return file_path


