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
from config.config import DatasetConfig
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

def initalize_paths_tester(model_name, weights, selected_bands, selected_dataset, epochs):
    experiment_path = DatasetConfig.experiment_path
    main_path = os.path.join(experiment_path, f"{model_name}_{weights}_{selected_bands}_{selected_dataset}_{epochs}epochs")
    
    if not os.path.exists(main_path):
        print(f"Path {main_path} does not exist. Trying with increments.")
        increment = 1
        while increment <= 5:
            new_main_path = f"{main_path}_{increment}"
            if os.path.exists(new_main_path):
                print(f"Found path: {new_main_path}")
                return new_main_path
            increment += 1
        raise Exception(f"Path not found after 5 increments. Last tried path: {new_main_path}")
    
    return main_path


