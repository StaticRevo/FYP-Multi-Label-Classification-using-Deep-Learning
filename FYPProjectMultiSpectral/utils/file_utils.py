# Standard library imports
import os
import random

# Third-party imports
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchinfo import summary as torchinfo_summary
from contextlib import redirect_stdout

# Local application imports
from config.config import DatasetConfig, ModelConfig
from models.models import *

# Initialize path for saving results
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

# Save the hyperparameters of the model
def save_hyperparameters(model_config, experiment_main_path):
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
        f.write(f"  lr_factor     : {ModelConfig.lr_factor}\n")
        f.write(f"  lr_patience   : {ModelConfig.lr_patience}\n")
        f.write("\n")
        f.write("Additional Training Controls:\n")
        f.write(f"  patience      : {ModelConfig.patience}\n")
        f.write(f"  dropout       : {ModelConfig.dropout}\n")
        f.write(f"  device        : {ModelConfig.device}\n")
        f.write(f"Loss Function: {model_config.loss_fn}\n")
        
    return file_path

# Save the Model Architecture to a file
def save_model_architecture(model, input_size, hyperparams_file_path, filename="model_architecture"):
    save_dir = os.path.dirname(hyperparams_file_path)
    save_path = os.path.join(save_dir, f'{filename}.txt')
    os.makedirs(save_dir, exist_ok=True)
    
    device = ModelConfig.device
    model.to(device)
    
    with open(save_path, 'w', encoding='utf-8') as f: # Redirect stdout to write the summary to a file
        with redirect_stdout(f):
            torchinfo_summary(model, input_size=(1, *input_size))
            
    return save_path

