from models.resnet18.resnet18 import BigEarthNetSubsetModel
from dataloader import BigEarthNetSubsetDataModule
import pytorch_lightning as pl
from config import ModelConfig
import os
import torch
import subprocess
from PIL import Image


print()
# Get the current directory
current_dir = os.getcwd()
print(current_dir)

# Define the log directory
log_dir = os.path.join(current_dir, 'FYPProject', 'experiments', 'logs')

# Initialize the data module
data_module = BigEarthNetSubsetDataModule()
data_module.setup()

subprocess.run(['tensorboard', '--logdir', log_dir])