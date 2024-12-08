import json
import os
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
from dataset_tif import BigEarthNetDatasetTIF
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import sys
from utils.helper_functions import save_tensorboard_graphs
from transformations.transforms import TransformsConfig

if __name__ == '__main__':
    # Testing from the dataset class directly
    test_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'test']
    test_dataset = BigEarthNetDatasetTIF(df=test_df, root_dir=DatasetConfig.dataset_path, transforms=None, selected_bands=DatasetConfig.rgb_bands)

    image, label = test_dataset[5]
    print(f"Image shape: {image.size}, Label: {label}")
    print()

    test_img_path = os.path.join(DatasetConfig.dataset_path, "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_68.tif")
    # Call the get_label method and print the result
    label = test_dataset.get_label(test_img_path)
    print(f"Label for {test_img_path}: {label}")

    # # Testing from the dataloader
    # # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=DatasetConfig.rgb_bands)
    data_module.setup(stage=None)

    # # Get the test DataLoader
    test_dataloader = data_module.test_dataloader()

    # Create an iterator from the DataLoader
    test_iterator = iter(test_dataloader)

    # # Get the next batch
    inputs, labels = next(test_iterator)

    # # Print the inputs and labels
    print("Sample Inputs:", inputs[:2])
    print("Sample Labels:", labels[:2])