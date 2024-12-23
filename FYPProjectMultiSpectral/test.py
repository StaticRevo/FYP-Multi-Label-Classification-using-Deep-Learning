import os
import subprocess
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import *
from config.config import clean_and_parse_labels, calculate_class_weights
from models.models import *
from utils.visualisation import (
    register_hooks, 
    clear_activations, 
    visualize_activations,
    activations  
)
# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')
set_random_seeds()

def test_model(
    checkpoint_path, 
    metadata_path, 
    dataset_dir, 
    model_class, 
    model_name, 
    dataset_name, 
    bands, 
    num_classes, 
    in_channels, 
    model_weights
):

    # Load metadata
    metadata_csv = pd.read_csv(metadata_path)
    class_labels = DatasetConfig.class_labels

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(
        bands=bands, 
        dataset_dir=dataset_dir, 
        metadata_csv=metadata_csv
    )
    data_module.setup(stage='test')

    # Calculate class weights
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    # Load your trained model
    model = model_class.load_from_checkpoint(
        checkpoint_path, 
        class_weights=class_weights, 
        num_classes=num_classes, 
        in_channels=in_channels, 
        model_weights=model_weights
    )
    model.eval()

    register_hooks(model)

    # # Set up Trainer for testing
    # trainer = pl.Trainer(
    #     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    #     devices=1 if torch.cuda.is_available() else None,
    #     precision='16-mixed',
    #     deterministic=True,
    # )

    # Run .test() on your model
    #trainer.test(model, datamodule=data_module)

    # # Collect predictions and true labels
    # all_preds = []
    # all_labels = []
            
    # for batch in tqdm(data_module.test_dataloader(), desc="Processing Batches"):
    #     inputs, labels = batch
    #     inputs = inputs.to(model.device)
    #     labels = labels.to(model.device)

    #     with torch.no_grad():
    #         logits = model(inputs)
    #         preds = torch.sigmoid(logits) > 0.5

    #     all_preds.extend(preds.cpu().numpy().astype(int))  # Convert boolean to int
    #     all_labels.extend(labels.cpu().numpy().astype(int))  # Ensure labels are int

    # # Convert lists to numpy arrays
    # all_preds = np.array(all_preds)
    # all_labels = np.array(all_labels)

    # # Save predictions and true labels
    # save_path = f'test_predictions_{model_name}_{dataset_name}.npz'
    # np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    test_loader = data_module.test_dataloader()
    example_batch = next(iter(test_loader))
    example_imgs, example_lbls = example_batch
    
    # Move to device
    example_imgs = example_imgs.to(model.device)

    print(example_imgs[0])
    # # Clear old activations before forward pass
    # clear_activations()

    # # Forward pass on first image (shape: [1, in_channels, H, W])
    # with torch.no_grad():
    #     _ = model(example_imgs[0].unsqueeze(0))

    # # Now 'activations' dictionary is populated
    # visualize_activations(num_filters=32)  # Show 8 filters per layer

if __name__ == "__main__":
    checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_rgb_bands_5%_BigEarthNet\final.ckpt'
    metadata_path = r'C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet\metadata_5_percent.csv'
    dataset_dir = DatasetConfig.dataset_paths["5"]
    model_class = BigEarthNetResNet18ModelTIF
    model_name = "ResNet18"
    dataset_name = "5%_BigEarthNet"
    bands = DatasetConfig.rgb_bands
    num_classes = 19
    in_channels = 3
    model_weights = 'ResNet18_Weights.DEFAULT'

    test_model(
        checkpoint_path, 
        metadata_path, 
        dataset_dir, 
        model_class, 
        model_name, 
        dataset_name, 
        bands, 
        num_classes, 
        in_channels, 
        model_weights
    )
