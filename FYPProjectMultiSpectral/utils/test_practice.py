import os
import subprocess
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import get_labels_for_image, display_image, display_image_and_labels
from config.config import clean_and_parse_labels, calculate_class_weights

from models.models import *

# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')

def test_model(checkpoint_path, metadata_path, dataset_dir, model_class, model_name, dataset_name, bands, num_classes, in_channels, model_weights):
    metadata_csv = pd.read_csv(metadata_path)
    metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

    class_labels = set()
    for labels in metadata_csv['labels']:
        class_labels.update(labels)

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    model = model_class.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=num_classes, in_channels=in_channels, model_weights=model_weights)
    model.eval()

    # Model Testing with mixed precision
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='16-mixed'  
    )

    # Run test
    trainer.test(model, datamodule=data_module)

    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    for batch in tqdm(data_module.test_dataloader(), desc="Processing Batches"):
        inputs, labels = batch
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)

        with torch.no_grad():
            logits = model(inputs)  
            #print(f"Raw logits: {logits}")  
            preds = torch.sigmoid(logits) > 0.5
            #print(f"Sigmoid outputs: {torch.sigmoid(logits)}")  

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save predictions and true labels to a file
    save_path = f'test_predictions_{model_name}_{dataset_name}.npz'
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

if __name__ == "__main__":
    checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_0.5%_BigEarthNet\last.ckpt'
    metadata_path = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5_percent.csv'
    dataset_dir = DatasetConfig.dataset_paths["0.5"]
    model_class = BigEarthNetResNet18ModelTIF
    model_name = "ResNet18"
    dataset_name = "0.5%_BigEarthNet"
    bands = DatasetConfig.all_bands
    num_classes = 19
    in_channels = 12
    model_weights = 'ResNet18_Weights.DEFAULT'

    test_model(checkpoint_path, metadata_path, dataset_dir, model_class, model_name, dataset_name, bands, num_classes, in_channels, model_weights)