import os
import subprocess
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
from models.ResNet18 import BigEarthNetResNet18ModelTIF
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
from utils.test_functions import collect_predictions_and_plot_confusion_matrix, display_image_and_labels


# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\5PercentBigEarthNetSubset\metadata_5_percent.csv'
metadata_csv = pd.read_csv(metadata_path)

# Function to clean and parse labels
def clean_and_parse_labels(label_string):
    cleaned_labels = label_string.replace(" '", ", '").replace("[", "[").replace("]", "]")
    return ast.literal_eval(cleaned_labels)

metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

# Testing the model
def main():
    data_module = BigEarthNetTIFDataModule()
    data_module.setup(stage=None, bands=DatasetConfig.rgb_bands)

    # Load the trained model checkpoint
    checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\ResNet18-ResNet18_Weights.DEFAULT-epoch=07-val_acc=0.95.ckpt'
    model = BigEarthNetResNet18ModelTIF.load_from_checkpoint(checkpoint_path, class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3, model_weights='DEFAULT')
    
    # Model Testing with mixed precision
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='16-mixed'  
    )

    # Run test
    trainer.test(model, datamodule=data_module)

    # Collect predictions and plot confusion matrix
    collect_predictions_and_plot_confusion_matrix(model, data_module, DatasetConfig)

    image_path = r'C:\Users\isaac\Desktop\BigEarthTests\5PercentBigEarthNetSubset\CombinedImages\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_28_56.tif'
    display_image_and_labels(image_path, model, data_module.train_dataset.patch_to_labels)
 

if __name__ == "__main__":
    main()