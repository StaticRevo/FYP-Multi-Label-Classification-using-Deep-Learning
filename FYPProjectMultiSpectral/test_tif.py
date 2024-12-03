import json
import os
import subprocess
import sys
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
from utils.test_functions import collect_predictions_and_plot_confusion_matrix, display_image_and_labels
from config.config import clean_and_parse_labels

from models.CustomModel import CustomModel
from models.ResNet18 import BigEarthNetResNet18ModelTIF
from models.ResNet50 import BigEarthNetResNet50ModelTIF
from models.VGG16 import BigEarthNetVGG16ModelTIF
from models.VGG19 import BigEarthNetVGG19ModelTIF
from models.DenseNet121 import BigEarthNetDenseNet121ModelTIF
from models.EfficientNetB0 import BigEarthNetEfficientNetB0ModelTIF
from models.VisionTransformer import BigEarthNetVitTransformerModelTIF
from models.SwinTransformer import BigEarthNetSwinTransformerModelTIF

# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\metadata_0.5%_BigEarthNet.csv'
metadata_csv = pd.read_csv(metadata_path)
metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

class_labels = set()
for labels in metadata_csv['labels']:
    class_labels.update(labels)

# Testing the model
def main():
    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    acc_checkpoint_path = sys.argv[5]
    loss_checkpoint_path = sys.argv[6]
    in_channels = int(sys.argv[7])

    # Allow user to choose checkpoint
    checkpoint_choice = input(f"Select checkpoint to test:\n1. Best Accuracy ({acc_checkpoint_path})\n2. Best Loss ({loss_checkpoint_path})\nChoice [1/2]: ")
    if checkpoint_choice == "1":
        checkpoint_path = acc_checkpoint_path
    elif checkpoint_choice == "2":
        checkpoint_path = loss_checkpoint_path
    else:
        print("Invalid choice. Defaulting to Best Accuracy checkpoint.")
        checkpoint_path = loss_checkpoint_path

    print(f"Testing {model_name} with {weights} weights on bands: {selected_bands}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=DatasetConfig.all_bands)
    data_module.setup(stage=None)

    model_mapping = {
        'custom_model': (CustomModel, 'custom_model.png'),
        'ResNet18': (BigEarthNetResNet18ModelTIF, 'resnet18.png'),
        'ResNet50': (BigEarthNetResNet50ModelTIF, 'resnet50.png'),
        'VGG16': (BigEarthNetVGG16ModelTIF, 'vgg16.png'),
        'VGG19': (BigEarthNetVGG19ModelTIF, 'vgg19.png'),
        'DenseNet121': (BigEarthNetDenseNet121ModelTIF, 'densenet121.png'),
        'EfficientNetB0': (BigEarthNetEfficientNetB0ModelTIF, 'efficientnetb0.png'),
        'Vit-Transformer': (BigEarthNetVitTransformerModelTIF, 'vit_transformer.png'),
        'Swin-Transformer': (BigEarthNetSwinTransformerModelTIF, 'swin_transformer.png')
    }

    if model_name in model_mapping:
        model_class = model_mapping[model_name]
        model = model_class.load_from_checkpoint(checkpoint_path, class_weights=DatasetConfig.class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=weights)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
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