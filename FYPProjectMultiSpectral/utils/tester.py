import os
import sys
import json
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.helper_functions import *
from utils.test_functions import *
from utils.visualisation import *
from models.models import *

# Testing the model
def main():
    set_random_seeds()
    torch.set_float32_matmul_precision('high')

    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3] # string of selected bands ex. all_labels
    selected_dataset = sys.argv[4]
    acc_checkpoint_path = sys.argv[5]
    loss_checkpoint_path = sys.argv[6]
    last_checkpoint_path = sys.argv[7]
    in_channels = int(sys.argv[8])
    class_weights = json.loads(sys.argv[9])
    metadata_csv = pd.read_csv(sys.argv[10])
    dataset_dir = sys.argv[11]
    bands = json.loads(sys.argv[12]) # List of selected bands

   # Allow user to choose checkpoint
    checkpoint_choice = input(f"Select checkpoint to test:\n"
                              f"1. Best Accuracy ({acc_checkpoint_path})\n"
                              f"2. Best Loss ({loss_checkpoint_path})\n"
                              f"3. Final ({last_checkpoint_path})\n"
                              f"Choice [1/2/3]: ")
    if checkpoint_choice == "1":
        checkpoint_path = acc_checkpoint_path
    elif checkpoint_choice == "2":
        checkpoint_path = loss_checkpoint_path
    elif checkpoint_choice == "3":
        checkpoint_path = last_checkpoint_path
    else:
        print("Invalid choice. Defaulting to Last Saved checkpoint.")
        checkpoint_path = last_checkpoint_path

    print(f"\nUsing checkpoint: {checkpoint_path}\n")

    model_mapping = {
        'custom_model': (CustomModel, 'custom_model'),
        'ResNet18': (BigEarthNetResNet18ModelTIF, 'resnet18'),
        'ResNet50': (BigEarthNetResNet50ModelTIF, 'resnet50'),
        'VGG16': (BigEarthNetVGG16ModelTIF, 'vgg16'),
        'VGG19': (BigEarthNetVGG19ModelTIF, 'vgg19'),
        'DenseNet121': (BigEarthNetDenseNet121ModelTIF, 'densenet121'),
        'EfficientNetB0': (BigEarthNetEfficientNetB0ModelTIF, 'efficientnetb0'),
        'EfficientNet_v2': (BigEarthNetEfficientNetV2MModelTIF, 'efficientnet_v2'),
        'Vit-Transformer': (BigEarthNetVitTransformerModelTIF, 'vit_transformer'),
        'Swin-Transformer': (BigEarthNetSwinTransformerModelTIF, 'swin_transformer')
    }

    if model_name in model_mapping:
        model_class, _ = model_mapping[model_name]  
        model = model_class.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=weights)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    # Load model and data
    model, data_module, class_labels = load_model_and_data(
        checkpoint_path=checkpoint_path,
        metadata_csv=metadata_csv,
        dataset_dir=dataset_dir,
        model_class=model_class,
        bands=bands,
        in_channels=in_channels,
        model_weights=weights if weights != 'None' else None,
        num_classes=DatasetConfig.num_classes
    )

    # Calculate metrics and save results
    all_preds, all_labels = calculate_metrics_and_save_results(
        model=model,
        data_module=data_module,
        model_name=model_name,
        dataset_name=selected_dataset
    )
    # Visualize predictions and results
    visualize_predictions_and_heatmaps(
        model=model,
        data_module=data_module,
        all_preds=all_preds,
        all_labels=all_labels,
        class_labels=class_labels,
        model_name=model_name
    )

    # Generate Grad-CAM visualizations
    generate_gradcam_visualizations(
        model=model,
        data_module=data_module,
        class_labels=class_labels,
        model_name=model_name
    )

if __name__ == "__main__":
    main()