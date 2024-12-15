import os
import sys
import json
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.helper_functions import *
from utils.test_functions import *
from models.models import *

# Testing the model
def main():
    set_random_seeds()
    torch.set_float32_matmul_precision('high')

    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    acc_checkpoint_path = sys.argv[5]
    loss_checkpoint_path = sys.argv[6]
    last_checkpoint_path = sys.argv[7]
    in_channels = int(sys.argv[8])
    class_weights = json.loads(sys.argv[9])
    metadata_csv = pd.read_csv(sys.argv[10])
    dataset_dir = sys.argv[11]
    bands = json.loads(sys.argv[12])

    # Allow user to choose checkpoint
    checkpoint_choice = input(f"Select checkpoint to test:\n1. Best Accuracy ({acc_checkpoint_path})\n2. Best Loss ({loss_checkpoint_path})\n3. Final ({last_checkpoint_path})\nChoice [1/2/3]: ")
    if checkpoint_choice == "1":
        checkpoint_path = acc_checkpoint_path
    elif checkpoint_choice == "2":
        checkpoint_path = loss_checkpoint_path
    elif checkpoint_choice == "3":
        checkpoint_path = last_checkpoint_path
    else:
        print("Invalid choice. Defaulting to Last Saved checkpoint.")
        checkpoint_path = last_checkpoint_path

    print()
    print(f"Using checkpoint: {checkpoint_path}")
    print()

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

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
    model_layer_mapping = {
        'custom_model': '0',
        'ResNet18': 'layer4',
        'ResNet50': 'layer4',
        'VGG16': 'features.29',
        'VGG19': 'features.35',
        'DenseNet121': 'features.denseblock4',
        'EfficientNetB0': 'features.7',
        'Vit-Transformer': 'blocks.11',  
        'Swin-Transformer': 'layers.3'
    }

    if model_name in model_mapping:
        model_class, _ = model_mapping[model_name]  
        model = model_class.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=weights)
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

    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    # Add progress bar using tqdm
    for batch in tqdm(data_module.test_dataloader(), desc="Processing Batches"):
        inputs, labels = batch
        inputs = inputs.to(model.device)  
        labels = labels.to(model.device)  
        preds = model(inputs).sigmoid() > 0.5  
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results_dir = f"FYPProjectMultiSpectral/experiments/results/{model_name}_{selected_bands}_{weights}_{selected_dataset}"
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions and true labels in an npz file
    save_path = os.path.join(results_dir, 'test_predictions.npz')
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    # Load predictions and true labels
    data = np.load(save_path)
    all_preds = data['all_preds']
    all_labels = data['all_labels']

    # Check the shapes
    print(f"Predictions shape: {all_preds.shape}")
    print(f"Labels shape: {all_labels.shape}")

    # Plot confusion matrix
    plot_confusion_matrix(all_preds, all_labels, DatasetConfig)
    plot_normalized_confusion_matrix(all_preds, all_labels, DatasetConfig)

    # Predict and display a random image
    predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.7, bands=DatasetConfig.all_bands)

    # Get the target layer for the selected model
    if model_name in model_layer_mapping:
        selected_layer = model_layer_mapping[model_name]
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    # Plot ROC AUC curve
    plot_roc_auc(all_labels, all_preds, DatasetConfig.class_labels)
    
if __name__ == "__main__":
    main()