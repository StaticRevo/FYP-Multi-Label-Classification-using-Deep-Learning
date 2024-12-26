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

    if weights == 'None':
        weights = None
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
    data_module.setup(stage='test')

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
    
    class_labels = DatasetConfig.class_labels
    model.eval()

    register_hooks(model)
    
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

    per_class_metrics_path = f'test_per_class_metrics_ResNet.json'
    if os.path.exists(per_class_metrics_path):
        with open(per_class_metrics_path, 'r') as f:
            per_class_metrics = json.load(f)
        
        # Print per-class metrics with class labels
        print("\nPer-Class Metrics:")
        for metric, values in per_class_metrics.items():
            if metric == 'class_labels':
                continue  # Skip class_labels key
            print(f"\n{metric.capitalize()}:")
            for i, val in enumerate(values):
                class_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
                print(f"  {i} ({class_name}): {val:.4f}")
    else:
        print(f"\nPer-class metrics file not found at {per_class_metrics_path}")
    
if __name__ == "__main__":
    main()