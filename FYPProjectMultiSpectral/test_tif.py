import os
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
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import *
from utils.test_functions import *
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

# Testing the model
def main():
    # Set float32 matmul precision to 'high' to utilize Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    acc_checkpoint_path = sys.argv[5]
    loss_checkpoint_path = sys.argv[6]
    in_channels = int(sys.argv[7])
    class_weights = sys.argv[8]
    metadata_csv = sys.argv[9]
    dataset_dir = sys.argv[10]
    bands = sys.argv[11]

    # Allow user to choose checkpoint
    checkpoint_choice = input(f"Select checkpoint to test:\n1. Best Accuracy ({acc_checkpoint_path})\n2. Best Loss ({loss_checkpoint_path})\nChoice [1/2]: ")
    if checkpoint_choice == "1":
        checkpoint_path = acc_checkpoint_path
    elif checkpoint_choice == "2":
        checkpoint_path = loss_checkpoint_path
    else:
        print("Invalid choice. Defaulting to Best Accuracy checkpoint.")
        checkpoint_path = loss_checkpoint_path

    print(f"Using checkpoint: {checkpoint_path}")

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
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

    results_dir = f"experiments/results/{model_name}_{selected_bands}_{weights}_{selected_dataset}"
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
    dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages'
    predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.7, bands=DatasetConfig.all_bands)
    
    # Select an image for GradCAM
    random_image_file = random.choice(metadata_csv[metadata_csv['split'] == 'test']['patch_id'].apply(lambda x: f"{x}.tif").tolist())
    image_path = os.path.join(dataset_dir, random_image_file)
    print()
    print(f"Selected image for GradCAM: {random_image_file}")
    selected_layer = 'model.layer4'
    display_gradcam_heatmap(model, image_path, DatasetConfig.class_labels, selected_layer, threshold=0.70)

    # Plot ROC AUC curve
    plot_roc_auc(all_labels, all_preds, DatasetConfig.class_labels)
    
if __name__ == "__main__":
    main()