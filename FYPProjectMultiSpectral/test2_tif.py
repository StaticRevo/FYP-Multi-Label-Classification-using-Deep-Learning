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
from utils.test_functions import collect_predictions_and_plot_confusion_matrix, predict_and_display_random_image
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
    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=DatasetConfig.rgb_bands)
    data_module.setup(stage=None)

    # Load the trained model checkpoint
    checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\custom_model-custom_model_Weights.DEFAULT-epoch=00-val_acc=0.94.ckpt'
    model = CustomModel.load_from_checkpoint(checkpoint_path, class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3, weights='DEFAULT')

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
        inputs = inputs.to(model.device)  # Move inputs to the same device as the model
        labels = labels.to(model.device)  # Move labels to the same device as the model
        preds = model(inputs).sigmoid() > 0.5  # Apply sigmoid and threshold at 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save predictions and true labels to a file
    save_path = 'test_predictions.npz'
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    # Load predictions and true labels
    data = np.load(save_path)
    all_preds = data['all_preds']
    all_labels = data['all_labels']

    # Plot confusion matrix
    collect_predictions_and_plot_confusion_matrix(all_preds, all_labels, DatasetConfig)

    # Predict and display a random image
    dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet\CombinedImages'
    predict_and_display_random_image(model, dataset_dir, metadata_csv, bands=[2, 3, 4])


if __name__ == "__main__":
    main()