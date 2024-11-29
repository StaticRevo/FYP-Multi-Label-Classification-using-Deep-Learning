import os
import subprocess
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
from model_tif import BigEarthNetResNet18ModelTIF
import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from sklearn.metrics import multilabel_confusion_matrix

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
    data_module.setup()

    # Load the trained model checkpoint
    checkpoint_path = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints\resnet18-none-epoch=09-val_acc=0.98.ckpt'
    model = BigEarthNetResNet18ModelTIF.load_from_checkpoint(checkpoint_path)

    # Model Testing with mixed precision
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision=16  # Use mixed precision
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

    # Compute multilabel confusion matrix
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    print("Multilabel Confusion Matrix:")
    print(mcm)

    # Aggregate confusion matrices into a single 19x19 matrix
    cm = np.zeros((DatasetConfig.num_classes, DatasetConfig.num_classes), dtype=int)
    for i in range(DatasetConfig.num_classes):
        cm[i, 0] = mcm[i, 0, 0]  # True Negative
        cm[i, 1] = mcm[i, 0, 1]  # False Positive
        cm[i, 2] = mcm[i, 1, 0]  # False Negative
        cm[i, 3] = mcm[i, 1, 1]  # True Positive

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
if __name__ == "__main__":
    main()
