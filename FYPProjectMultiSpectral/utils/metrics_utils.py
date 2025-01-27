# Standard library imports
import os
import math
import random

# Third-party imports
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, f1_score, 
    hamming_loss, accuracy_score, multilabel_confusion_matrix
)
import rasterio

# Local application imports
from config.config import DatasetConfig
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap

def calculate_metrics_and_save_results(model, data_module, model_name, dataset_name, class_labels, result_path):
    all_preds, all_labels = [], []
    test_loader = data_module.test_dataloader()

    # Iterate through batches
    for batch in tqdm(test_loader, desc="Processing Batches"):
        inputs, labels = batch
        inputs, labels = inputs.to(model.device), labels.to(model.device)

        # Generate predictions
        with torch.no_grad():
            logits = model(inputs)
            preds = torch.sigmoid(logits) > 0.5

        all_preds.extend(preds.cpu().numpy().astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))

    # Convert lists to numpy arrays
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    # Save predictions and labels
    save_path = os.path.join(result_path, f'test_predictions_{model_name}_{dataset_name}.npz')
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    return all_preds, all_labels

def compute_aggregated_metrics(all_labels, all_preds):
    metrics_dict = {}
    
    # Micro-average: aggregates the contributions of all classes to compute the average metric
    metrics_dict['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    # Macro-average: computes metric independently for each class and then takes the average
    metrics_dict['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Hamming loss: fraction of labels incorrectly predicted
    metrics_dict['hamming_loss'] = hamming_loss(all_labels, all_preds)

    # Subset accuracy: only 1 if *all* labels match exactly
    metrics_dict['subset_accuracy'] = accuracy_score(all_labels, all_preds)

    return metrics_dict

def plot_roc_auc(all_labels, all_probs, class_labels):
    num_classes = all_labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute the ROC for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        plt.plot(
            fpr[i], tpr[i],
            lw=2,
            label=f'Class {class_labels[i]} (area = {roc_auc[i]:0.2f})'
        )

    # Plot micro-average
    plt.plot(
        fpr["micro"], tpr["micro"],
        color='deeppink', linestyle=':', linewidth=4,
        label=f'Micro-average (area = {roc_auc["micro"]:0.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-label ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_per_label_confusion_matrices_grid(all_labels, all_preds, class_names=None, cols=4):
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    n_labels = len(mcm)

    # Determine how many rows we need
    rows = math.ceil(n_labels / cols)

    # Create a figure with (rows x cols) subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    # If there's only 1 row, axes is not a 2D array; make it a list for consistency
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
    axes = axes.flatten()  # flatten in case we have multiple rows

    for i, matrix in enumerate(mcm):
        # Flatten the 2x2 matrix into TN, FP, FN, TP
        tn, fp, fn, tp = matrix.ravel()
        label_name = class_names[i] if class_names else f"Label {i}"

        # Plot a heatmap for this label's 2x2 matrix on the i-th subplot
        ax = axes[i]
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'],
            ax=ax
        )
        ax.set_title(f'{label_name}\n(TN={tn}, FP={fp}, FN={fn}, TP={tp})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  
    plt.show()

def plot_cooccurrence_matrix(all_labels, all_preds, class_names=None):
    num_classes = all_labels.shape[1]
    cooccur = np.zeros((num_classes, num_classes), dtype=int)

    # For each sample
    for n in range(all_labels.shape[0]):
        # find all true labels
        true_idxs = np.where(all_labels[n] == 1)[0]
        # find all predicted labels
        pred_idxs = np.where(all_preds[n] == 1)[0]
        # increment co-occurrences
        for i in true_idxs:
            for j in pred_idxs:
                cooccur[i, j] += 1

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cooccur,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(num_classes),
        yticklabels=class_names if class_names else range(num_classes),
        cbar_kws={'shrink': 0.75}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Multi-label Co-occurrence Matrix", fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2.0)
    plt.show()

    return cooccur