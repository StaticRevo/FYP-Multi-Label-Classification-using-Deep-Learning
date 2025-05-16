# Standard library imports
import os
import json
import logging

# Third-party imports
import torch
import pandas as pd
import numpy as np
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy, 
    MultilabelHammingDistance, MultilabelFBetaScore, MultilabelRankingAveragePrecision
)   
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, hamming_loss, accuracy_score, average_precision_score
from tqdm import tqdm

# Local application imports
from archive.ensemble_classes.ensemble import EnsembleModel
from dataloader import BigEarthNetDataLoader
from config.config import DatasetConfig, ModelConfig, calculate_class_weights

logging.basicConfig(level=logging.INFO) # Set up logging with suppression of rasterio GDAL errors
logger = logging.getLogger()
logging.getLogger("rasterio._env").setLevel(logging.WARNING) 

# Compute aggregated metrics with sklearn
def compute_aggregated_metrics(all_labels, all_preds, all_probs=None, logger=None):
    logger.info("Computing aggregated metrics with sklearn...")
    metrics_dict = {}
    
    metrics_dict['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f2_micro'] = fbeta_score(all_labels, all_preds, beta=2.0, average='micro', zero_division=0)
    metrics_dict['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f2_macro'] = fbeta_score(all_labels, all_preds, beta=2.0, average='macro', zero_division=0)
    metrics_dict['hamming_loss'] = hamming_loss(all_labels, all_preds)
    metrics_dict['subset_accuracy'] = accuracy_score(all_labels, all_preds)

    if all_probs is not None:
        metrics_dict['avg_precision'] = average_precision_score(all_labels, all_probs, average='micro')
        top_pred_labels = np.argmax(all_probs, axis=1)
        true_positives = np.any(all_labels > 0, axis=1)
        top_correct = np.array([all_labels[i, top_pred_labels[i]] for i in range(len(top_pred_labels))])
        if true_positives.sum() > 0:
            metrics_dict['one_error'] = 1 - (top_correct.sum() / true_positives.sum())
        else:
            metrics_dict['one_error'] = 0.0
            logger.warning("No positive labels in the dataset; one_error set to 0.")
    else:
        logger.warning("Probability outputs not provided. Skipping avg_precision and one_error.")

    return metrics_dict

def run_ensemble_inference():
    device = ModelConfig.device
    num_classes = DatasetConfig.num_classes
    in_channels = 12
    model_weights = None
    metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['10'])
    class_weights = calculate_class_weights(metadata_csv)  

    # Define the configurations for each trained model
    model_configs = [
        {
            'arch': 'custom_model9',
            'ckpt_path':  r"C:\Users\isaac\Desktop\experiments\CustomModelV9_None_all_bands_10%_BigEarthNet_50epochs_1\checkpoints\last.ckpt",
            'class_weights': class_weights,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'model_weights': model_weights,
            'main_path': r'C:\Users\isaac\Desktop\experiments\CustomModelV9_None_all_bands_10%_BigEarthNet_50epochs_1'
        }
        ,{
            'arch': 'custom_model6',
            'ckpt_path':  r"C:\Users\isaac\Desktop\experiments\CustomModelV6_None_all_bands_10%_BigEarthNet_50epochs\checkpoints\last.ckpt",
            'class_weights': class_weights,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'model_weights': model_weights,
            'main_path': r'C:\Users\isaac\Desktop\experiments\CustomModelV6_None_all_bands_10%_BigEarthNet_50epochs'
        }

    ]
    ensemble_weights = [0.82, 0.85] # Example weights for the models in the ensemble depending on their performance

    # Build the ensemble with weights
    ensemble = EnsembleModel(model_configs, weights=ensemble_weights, device=device)
    ensemble.eval()

    # Save ensemble as .ckpt
    checkpoint = {
        "state_dict": ensemble.state_dict(),
        "model_configs": model_configs,
        "weights": ensemble.get_weights()  # Save normalized weights
    }
    arch_names = "_".join([config['arch'] for config in model_configs])
    ensemble_dir = os.path.join('FYPProjectMultiSpectral', 'ensemble_results', arch_names)
    os.makedirs(ensemble_dir, exist_ok=True)
    ckpt_filename = f"ensemble_model_{arch_names}.ckpt"
    ckpt_path = os.path.join(ensemble_dir, ckpt_filename)
    torch.save(checkpoint, ckpt_path)
    print(f"Ensemble checkpoint saved to {ckpt_path}")
    print(f"Model weights: {ensemble.get_weights()}")

    # Load the data
    dataset_dir = DatasetConfig.dataset_paths['10']
    bands = DatasetConfig.all_bands
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Initialize torchmetrics
    ensemble_accuracy = MultilabelAccuracy(num_labels=num_classes).to(device)
    ensemble_precision = MultilabelPrecision(num_labels=num_classes).to(device)
    ensemble_recall = MultilabelRecall(num_labels=num_classes).to(device)
    ensemble_f1 = MultilabelF1Score(num_labels=num_classes).to(device)
    ensemble_f2 = MultilabelFBetaScore(num_labels=num_classes, beta=2.0).to(device)
    ensemble_avg_precision = MultilabelRankingAveragePrecision(num_labels=num_classes).to(device)
    ensemble_hamming_loss = MultilabelHammingDistance(num_labels=num_classes).to(device)

    # Per-class metrics
    ensemble_f1_per_class = MultilabelF1Score(num_labels=num_classes, average='none').to(device)
    ensemble_f2_per_class = MultilabelFBetaScore(num_labels=num_classes, beta=2.0, average='none').to(device)
    ensemble_precision_per_class = MultilabelPrecision(num_labels=num_classes, average='none').to(device)
    ensemble_recall_per_class = MultilabelRecall(num_labels=num_classes, average='none').to(device)
    ensemble_accuracy_per_class = MultilabelAccuracy(num_labels=num_classes, average='none').to(device)

    # Collect predictions and labels for sklearn
    all_preds = []
    all_labels = []
    all_probs = []

    # Inference loop
    total_batches = len(test_loader)
    logging.info(f"Starting inference on {total_batches} batches.")
    for batch in tqdm(test_loader, total=total_batches, desc="Inference Progress"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            logits = ensemble(inputs)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        ensemble_accuracy.update(preds, labels)
        ensemble_precision.update(preds, labels)
        ensemble_recall.update(preds, labels)
        ensemble_f1.update(preds, labels)
        ensemble_f2.update(preds, labels)
        ensemble_avg_precision.update(probs, labels)
        ensemble_hamming_loss.update(preds, labels)
        ensemble_f1_per_class.update(preds, labels)
        ensemble_f2_per_class.update(preds, labels)
        ensemble_precision_per_class.update(preds, labels)
        ensemble_recall_per_class.update(preds, labels)
        ensemble_accuracy_per_class.update(preds, labels)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # Concatenate collected data
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # Save predictions, labels, and probabilities as an .npz file
    predictions_filename = f'ensemble_predictions_{arch_names}.npz'
    predictions_save_path = os.path.join(ensemble_dir, predictions_filename)

    np.savez(
        predictions_save_path,
        predictions=all_preds,
        labels=all_labels,
        probabilities=all_probs
    )
    print(f"Predictions saved to {predictions_save_path}")

    # Compute torchmetrics
    accuracy = ensemble_accuracy.compute()
    precision = ensemble_precision.compute()
    recall = ensemble_recall.compute()
    f1 = ensemble_f1.compute()
    f2 = ensemble_f2.compute()
    avg_precision = ensemble_avg_precision.compute()
    hamming_loss_tm = ensemble_hamming_loss.compute()
    f1_per_class = ensemble_f1_per_class.compute()
    f2_per_class = ensemble_f2_per_class.compute()
    precision_per_class = ensemble_precision_per_class.compute()
    recall_per_class = ensemble_recall_per_class.compute()
    accuracy_per_class = ensemble_accuracy_per_class.compute()

    # Compute sklearn metrics
    sklearn_metrics = compute_aggregated_metrics(all_labels, all_preds, all_probs, logging.getLogger())

    # Print torchmetrics results
    print(f"\nTorchmetrics Overall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Hamming Loss: {hamming_loss_tm:.4f}")

    print(f"\nTorchmetrics Per-Class Metrics:")
    for i in range(num_classes):
        class_name = DatasetConfig.class_labels[i] if i < len(DatasetConfig.class_labels) else f"Class {i}"
        print(f"Class {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}, F2: {f2_per_class[i]:.4f}, Accuracy: {accuracy_per_class[i]:.4f}")

    # Print sklearn results
    print(f"\nSklearn Aggregated Metrics:")
    for metric, value in sklearn_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # Save combined metrics
    metrics_filename = f'ensemble_metrics_{arch_names}.json'
    metrics_save_path = os.path.join(ensemble_dir, metrics_filename)

    metrics_to_save = {
        'torchmetrics': {
            'overall': {
                'accuracy': accuracy.cpu().item(),
                'precision': precision.cpu().item(),
                'recall': recall.cpu().item(),
                'f1': f1.cpu().item(),
                'f2': f2.cpu().item(),
                'avg_precision': avg_precision.cpu().item(),
                'hamming_loss': hamming_loss_tm.cpu().item()
            },
            'per_class': {
                'precision': precision_per_class.cpu().tolist(),
                'recall': recall_per_class.cpu().tolist(),
                'f1': f1_per_class.cpu().tolist(),
                'f2': f2_per_class.cpu().tolist(),
                'accuracy': accuracy_per_class.cpu().tolist()
            }
        },
        'sklearn': sklearn_metrics,
        'class_labels': DatasetConfig.class_labels
    }

    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Combined metrics saved to {metrics_save_path}")

if __name__ == "__main__":
    run_ensemble_inference()