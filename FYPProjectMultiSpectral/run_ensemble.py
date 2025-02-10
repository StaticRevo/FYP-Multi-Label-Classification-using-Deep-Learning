import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.ensemble import EnsembleModel
from utils.test_utils import calculate_metrics_and_save_results, visualize_predictions_and_heatmaps
from dataloader import BigEarthNetDataLoader
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelHammingDistance
import os
import json
from tqdm import tqdm
import logging

def run_ensemble_inference():
    device = ModelConfig.device
    num_classes = DatasetConfig.num_classes
    in_channels = 12
    model_weights = None
    metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['10'])
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    # Define the configurations for each trained model
    model_configs = [
        {
            'arch': 'resnet18',
            'ckpt_path': r'C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_10%_BigEarthNet_2epochs\checkpoints\final.ckpt',
            'class_weights': class_weights,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'model_weights': model_weights,
            'main_path': r'C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_10%_BigEarthNet_2epochs'
        },
        {
            'arch': 'resnet50',
            'ckpt_path': r'C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_2epochs\checkpoints\final.ckpt',
            'class_weights': class_weights,
            'num_classes': num_classes,
            'in_channels': in_channels,
            'model_weights': model_weights,
            'main_path': r'C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_2epochs'
        },
    ]

    # Build the ensemble
    ensemble = EnsembleModel(model_configs, device=device)
    ensemble.eval()
    
    # Load the data
    dataset_dir = DatasetConfig.dataset_paths['10']
    bands = DatasetConfig.all_bands  
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Initialize metrics (TorchMetrics)
    ensemble_accuracy = MultilabelAccuracy(num_labels=num_classes).to(device)
    ensemble_precision = MultilabelPrecision(num_labels=num_classes).to(device)
    ensemble_recall = MultilabelRecall(num_labels=num_classes).to(device)
    ensemble_f1 = MultilabelF1Score(num_labels=num_classes).to(device)
    ensemble_hamming_loss = MultilabelHammingDistance(num_labels=num_classes).to(device)

    # Per-class metrics
    ensemble_f1_per_class = MultilabelF1Score(num_labels=num_classes, average='none').to(device)
    ensemble_precision_per_class = MultilabelPrecision(num_labels=num_classes, average='none').to(device)
    ensemble_recall_per_class = MultilabelRecall(num_labels=num_classes, average='none').to(device)
    ensemble_accuracy_per_class = MultilabelAccuracy(num_labels=num_classes, average='none').to(device)

    # Get the total number of batches for tqdm
    total_batches = len(test_loader)
    logging.info(f"Starting inference on {total_batches} batches.")

    for batch in tqdm(test_loader, total=total_batches, desc="Inference Progress"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            logits = ensemble(inputs)  

        probs = torch.sigmoid(logits)  
        preds = (probs > 0.5).int()

        # Update overall metrics
        ensemble_accuracy.update(preds, labels)
        ensemble_precision.update(preds, labels)
        ensemble_recall.update(preds, labels)
        ensemble_f1.update(preds, labels)
        ensemble_hamming_loss.update(preds, labels)

        # Update per-class metrics
        ensemble_f1_per_class.update(preds, labels)
        ensemble_precision_per_class.update(preds, labels)
        ensemble_recall_per_class.update(preds, labels)
        ensemble_accuracy_per_class.update(preds, labels)

    # Compute final aggregated metrics
    accuracy = ensemble_accuracy.compute()
    precision = ensemble_precision.compute()
    recall = ensemble_recall.compute()
    f1 = ensemble_f1.compute()
    hamming_loss = ensemble_hamming_loss.compute()

    # Compute per-class metrics
    f1_per_class = ensemble_f1_per_class.compute()
    precision_per_class = ensemble_precision_per_class.compute()
    recall_per_class = ensemble_recall_per_class.compute()
    accuracy_per_class = ensemble_accuracy_per_class.compute()

    # Print overall metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Hamming Loss: {hamming_loss:.4f}")

    # Print per-class results
    for i in range(num_classes):
        class_name = DatasetConfig.class_labels[i] if i < len(DatasetConfig.class_labels) else f"Class {i}"
        print(f"Class {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}, Accuracy: {accuracy_per_class[i]:.4f}")

    combined_arch_name = "_".join([config['arch'] for config in model_configs])
    filename = f'ensemble_per_class_metrics_{combined_arch_name}.json'
    metrics_save_path = os.path.join('FYPProjectMultiSpectral', 'ensemble_results', filename)
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    metrics_to_save = {
        'precision': precision_per_class.cpu().tolist(),
        'recall': recall_per_class.cpu().tolist(),
        'f1': f1_per_class.cpu().tolist(),
        'accuracy': accuracy_per_class.cpu().tolist(),
        'hamming_loss': hamming_loss.cpu().item(),
        'class_labels': DatasetConfig.class_labels
    }

    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Ensemble per-class metrics saved to {metrics_save_path}")

if __name__ == "__main__":
    run_ensemble_inference()