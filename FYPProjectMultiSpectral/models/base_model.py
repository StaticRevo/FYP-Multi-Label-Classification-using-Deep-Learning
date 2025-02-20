# Standard library imports
import os
import json
from contextlib import redirect_stdout
import sys

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy, 
    MultilabelHammingDistance, MultilabelAveragePrecision, MultilabelFBetaScore,
    MultilabelRankingAveragePrecision
)
from torchsummary import summary
from torchinfo import summary as torchinfo_summary
from torchviz import make_dot
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from prettytable import PrettyTable 
import numpy as np

# Local application imports
from config.config import ModelConfig, DatasetConfig, ModuleConfig
from models.modules import *
from .Metrics.one_error import OneError
from models.losses import CombinedFocalLossWithPosWeight

# Base model class for all models
device = ModelConfig.device

# Base model class for all models
class BaseModel(pl.LightningModule):
    def __init__(self, model, num_classes, class_weights, in_channels, metrics_save_dir):
        super(BaseModel, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.class_labels = DatasetConfig.class_labels  
        self.metrics_save_dir = metrics_save_dir

        self.criterion = CombinedFocalLossWithPosWeight(self.class_weights, alpha=ModuleConfig.focal_alpha, gamma=ModuleConfig.focal_gamma, reduction='mean')
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights) 
        
        # Aggregate Metrics
        self.train_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.val_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.test_acc = MultilabelAccuracy(num_labels=self.num_classes)

        self.train_recall = MultilabelRecall(num_labels=self.num_classes)
        self.val_recall = MultilabelRecall(num_labels=self.num_classes)
        self.test_recall = MultilabelRecall(num_labels=self.num_classes)

        self.train_precision = MultilabelPrecision(num_labels=self.num_classes)
        self.val_precision = MultilabelPrecision(num_labels=self.num_classes)
        self.test_precision = MultilabelPrecision(num_labels=self.num_classes)

        self.train_f1 = MultilabelF1Score(num_labels=self.num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes)
        self.test_f1 = MultilabelF1Score(num_labels=self.num_classes)

        self.train_f2 = MultilabelFBetaScore(num_labels=self.num_classes, beta=2.0)
        self.val_f2 = MultilabelFBetaScore(num_labels=self.num_classes, beta=2.0)
        self.test_f2 = MultilabelFBetaScore(num_labels=self.num_classes, beta=2.0)

        self.train_one_error = OneError(num_labels=self.num_classes)
        self.val_one_error = OneError(num_labels=self.num_classes)
        self.test_one_error = OneError(num_labels=self.num_classes)

        self.train_avg_precision = MultilabelRankingAveragePrecision(num_labels=self.num_classes)
        self.val_avg_precision = MultilabelRankingAveragePrecision(num_labels=self.num_classes)
        self.test_avg_precision = MultilabelRankingAveragePrecision(num_labels=self.num_classes)

        self.hamming_loss = MultilabelHammingDistance(num_labels=self.num_classes)

        # Per-Class Metrics for Testing Only
        self.test_precision_per_class = MultilabelPrecision(num_labels=self.num_classes, average='none')
        self.test_recall_per_class = MultilabelRecall(num_labels=self.num_classes, average='none')
        self.test_f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average='none')
        self.test_acc_per_class = MultilabelAccuracy(num_labels=self.num_classes, average='none')
        self.test_f2_per_class = MultilabelFBetaScore(num_labels=self.num_classes, average='none', beta=2.0)

        # Per-Class Metrics for Training Only
        self.train_precision_per_class = MultilabelPrecision(num_labels=self.num_classes, average='none')
        self.train_recall_per_class = MultilabelRecall(num_labels=self.num_classes, average='none')
        self.train_f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average='none')
        self.train_acc_per_class = MultilabelAccuracy(num_labels=self.num_classes, average='none')
        self.train_f2_per_class = MultilabelFBetaScore(num_labels=self.num_classes, average='none', beta=2.0)

        # Per-Class Metrics for Validation Only
        self.val_precision_per_class = MultilabelPrecision(num_labels=self.num_classes, average='none')
        self.val_recall_per_class = MultilabelRecall(num_labels=self.num_classes, average='none')
        self.val_f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average='none')
        self.val_f2_per_class = MultilabelFBetaScore(num_labels=self.num_classes, average='none', beta=2.0)
        self.val_acc_per_class = MultilabelAccuracy(num_labels=self.num_classes, average='none')
      
        # Initialize per-class metrics storage for all phases
        self.per_class_metrics = {
            'train': {
                'precision': [],
                'recall': [],
                'f1': [],
                'f2': [],
                'accuracy': []
            },
            'val': {
                'precision': [],
                'recall': [],
                'f1': [],
                'f2': [],
                'accuracy': []
            },
            'test': {
                'precision': [],
                'recall': [],
                'f1': [],
                'f2': [],
                'accuracy': []
            }
        }
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=ModelConfig.lr_factor, patience=ModelConfig.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def cross_entropy_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def _step(self, batch, batch_idx, phase):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        # Convert logits to binary predictions
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        # Convert to int for metrics calculation
        preds = preds.int()
        y = y.int()

        # Calculate aggregate metrics
        acc = getattr(self, f'{phase}_acc')(preds, y)
        recall = getattr(self, f'{phase}_recall')(preds, y)
        f1 = getattr(self, f'{phase}_f1')(preds, y)
        f2 = getattr(self, f'{phase}_f2')(preds, y)
        precision = getattr(self, f'{phase}_precision')(preds, y)
        one_error = getattr(self, f'{phase}_one_error')(probs, y)
        hamming_distance = self.hamming_loss(preds, y)
        hamming_loss_val = hamming_distance / y.size(1)

        # Log aggregate metrics
        self.log(f'{phase}_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_acc', acc, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_recall', recall, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_f1', f1, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_f2', f2, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_precision', precision, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_one_error', one_error, on_epoch=True, prog_bar=True, batch_size=len(x))
        self.log(f'{phase}_hamming_loss', hamming_loss_val, on_epoch=True, prog_bar=True, batch_size=len(x))

        if phase == 'train':
            avg_precision = self.train_avg_precision(probs, y)
            self.log('train_avg_precision', avg_precision, on_epoch=True, prog_bar=True, batch_size=len(x))
        elif phase == 'val':
            avg_precision = self.val_avg_precision(probs, y)
            self.log('val_avg_precision', avg_precision, on_epoch=True, prog_bar=True, batch_size=len(x))
        elif phase == 'test':
            avg_precision = self.test_avg_precision(probs, y)
            self.log('test_avg_precision', avg_precision, on_epoch=True, prog_bar=True, batch_size=len(x))

        # Compute and log per-class metrics
        if phase in ['train', 'val', 'test']:
            if phase == 'test':
                per_class_precision = self.test_precision_per_class(preds, y).cpu().tolist()
                per_class_recall = self.test_recall_per_class(preds, y).cpu().tolist()
                per_class_f1 = self.test_f1_per_class(preds, y).cpu().tolist()
                per_class_f2 = self.test_f2_per_class(preds, y).cpu().tolist()
                per_class_acc = self.test_acc_per_class(preds, y).cpu().tolist()
            elif phase == 'val':
                per_class_precision = self.val_precision_per_class(preds, y).cpu().tolist()
                per_class_recall = self.val_recall_per_class(preds, y).cpu().tolist()
                per_class_f1 = self.val_f1_per_class(preds, y).cpu().tolist()
                per_class_f2 = self.val_f2_per_class(preds, y).cpu().tolist()
                per_class_acc = self.val_acc_per_class(preds, y).cpu().tolist()
            elif phase == 'train':
                per_class_precision = self.train_precision_per_class(preds, y).cpu().tolist()
                per_class_recall = self.train_recall_per_class(preds, y).cpu().tolist()
                per_class_f1 = self.train_f1_per_class(preds, y).cpu().tolist()
                per_class_f2 = self.train_f2_per_class(preds, y).cpu().tolist()
                per_class_acc = self.train_acc_per_class(preds, y).cpu().tolist()

            # Store metrics
            self.per_class_metrics[phase]['precision'].append(per_class_precision)
            self.per_class_metrics[phase]['recall'].append(per_class_recall)
            self.per_class_metrics[phase]['f1'].append(per_class_f1)
            self.per_class_metrics[phase]['f2'].append(per_class_f2)
            self.per_class_metrics[phase]['accuracy'].append(per_class_acc)

            # Log per-class metrics to the logger (e.g., TensorBoard)
            for i in range(self.num_classes):
                class_name = self.class_labels[i] if i < len(self.class_labels) else f"Class {i}"
                self.log(f'{phase}_precision_class_{i}', per_class_precision[i], on_epoch=True, prog_bar=False, batch_size=len(x))
                self.log(f'{phase}_recall_class_{i}', per_class_recall[i], on_epoch=True, prog_bar=False, batch_size=len(x))
                self.log(f'{phase}_f1_class_{i}', per_class_f1[i], on_epoch=True, prog_bar=False, batch_size=len(x))
                self.log(f'{phase}_f2_class_{i}', per_class_f2[i], on_epoch=True, prog_bar=False, batch_size=len(x))
                self.log(f'{phase}_acc_class_{i}', per_class_acc[i], on_epoch=True, prog_bar=False, batch_size=len(x))

        return loss
    
    def on_train_epoch_end(self):
        self._save_epoch_metrics('train')

    def on_validation_epoch_end(self):
        self._save_epoch_metrics('val')

    def on_test_epoch_end(self):
        self._save_epoch_metrics('test')

    def _save_epoch_metrics(self, phase):
        # Check if per-class metrics have been collected
        if not any(self.per_class_metrics[phase].values()):
            print(f"No per-class metrics were collected during {phase}ing.")
            return
        
        # Aggregate per-class metrics by averaging over batches
        avg_precision = np.mean(self.per_class_metrics[phase]['precision'], axis=0)
        avg_recall = np.mean(self.per_class_metrics[phase]['recall'], axis=0)
        avg_f1 = np.mean(self.per_class_metrics[phase]['f1'], axis=0)
        avg_f2 = np.mean(self.per_class_metrics[phase]['f2'], axis=0)
        avg_acc = np.mean(self.per_class_metrics[phase]['accuracy'], axis=0)

        table = PrettyTable() # Create a PrettyTable for displaying per-class metrics with labels
        table.field_names = ["Class Index", "Class Name", "Precision", "Recall", "F1 Score", "F2 Score", "Accuracy"]

        for i in range(self.num_classes):
            class_name = self.class_labels[i] if i < len(self.class_labels) else f"Class {i}"
            table.add_row([
                f"{i}", 
                class_name,
                round(avg_precision[i], 4),
                round(avg_recall[i], 4),
                round(avg_f1[i], 4),
                round(avg_f2[i], 4),
                round(avg_acc[i], 4)
            ])

        print(f"\n{phase.capitalize()} Metrics per Class:\n{table}")

        # Save these metrics to a JSON file with class labels
        metrics_to_save = {
            'precision': avg_precision.tolist(),
            'recall': avg_recall.tolist(),
            'f1': avg_f1.tolist(),
            'f2': avg_f2.tolist(),
            'accuracy': avg_acc.tolist(),
            'class_labels': self.class_labels  
        }
        save_path = os.path.join(self.metrics_save_dir, 'results', f'{phase}_per_class_metrics_{self.model.__class__.__name__}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"{phase.capitalize()} per-class metrics saved to {save_path}")

        # Reset the metrics storage for the next epoch
        self.per_class_metrics[phase] = {
            'precision': [],
            'recall': [],
            'f1': [],
            'f2': [],
            'accuracy': []
        }
    
    # Print the model summary
    def print_summary(self, input_size, filename):
        current_directory = os.getcwd()
        save_dir = os.path.join(current_directory, 'FYPProjectMultiSpectral', 'models', 'Architecture', filename)
        save_path = os.path.join(save_dir, f'{filename}_summary.txt')
        os.makedirs(save_dir, exist_ok=True)  

        device = ModelConfig.device
        self.model.to(device)

        # Redirect the summary output to a file
        with open(save_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                torchinfo_summary(self.model, input_size=(1, *input_size))

    # Visualize the model architecture as a graph
    def visualize_model(self, input_size, model_name):
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, 'FYPProjectMultiSpectral', 'models', 'Architecture', model_name)
        os.makedirs(save_path, exist_ok=True)  
        self.model.to(device) # Move model to the correct device

        x = torch.randn(1, *input_size).to(device) # Create a random tensor input based on the input size
        y = self.model(x) # Pass the tensor through the model

        # Create the visualization and save it at the specified path
        file_path = os.path.join(save_path, f'{model_name}')
        make_dot(y, params=dict(self.model.named_parameters())).render(file_path)