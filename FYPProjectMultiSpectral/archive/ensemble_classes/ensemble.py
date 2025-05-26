# Standard library imports
import pandas as pd

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy, 
    MultilabelHammingDistance, MultilabelFBetaScore, MultilabelRankingAveragePrecision
)

# Local application imports
from models.models import *
from config.config import ModelConfig, DatasetConfig
from config.config_utils import calculate_class_weights

# Ensemble model class
class EnsembleModel(nn.Module):
    def __init__(self, model_configs, weights=None, device=ModelConfig.device):
        super().__init__()
        self.device = device
        self.model_configs = model_configs  
        self.models = nn.ModuleList()

        # Initialize weights
        if weights is None:
            self.weights = torch.ones(len(model_configs), device=device, dtype=torch.float32) / len(model_configs)  # Equal weights
        else:
            assert len(weights) == len(model_configs), "Number of weights must match number of models"
            self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()  # Normalize to sum to 1

        # Build each model based on its configuration
        for config in model_configs:
            arch = config['arch']
            ckpt_path = config['ckpt_path']
            class_weights = config.get('class_weights', None)
            num_classes = config.get('num_classes', DatasetConfig.num_classes)
            in_channels = config.get('in_channels', 3)
            model_weights = config.get('model_weights', None)
            main_path = config.get('main_path', None)

            model = self._create_model(arch, class_weights, num_classes, in_channels, model_weights, main_path)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            model.to(self.device)
            self.models.append(model)

    def _create_model(self, arch, class_weights, num_classes, in_channels, model_weights, main_path):
        if arch == 'resnet18':
            return ResNet18(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'resnet50':
            return ResNet50(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'vgg16':
            return VGG16(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'vgg19':
            return VGG19(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'densenet121':
            return DenseNet121(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'efficientnetb0':
            return EfficientNetB0(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'efficientnet_v2':
            return EfficientNetV2(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'vit_transformer':
            return VitTransformer(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'swin_transformer':
            return SwinTransformer(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'custom_model9':
            return CustomModelV9(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'custom_model6':
            return CustomModelV6(class_weights, num_classes, in_channels, model_weights, main_path)
        else:
            raise ValueError(f"Unsupported architecture '{arch}'")

    @torch.no_grad()
    def forward(self, x):
        outputs = [model(x.to(self.device)) for model in self.models]  
        all_outputs = torch.stack(outputs, dim=0) 
        weighted_output = torch.einsum('m,mbs->bs', self.weights, all_outputs) 
        
        return weighted_output

    def get_configs(self):
        return self.model_configs

    def get_weights(self):
        return self.weights.tolist()
    
# Custom Ensemble Lightning Module for testing
class EnsembleLightningModule(pl.LightningModule):
    def __init__(self, model_configs, weights):
        super().__init__()
        self.ensemble = EnsembleModel(model_configs, weights=weights, device=self.device)
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.ensemble(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        self.test_step_outputs.append({
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'probs': probs.cpu()
        })
        return {'preds': preds, 'labels': labels, 'probs': probs}

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs], dim=0)
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs], dim=0)
        self.log_dict({
            'test_acc': MultilabelAccuracy(num_labels=DatasetConfig.num_classes)(all_preds, all_labels),
            'test_f1': MultilabelF1Score(num_labels=DatasetConfig.num_classes)(all_preds, all_labels),
            'test_f2': MultilabelFBetaScore(num_labels=DatasetConfig.num_classes, beta=2.0)(all_preds, all_labels),
            'test_precision': MultilabelPrecision(num_labels=DatasetConfig.num_classes)(all_preds, all_labels),
            'test_recall': MultilabelRecall(num_labels=DatasetConfig.num_classes)(all_preds, all_labels),
            'test_hamming_loss': MultilabelHammingDistance(num_labels=DatasetConfig.num_classes)(all_preds, all_labels),
            'test_avg_precision': MultilabelRankingAveragePrecision(num_labels=DatasetConfig.num_classes)(all_probs, all_labels)
        })
        self.test_step_outputs.clear()

    @staticmethod
    def load_from_checkpoint(checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        model_configs = checkpoint['model_configs']
        weights = checkpoint['weights']
        return EnsembleLightningModule(model_configs, weights)