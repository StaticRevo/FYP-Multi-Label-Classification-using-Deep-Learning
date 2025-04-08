# Standard library imports
import pandas as pd

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local application imports
from models.models import *
from config.config import ModelConfig, DatasetConfig
from config.config_utils import calculate_class_weights

# Ensemble model class
class EnsembleModel(nn.Module):
    def __init__(self, model_configs, device=ModelConfig.device):
        super().__init__()
        self.device = device
        self.model_configs = model_configs  
        self.models = nn.ModuleList()

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
        if arch == 'CustomModel':
            return CustomModel(class_weights, num_classes, in_channels, model_weights, main_path)
        elif arch == 'resnet18':
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
        else:
            raise ValueError(f"Unsupported architecture '{arch}'")

    @torch.no_grad()
    def forward(self, x):
        outputs = [model(x.to(self.device)) for model in self.models]
        all_outputs = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        avg_output = torch.mean(all_outputs, dim=0)  # [batch_size, num_classes]
        return avg_output

    def get_configs(self):
        """Return model_configs for reinitialization."""
        return self.model_configs