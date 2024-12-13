import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy, MultilabelHammingDistance, MultilabelAveragePrecision
)
from torchsummary import summary
from torchviz import make_dot
import os
from config.config import ModelConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .modules.modules import *
from contextlib import redirect_stdout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseModel(pl.LightningModule):
    def __init__(self, model, num_classes, class_weights, in_channels):
        super(BaseModel, self).__init__()
        # Model 
        self.model = model
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        self.sigmoid = nn.Sigmoid() # Initialize Sigmoid layer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights) # Define loss function

        # Modules
        # self.se_block = SE(in_channels=in_channels, config=ModuleConfig)
       
        # Accuracy metrics
        self.train_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.val_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.test_acc = MultilabelAccuracy(num_labels=self.num_classes)
        # Recall metrics
        self.train_recall = MultilabelRecall(num_labels=self.num_classes)
        self.val_recall = MultilabelRecall(num_labels=self.num_classes)
        self.test_recall = MultilabelRecall(num_labels=self.num_classes)
        # Precision metrics
        self.train_precision = MultilabelPrecision(num_labels=self.num_classes)
        self.val_precision = MultilabelPrecision(num_labels=self.num_classes)
        self.test_precision = MultilabelPrecision(num_labels=self.num_classes)
        # F1 Score metrics
        self.train_f1 = MultilabelF1Score(num_labels=self.num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes)
        self.test_f1 = MultilabelF1Score(num_labels=self.num_classes)

        self.gradients = None

    def forward(self, x):
        x = self.model(x)
        # x = self.se_block(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=ModelConfig.learning_rate)
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
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = getattr(self, f'{phase}_acc')(logits, y)
        recall = getattr(self, f'{phase}_recall')(logits, y)
        f1 = getattr(self, f'{phase}_f1')(logits, y)
        precision = getattr(self, f'{phase}_precision')(logits, y)

        self.log(f'{phase}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{phase}_acc', acc, on_epoch=True, prog_bar=True)
        self.log(f'{phase}_recall', recall, on_epoch=True, prog_bar=True)
        self.log(f'{phase}_f1', f1, on_epoch=True, prog_bar=True)
        self.log(f'{phase}_precision', precision, on_epoch=True, prog_bar=True)

        return loss

    def on_epoch_end(self, phase):
        self.log(f'{phase}_acc_epoch', getattr(self, f'{phase}_acc').compute())
        self.log(f'{phase}_recall_epoch', getattr(self, f'{phase}_recall').compute())
        self.log(f'{phase}_f1_epoch', getattr(self, f'{phase}_f1').compute())
        self.log(f'{phase}_precision_epoch', getattr(self, f'{phase}_precision').compute())

        # Reset metrics
        getattr(self, f'{phase}_acc').reset()
        getattr(self, f'{phase}_recall').reset()
        getattr(self, f'{phase}_f1').reset()
        getattr(self, f'{phase}_precision').reset()

    def print_summary(self, input_size, filename):
        current_directory = os.getcwd()
        save_dir = os.path.join(current_directory, 'FYPProjectMultiSpectral', 'models', 'Architecture', filename)
        save_path = os.path.join(save_dir, f'{filename}_summary.txt)')
        os.makedirs(save_dir, exist_ok=True)  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Create a dummy input tensor with the specified input size
        dummy_input = torch.zeros(1, *input_size).to(device)

        # Redirect the summary output to a file
        with open(save_path, 'w') as f:
            with redirect_stdout(f):
                summary(self.model, input_size)

    def visualize_model(self, input_size, model_name):
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, 'FYPProjectMultiSpectral', 'models', 'Architecture', model_name)
        os.makedirs(save_path, exist_ok=True)  

        # Move the model to the correct device
        self.model.to(device)

        # Create a random tensor input based on the input size
        x = torch.randn(1, *input_size).to(device)  
        # Pass the tensor through the model
        y = self.model(x)

        # Create the visualization and save it at the specified path
        file_path = os.path.join(save_path, f'{model_name}')
        make_dot(y, params=dict(self.model.named_parameters())).render(file_path)
