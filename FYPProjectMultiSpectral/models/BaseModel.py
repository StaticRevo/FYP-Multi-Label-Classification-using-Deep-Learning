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
from config.config import DatasetConfig, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseModel(pl.LightningModule):
    def __init__(self, model, num_classes, class_weights):
        super(BaseModel, self).__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # Initialize Sigmoid layer
        self.sigmoid = nn.Sigmoid()
        
        # Define loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

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


    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=ModelConfig.learning_rate)

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
        # Ensure this is within a hook that receives 'outputs'
        self.log(f'{phase}_acc_epoch', getattr(self, f'{phase}_acc').compute())
        self.log(f'{phase}_recall_epoch', getattr(self, f'{phase}_recall').compute())
        self.log(f'{phase}_f1_epoch', getattr(self, f'{phase}_f1').compute())
        self.log(f'{phase}_precision_epoch', getattr(self, f'{phase}_precision').compute())

        # Reset metrics
        getattr(self, f'{phase}_acc').reset()
        getattr(self, f'{phase}_recall').reset()
        getattr(self, f'{phase}_f1').reset()
        getattr(self, f'{phase}_precision').reset()

    def print_summary(self, input_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
    
        # Create a dummy input tensor with the specified input size
        dummy_input = torch.zeros(1, *input_size).to(device)
        summary(self.model, input_size)

    def visualize_model(self, input_size, model_name):
        # Get the current working directory path
        current_directory = os.getcwd()
        save_path = os.path.join(current_directory, 'FYPProjectMultiSpectral', 'models', 'Architecture')
        os.makedirs(save_path, exist_ok=True)  

        # Move the model to the correct device
        self.model.to(device)

        # Create a random tensor input based on the input size
        x = torch.randn(1, *input_size).to(device)  
        # Pass the tensor through the model
        y = self.model(x)

        # Create the visualization and save it at the specified path
        file_path = os.path.join(save_path, f'{model_name}')
        make_dot(y, params=dict(self.model.named_parameters())).render(file_path, format="png")
