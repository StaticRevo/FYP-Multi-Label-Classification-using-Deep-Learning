import torch
import pytorch_lightning as pl

# Import your model class
from models.models import CustomModel  # Change this to your model's file & class name

model_weights = None  # Change this to the path of your model weights
model = CustomModel.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=model_weights, main_path=main_path)
# Set model to evaluation mode
model.eval()

print("✅ Model loaded from checkpoint!")
