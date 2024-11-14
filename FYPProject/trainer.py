from pytorch_lightning.loggers import TensorBoardLogger
from models.resnet18.resnet18 import BigEarthNetSubsetModel
from dataloader import BigEarthNetSubsetDataModule
from callback import BigEarthNetSubsetCallback
import pytorch_lightning as pl
import torch
from config import ModelConfig

# Training the model

# Initialize the data module
data_module = BigEarthNetSubsetDataModule()
data_module.setup()

model = BigEarthNetSubsetModel()


# Initialize the logger
log_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\experiments\logs'
logger = TensorBoardLogger(log_dir, name="my_model_resnet18_eurosat_notpretrained")

checkpoint_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\experiments\checkpoints'
model_callback = BigEarthNetSubsetCallback(checkpoint_dir, ModelConfig.model_name)

# Model Training with custom callback
trainer = pl.Trainer(
    default_root_dir=checkpoint_dir,
    max_epochs=10,
    logger=logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None,
    log_every_n_steps=1,
    callbacks=[model_callback]
)

# Start training
trainer.fit(model, data_module)

