import sys
from pytorch_lightning.loggers import TensorBoardLogger
from models.resnet18.resnet18 import BigEarthNetResNet18Model
from models.resnet50.resnet50 import BigEarthNetResNet50Model
from models.vgg16.vgg16 import BigEarthNetVGG16Model
from dataloader import BigEarthNetSubsetDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from config import ModelConfig
import os

# Training the model
def main(model_name, weights):
    # Initialize the data module
    data_module = BigEarthNetSubsetDataModule()
    data_module.setup()

    # Initialize the model with the selected weights
    if model_name == 'ResNet18':
        if weights == 'None':
            model = BigEarthNetResNet18Model(weights=None)
        else:
            model = BigEarthNetResNet18Model(weights=weights)
    elif model_name == 'ResNet50':
        if weights == 'None':
            model = BigEarthNetResNet50Model(weights=None)
        else:
            model = BigEarthNetResNet50Model(weights=weights)
    elif model_name == 'VGG16':
        if weights == 'None':
            model = BigEarthNetVGG16Model(weights=None)
        else:
            model = BigEarthNetVGG16Model(weights=weights)

    # Initialize the logger
    log_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\experiments\logs'
    logger = TensorBoardLogger(log_dir, name=f"my_model_{model_name}_eurosat_notpretrained")

    weights_info = 'none' if weights == 'None' else 'pretrained'
    checkpoint_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProject\experiments\checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{weights_info}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Model Training with custom callback
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=ModelConfig.num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )

    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trainer.py <model_name> <weights>")
    else:
        model_name = sys.argv[1]
        weights = sys.argv[2]
        main(model_name, weights)