from models.resnet18.resnet18 import BigEarthNetResNet18Model
from dataloader import BigEarthNetSubsetDataModule
import pytorch_lightning as pl
from config import ModelConfig
import os
import torch
import subprocess
from PIL import Image

def main():
    # Define the log directory
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, 'FYPProject', 'experiments', 'logs')

    # Initialize the data module
    data_module = BigEarthNetSubsetDataModule()
    data_module.setup()

    # Load the trained model manually
    checkpoint_path = os.path.join(current_dir, 'FYPProject', 'experiments', 'checkpoints', 'ResNet50-none-epoch=00-val_loss=0.69.ckpt')

    # Load the model
    model = BigEarthNetResNet18Model.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_checkpointing=True,
        inference_mode=True,
        devices=1 if torch.cuda.is_available() else None,
    )

    test_loader = data_module.test_dataloader()
    trainer.test(model, test_loader)

    subprocess.run(['tensorboard', '--logdir', log_dir])

if __name__ == "__main__":
    main()





