from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
from model_tif import BigEarthNetResNet18ModelTIF
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess

# Training the model
def main():
    data_module = BigEarthNetTIFDataModule()
    data_module.setup()

    model = BigEarthNetResNet18ModelTIF()

    # Initialize the logger
    log_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\logs'
    logger = TensorBoardLogger(log_dir, name=f"my_model_resnet50_eurosat_notpretrained_4channel")

    weights_info = 'none'
    checkpoint_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints'

    # Checkpoint callback for val_loss
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'resnet18-{weights_info}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Checkpoint callback for val_acc
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'resnet18-{weights_info}-{{epoch:02d}}-{{val_acc:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    # Model Training with custom callbacks
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=ModelConfig.num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_acc]
    )

    # Start training
    trainer.fit(model, data_module)

    # Run test after training
    trainer.test(model, datamodule=data_module)

    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

if __name__ == "__main__":
    main()