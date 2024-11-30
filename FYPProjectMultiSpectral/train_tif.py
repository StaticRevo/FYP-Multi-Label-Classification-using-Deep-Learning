from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import sys

#from model_tif import BigEarthNetResNet18ModelTIF
from models.ResNet18 import BigEarthNetResNet18ModelTIF
from models.CustomModel import CustomModel
from models.VGG16 import BigEarthNetVGG16ModelTIF

# Training the model
def main():
    torch.set_float32_matmul_precision('high')

    # Initalising the variables from the command line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]

    # Determine the number of input channels based on the selected bands
    if selected_bands == 'all_bands':
        in_channels = len(DatasetConfig.all_bands)
        bands = DatasetConfig.all_bands
    elif selected_bands == 'rgb_bands':
        in_channels = len(DatasetConfig.rgb_bands)
        bands = DatasetConfig.rgb_bands
    elif selected_bands == 'rgb_nir_bands':
        in_channels = len(DatasetConfig.rgb_nir_bands)
        bands = DatasetConfig.rgb_nir_bands
    elif selected_bands == 'rgb_swir_bands':
        in_channels = len(DatasetConfig.rgb_swir_bands)
        bands = DatasetConfig.rgb_swir_bands
    elif selected_bands == 'rgb_nir_swir_bands':
        in_channels = len(DatasetConfig.rgb_nir_swir_bands)
        bands = DatasetConfig.rgb_nir_swir_bands
    else:
        print(f"Band combination {selected_bands} is not supported.")
        sys.exit(1)
    
    # Initialize the data module
    data_module = BigEarthNetTIFDataModule()
    data_module.setup(bands)

    # Initialize the model
    if model_name == 'ResNet18':
        model = BigEarthNetResNet18ModelTIF(DatasetConfig.class_weights, DatasetConfig.num_classes, in_channels, weights)
        model.print_summary((3, 120, 120))
        model.visualize_model((3, 120, 120), 'resnet18.png')
    elif model_name == 'custom_model':
        model = CustomModel(DatasetConfig.class_weights, DatasetConfig.num_classes, in_channels, weights)
        model.print_summary((3, 120, 120))
        model.visualize_model((3, 120, 120), 'custom_model.png')
    elif model_name == 'VGG16':
        model = BigEarthNetVGG16ModelTIF(DatasetConfig.class_weights, DatasetConfig.num_classes, in_channels, weights)
        model.print_summary((3, 120, 120))
        model.visualize_model((3, 120, 120), 'vgg16.png')
    else:
        print("Invalid model name. Please try again.")
        return
    
    print(f"Training {model_name} model with {weights} weights and {selected_bands} bands.")

    # Initialize the logger
    log_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\logs'
    logger = TensorBoardLogger(log_dir, name=f"{model_name}_{weights}_{selected_bands}_experiment_5percent")

    checkpoint_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\checkpoints'

    # Checkpoint callback for val_loss
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{weights}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Checkpoint callback for val_acc
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{weights}-{{epoch:02d}}-{{val_acc:.2f}}',
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

    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

if __name__ == "__main__":
    main()