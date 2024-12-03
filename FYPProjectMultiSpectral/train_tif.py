import json
from config.config import DatasetConfig, ModelConfig
from dataloader_tif import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import sys

#from model_tif import BigEarthNetResNet18ModelTIF
from models.CustomModel import CustomModel
from models.ResNet18 import BigEarthNetResNet18ModelTIF
from models.ResNet50 import BigEarthNetResNet50ModelTIF
from models.VGG16 import BigEarthNetVGG16ModelTIF
from models.VGG19 import BigEarthNetVGG19ModelTIF
from models.DenseNet121 import BigEarthNetDenseNet121ModelTIF
from models.EfficientNetB0 import BigEarthNetEfficientNetB0ModelTIF
from models.VisionTransformer import BigEarthNetVitTransformerModelTIF
from models.SwinTransformer import BigEarthNetSwinTransformerModelTIF

# Training the model
def main():
    torch.set_float32_matmul_precision('high')

    # Initalising the variables from the command line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]

    print(f"Selected bands from command line: {selected_bands}")

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
    data_module = BigEarthNetTIFDataModule(bands=bands)
    data_module.setup(stage=None)

    model_mapping = {
        'custom_model': (CustomModel, 'custom_model.png'),
        'ResNet18': (BigEarthNetResNet18ModelTIF, 'resnet18.png'),
        'ResNet50': (BigEarthNetResNet50ModelTIF, 'resnet50.png'),
        'VGG16': (BigEarthNetVGG16ModelTIF, 'vgg16.png'),
        'VGG19': (BigEarthNetVGG19ModelTIF, 'vgg19.png'),
        'DenseNet121': (BigEarthNetDenseNet121ModelTIF, 'densenet121.png'),
        'EfficientNetB0': (BigEarthNetEfficientNetB0ModelTIF, 'efficientnetb0.png'),
        'Vit-Transformer': (BigEarthNetVitTransformerModelTIF, 'vit_transformer.png'),
        'Swin-Transformer': (BigEarthNetSwinTransformerModelTIF, 'swin_transformer.png')
    }

    # Initialize the model
    if model_name in model_mapping:
        model_class, filename = model_mapping[model_name]
        model = model_class(DatasetConfig.class_weights, DatasetConfig.num_classes, in_channels, weights)
        model.print_summary((in_channels, 120, 120)) 
        model.visualize_model((in_channels, 120, 120), filename)
    else:
        print("Invalid model name. Please try again.")
    
    print()
    print(f"Training {model_name} model with {weights} weights and {selected_bands} bands on the {selected_dataset}.")

    # Initialize the logger
    log_dir = r'C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\experiments\logs'
    logger = TensorBoardLogger(log_dir, name=f"{model_name}_{weights}_{selected_bands}_experiment_{selected_dataset}")

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

    best_acc_checkpoint_path = checkpoint_callback_acc.best_model_path
    best_loss_checkpoint_path = checkpoint_callback_loss.best_model_path


    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Run test
    subprocess.run(['python', 'FYPProjectMultiSpectral\\test_tif.py', model_name, weights, selected_bands, selected_dataset, best_acc_checkpoint_path, best_loss_checkpoint_path,  str(in_channels)])

if __name__ == "__main__":
    main()