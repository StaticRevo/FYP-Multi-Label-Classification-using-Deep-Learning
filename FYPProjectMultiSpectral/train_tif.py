import os
import time

from matplotlib import pyplot as plt
import pandas as pd
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from dataloader_tif import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import subprocess
import sys
from utils.helper_functions import save_tensorboard_graphs, extract_number

from models.Models import *

# Training the model
def main():
    torch.set_float32_matmul_precision('high')

    # Initalising the variables from the command line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]

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
    
    dataset_num = extract_number(selected_dataset)
    if dataset_num == 0.5:
        dataset_dir = DatasetConfig.dataset_paths["0.5"]
        metadata_path = DatasetConfig.metadata_paths["0.5"]
        metadata_csv = pd.read_csv(metadata_path)
    elif dataset_num == 1:
        dataset_dir = DatasetConfig.dataset_paths["1"]
        metadata_path = DatasetConfig.metadata_paths["1"]
        metadata_csv = pd.read_csv(metadata_path)
    elif dataset_num == 5:
        dataset_dir = DatasetConfig.dataset_paths["5"]
        metadata_path = DatasetConfig.metadata_paths["5"]
        metadata_csv = pd.read_csv(metadata_path)
    elif dataset_num == 10:
        dataset_dir = DatasetConfig.dataset_paths["10"]
        metadata_path = DatasetConfig.metadata_paths["10"]
        metadata_csv = pd.read_csv(metadata_path)
    elif dataset_num == 50:
        dataset_dir = DatasetConfig.dataset_paths["50"]
        metadata_path = DatasetConfig.metadata_paths["50"]
        metadata_csv = pd.read_csv(metadata_path)
    
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    # Initialize the data module
    data_module = BigEarthNetTIFDataModule(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
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
        model = model_class(class_weights, DatasetConfig.num_classes, in_channels, weights)
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
        filename=f'{model_name}-{weights}-{selected_bands}-{selected_dataset}{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Checkpoint callback for val_acc
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{weights}-{selected_bands}-{selected_dataset}-{{epoch:02d}}-{{val_acc:.2f}}',
        save_top_k=1,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=ModelConfig.patience,
        verbose=True,
        mode='min'
    )

    # Model Training with custom callbacks
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=ModelConfig.num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='16-mixed',
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_acc, early_stopping]
    )

    # Start training while measuring training time
    start_time = time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    training_time = end_time - start_time

    best_acc_checkpoint_path = checkpoint_callback_acc.best_model_path
    best_loss_checkpoint_path = checkpoint_callback_loss.best_model_path

    # Save Tensorboard graphs as images
    output_dir = os.path.join('FYPProjectMultiSpectral/experiments/results', f"{model_name}_{weights}_{selected_bands}_experiment_{selected_dataset}_graphs")
    save_tensorboard_graphs(logger.log_dir, output_dir)

    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Print best metrics
    best_acc = checkpoint_callback_acc.best_model_score
    best_loss = checkpoint_callback_loss.best_model_score
    print(f"Best Validation Accuracy: {best_acc}")
    print(f"Best Validation Loss: {best_loss}")

    # Calculate inference rate
    batch = next(iter(data_module.test_dataloader()))
    x, y = batch
    start_time = time.time()
    model(x)
    end_time = time.time()
    inference_time = end_time - start_time
    inference_rate = len(x) / inference_time
    print(f"Inference Rate: {inference_rate:.2f} images/second")

    # Calculate model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)  # Model size in MB
    print(f"Model Size: {model_size:.2f} MB")

    # Print training time
    training_hours, rem = divmod(training_time, 3600)
    training_minutes, _ = divmod(rem, 60)
    print(f"Training Time: {int(training_hours)} hours {int(training_minutes)} minutes")

    metrics = {
        'Metric': [
            'Best Validation Accuracy', 
            'Best Validation Loss', 
            'Inference Rate (images/sec)', 
            'Model Size (MB)', 
            'Training Time (hours:minutes)'
        ],
        'Value': [
            best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc, 
            best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss,  
            f"{inference_rate:.2f}",
            f"{model_size:.2f}",
            f"{int(training_hours)}:{int(training_minutes)}"
        ]
    }
    df_metrics = pd.DataFrame(metrics)

    # Save DataFrame as an image
    fig, ax = plt.subplots(figsize=(10, 2))  
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
    plt.close()

    # Run test
    subprocess.run([
        'python', 
        'FYPProjectMultiSpectral\\test_tif.py', 
        model_name, 
        weights, 
        selected_bands, 
        selected_dataset, 
        best_acc_checkpoint_path, 
        best_loss_checkpoint_path, 
        str(in_channels),
        class_weights, 
        metadata_csv, 
        dataset_dir, 
        bands
    ])

if __name__ == "__main__":
    main()