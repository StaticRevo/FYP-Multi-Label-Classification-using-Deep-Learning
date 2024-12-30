# Standard library imports
import json
import os
import time
import subprocess
import sys

# Third-party imports
import pandas as pd
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from dataloader import BigEarthNetTIFDataModule
from utils.helper_functions import save_tensorboard_graphs, extract_number, set_random_seeds
from utils.visualisation import *
from models.models import *
from callbacks import BestMetricsCallback

# Training the model
def main():
    set_random_seeds()
    torch.set_float32_matmul_precision('high')

    # Initalising the variables from the command line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    test_variable = sys.argv[5]

    experiment_path = DatasetConfig.experiment_path
    epochs = ModelConfig.num_epochs
    main_path = fr'{experiment_path}\{model_name}_{weights}_{selected_bands}_{selected_dataset}_{epochs}epochs'
    if not os.path.exists(main_path):
        os.makedirs(main_path)

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
        'custom_model': (CustomModel, 'custom_model'),
        'ResNet18': (BigEarthNetResNet18ModelTIF, 'resnet18'),
        'ResNet50': (BigEarthNetResNet50ModelTIF, 'resnet50'),
        'VGG16': (BigEarthNetVGG16ModelTIF, 'vgg16'),
        'VGG19': (BigEarthNetVGG19ModelTIF, 'vgg19'),
        'DenseNet121': (BigEarthNetDenseNet121ModelTIF, 'densenet121'),
        'EfficientNetB0': (BigEarthNetEfficientNetB0ModelTIF, 'efficientnetb0'),
        'EfficientNet_v2': (BigEarthNetEfficientNetV2MModelTIF, 'efficientnet_v2'),
        'Vit-Transformer': (BigEarthNetVitTransformerModelTIF, 'vit_transformer'),
        'Swin-Transformer': (BigEarthNetSwinTransformerModelTIF, 'swin_transformer')
    }

    # Initialize the model
    if model_name in model_mapping:
        model_class, filename = model_mapping[model_name]
        model_weights = None if weights == 'None' else weights
        model = model_class(class_weights, DatasetConfig.num_classes, in_channels, model_weights, main_path)
        model.print_summary((in_channels, 120, 120), filename) 
        model.visualize_model((in_channels, 120, 120), filename)
    else:
        print("Invalid model name. Please try again.")
    
    print()
    print(f"Training {model_name} model with {weights} weights and {selected_bands} bands on the {selected_dataset}.")

    # Initialize the logger
    log_dir = os.path.join(main_path, 'logs')
    logger = TensorBoardLogger(log_dir)

    checkpoint_dir = os.path.join(main_path, 'checkpoints')

    # Checkpoint callback for val_loss
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )

    # Checkpoint callback for val_acc
    checkpoint_callback_acc = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{{epoch:02d}}-{{val_acc:.2f}}',
        save_top_k=1,
        verbose=False,
        monitor='val_acc',
        mode='max'
    )

    # Checkpoint callback for final model
    final_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'final',
        save_last=True
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=ModelConfig.patience,
        verbose=True,
        mode='min'
    )

    # Initialize the BestMetricsCallback
    metrics_to_track = ['val_acc', 'val_loss', 'val_f1', 'val_precision', 'val_recall', 'val_subset_accuracy', 'val_hamming_loss']
    best_metrics_path = os.path.join(main_path, 'results', 'best_metrics.json')
    best_metrics_callback = BestMetricsCallback(metrics_to_track=metrics_to_track, save_path=best_metrics_path)
    
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
        callbacks=[
                    checkpoint_callback_loss, 
                    checkpoint_callback_acc, 
                    best_metrics_callback,
                    final_checkpoint, 
                    early_stopping
                ]
    )

    trainer.fit(model, data_module)

    # Retrieve best checkpoint paths
    best_acc_checkpoint_path = checkpoint_callback_acc.best_model_path
    best_loss_checkpoint_path = checkpoint_callback_loss.best_model_path
    last_checkpoint_path = final_checkpoint.best_model_path

    # Save Tensorboard graphs as images
    output_dir = os.path.join(main_path, 'results', 'tensorboard_graphs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_tensorboard_graphs(logger.log_dir, output_dir)

    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Load the best metrics
    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, 'r') as f:
            best_metrics = json.load(f)
    else:
        best_metrics = {}
        print(f"No best metrics file found at {best_metrics_path}.")

    # Print best metrics
    print("\nBest Validation Metrics:")
    for metric, value in best_metrics.get('best_metrics', {}).items():
        epoch = best_metrics.get('best_epochs', {}).get(metric, 'N/A')
        print(f"  {metric}: {value} (Epoch {epoch})")

    # Print additional metrics
    print(f"  Training Time: {best_metrics.get('training_time_sec', 'N/A'):.2f} seconds")
    print(f"  Model Size: {best_metrics.get('model_size_MB', 'N/A'):.2f} MB")
    print(f"  Inference Rate: {best_metrics.get('inference_rate_images_per_sec', 'N/A'):.2f} images/second")

    if test_variable == 'True':
        # Run test
        args = [
            'python', 
            'FYPProjectMultiSpectral\\tester_runner.py', 
            model_name, 
            weights, 
            selected_bands, 
            selected_dataset, 
            best_acc_checkpoint_path, 
            best_loss_checkpoint_path, 
            last_checkpoint_path,
            str(in_channels),
            json.dumps(class_weights.tolist()),
            metadata_path, 
            dataset_dir, 
            json.dumps(bands)
        ]

        # Print the arguments
        print("Arguments to subprocess.run:")
        for arg in args:
            print(arg)

        # Run the subprocess
        subprocess.run(args)

if __name__ == "__main__":
    main()