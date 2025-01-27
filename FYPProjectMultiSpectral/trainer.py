# Standard library imports
import json
import os
import subprocess
import sys

# Third-party imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from dataloader import BigEarthNetDataLoader
from utils.helper_functions import save_tensorboard_graphs, set_random_seeds, initialize_paths, get_dataset_info, get_model_class 
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

    # Create the main path for the experiment
    main_path = initialize_paths(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)

    # Determine the number of input channels based on the selected bands
    bands_mapping = {
        'all_bands': DatasetConfig.all_bands,
        'rgb_bands': DatasetConfig.rgb_bands,
        'rgb_nir_bands': DatasetConfig.rgb_nir_bands,
        'rgb_swir_bands': DatasetConfig.rgb_swir_bands,
        'rgb_nir_swir_bands': DatasetConfig.rgb_nir_swir_bands
    }
    bands = bands_mapping.get(selected_bands)
    if bands is None:
        print(f"Band combination {selected_bands} is not supported.")
        sys.exit(1)
    in_channels = len(bands)
    
    # Get dataset information
    dataset_dir, metadata_path, metadata_csv = get_dataset_info(selected_dataset)
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    # Initialize the data module
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

    model_class, filename = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class(class_weights, DatasetConfig.num_classes, in_channels, model_weights, main_path)
    model.print_summary((in_channels, 120, 120), filename) 
    model.visualize_model((in_channels, 120, 120), filename)

    print(f"Training {model_name} model with {weights} weights and {selected_bands} bands on the {selected_dataset}.")
    print("Model parameter device:", next(model.parameters()).device)

    # Initialize the logger
    log_dir = os.path.join(main_path, 'logs')
    logger = TensorBoardLogger(log_dir)

    # Initialize callbacks
    checkpoint_dir = os.path.join(main_path, 'checkpoints')
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    checkpoint_callback_acc = ModelCheckpoint( # Checkpoint callback for val_acc
        dirpath=checkpoint_dir,
        filename=f'{{epoch:02d}}-{{val_acc:.2f}}',
        save_top_k=1,
        verbose=False,
        monitor='val_acc',
        mode='max'
    )
    final_checkpoint = ModelCheckpoint( # Checkpoint callback for final model
        dirpath=checkpoint_dir,
        filename=f'final',
        save_last=True
    )
    early_stopping = EarlyStopping( # Early stopping callback
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
                ],
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