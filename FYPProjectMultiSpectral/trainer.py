# Standard library imports
import json
import os
import subprocess
import sys

# Third-party imports
import torch
import pytorch_lightning as pl
import torch.nn.utils as nn_utils
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from dataloader import BigEarthNetDataLoader
from utils.setup_utils import set_random_seeds
from utils.file_utils import initialize_paths, save_hyperparameters, save_model_architecture
from utils.data_utils import get_dataset_info, extract_number
from utils.model_utils import get_model_class
from utils.visualisation_utils import save_tensorboard_graphs
from utils.logging_utils import setup_logger
from models.models import *
from callbacks import BestMetricsCallback, LogEpochEndCallback

def log_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # Compute L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient Norm before clipping: {total_norm:.4f}")
    return total_norm

class GradientLoggingCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        norm = log_gradient_norms(pl_module)

# Training the model
def main():
    # Set random seeds and enable cuDNN benchmark
    set_random_seeds()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Initalising the variables from the command line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    test_variable = sys.argv[5]

    # Create main path for experiment
    if len(sys.argv) > 6:
        main_path = sys.argv[6]
    else:
        main_path = initialize_paths(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)

    # Initialize the log directories
    log_dir = os.path.join(main_path, 'logs')
    training_log_path = os.path.join(log_dir, 'training_logs')
    tb_logger = TensorBoardLogger(save_dir=log_dir, name='lightning_logs')
    logger = setup_logger(log_dir=training_log_path, file_name='training.log')

    # Determine the number of input channels based on the selected bands
    bands_mapping = {
        'all_bands': DatasetConfig.all_bands,
        'all_imp_bands': DatasetConfig.all_imp_bands,
        'rgb_bands': DatasetConfig.rgb_bands,
        'rgb_nir_bands': DatasetConfig.rgb_nir_bands,
        'rgb_swir_bands': DatasetConfig.rgb_swir_bands,
        'rgb_nir_swir_bands': DatasetConfig.rgb_nir_swir_bands
    }

    # Get the selected bands
    bands = bands_mapping.get(selected_bands)
    if bands is None:
        logger.error(f"Band combination {selected_bands} is not supported.")
        sys.exit(1)
    in_channels = len(bands)
    logger.info(f"Using {in_channels} input channels based on '{selected_bands}'.")
    
    # Get dataset information
    dataset_dir, metadata_path, metadata_csv = get_dataset_info(selected_dataset)
    class_weights = calculate_class_weights(metadata_csv)
    logger.info(f"Class weights: {class_weights}")

    # Initialize the data module
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

    # Get the model class
    model_class, filename = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class(class_weights, DatasetConfig.num_classes, in_channels, model_weights, main_path)
    model.print_summary((in_channels, DatasetConfig.image_height, DatasetConfig.image_width), filename) 
    model.visualize_model((in_channels,  DatasetConfig.image_height, DatasetConfig.image_width), filename)

    logger.info(f"Training {model_name} model with {weights} weights and '{selected_bands}' bands on {selected_dataset}.")
    logger.info(f"Model parameter device: {next(model.parameters()).device}")
    epoch_end_logger_callback = LogEpochEndCallback(logger)

    # Initialize callbacks
    checkpoint_dir = os.path.join(main_path, 'checkpoints')
    checkpoint_callback_loss = ModelCheckpoint( # Checkpoint callback for val_loss
        dirpath=checkpoint_dir,
        filename='best_loss',
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    checkpoint_callback_acc = ModelCheckpoint( # Checkpoint callback for val_acc
        dirpath=checkpoint_dir,
        filename='best_acc',
        save_top_k=1,
        verbose=False,
        monitor='val_acc',
        mode='max'
    )
    final_checkpoint = ModelCheckpoint( # Checkpoint callback for final model
        dirpath=checkpoint_dir,
        filename='final',
        save_last=True
    )
    early_stopping = EarlyStopping( # Early stopping callback
        monitor='val_loss',
        patience=ModelConfig.patience,
        verbose=True,
        mode='min'
    )

    # Initialize the BestMetricsCallback
    metrics_to_track = ['val_acc', 'val_loss', 'val_f1', 'val_f2', 'val_precision', 'val_recall','val_one_error', 'val_hamming_loss', 'val_avg_precision']
    best_metrics_path = os.path.join(main_path, 'results', 'best_metrics.json')
    best_metrics_callback = BestMetricsCallback(metrics_to_track=metrics_to_track, save_path=best_metrics_path)
    
    num_gpus = torch.cuda.device_count()
    # Model Training with custom callbacks
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=ModelConfig.num_epochs,
        logger=tb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=num_gpus if torch.cuda.is_available() else 1,
        precision='16-mixed',
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        callbacks=[
                    checkpoint_callback_loss, 
                    checkpoint_callback_acc, 
                    best_metrics_callback,
                    final_checkpoint, 
                    early_stopping,
                    epoch_end_logger_callback,
                    GradientLoggingCallback()
                ],
    )

    logger.info("Starting model training...")
    trainer.fit(model, data_module)
    logger.info("Model training completed.")

    # Save Tensorboard graphs as images
    logger.info("Saving TensorBoard graphs as images...")
    output_dir = os.path.join(main_path, 'results', 'tensorboard_graphs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_tensorboard_graphs(tb_logger.log_dir, output_dir)
    logger.info(f"TensorBoard graphs saved to {output_dir}.")

    # Start TensorBoard
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Load the best metrics
    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, 'r') as f:
            best_metrics = json.load(f)
    else:
        best_metrics = {}
        logger.warning(f"No best metrics file found at {best_metrics_path}.")

    # Log best metrics
    logger.info("Best Validation Metrics:")
    for metric, value in best_metrics.get('best_metrics', {}).items():
        epoch = best_metrics.get('best_epochs', {}).get(metric, 'N/A')
        logger.info(f"  {metric}: {value} (Epoch {epoch})")

    # Print additional metrics
    logger.info(f"Training Time: {best_metrics.get('training_time_sec', 'N/A'):.2f} seconds")
    logger.info(f"Model Size: {best_metrics.get('model_size_MB', 'N/A'):.2f} MB")
    logger.info(f"Inference Rate: {best_metrics.get('inference_rate_images_per_sec', 'N/A'):.2f} images/second")
    logger.info("Training completed successfully")

    # Save hyperparameters
    file_path = save_hyperparameters(ModelConfig, main_path)
    logger.info(f"Hyperparameters saved to {file_path}")

    arch_path = save_model_architecture(model, (in_channels, 120, 120), file_path, filename=model_name)
    logger.info(f"Model architecture saved to {arch_path}")

    if test_variable == 'True':
        subprocess.run(['python', '../FYPProjectMultiSpectral/tester_runner.py', model_name, weights, selected_bands, selected_dataset])

if __name__ == "__main__":
    main()