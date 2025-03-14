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
from utils.setup_utils import set_random_seeds
from utils.file_utils import initialize_paths, save_hyperparameters, save_model_architecture
from utils.data_utils import get_dataset_info, extract_number
from utils.model_utils import get_model_class
from utils.visualisation_utils import save_tensorboard_graphs
from utils.logging_utils import setup_logger
from models.models import *
from callbacks import BestMetricsCallback, LogEpochEndCallback, GradientLoggingCallback, OnChangeLrLoggerCallback, EarlyStoppingLoggerCallback

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

    # Create main path for experiment
    if len(sys.argv) > 6:
        main_path = sys.argv[6]
    else:
        main_path = initialize_paths(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)

    # Initialize the log directories
    log_dir = os.path.join(main_path, 'logs')
    training_log_path = os.path.join(log_dir, 'training_logs')
    tb_logger = TensorBoardLogger(save_dir=log_dir, name='lightning_logs', version='version_0')
    logger = setup_logger(log_dir=training_log_path, file_name='training.log')

    resume_checkpoint = None
    resumed_epoch = 0
    if len(sys.argv) > 7 and os.path.exists(sys.argv[7]):
        resume_checkpoint = sys.argv[7]
        checkpoint = torch.load(resume_checkpoint, map_location=lambda storage, loc: storage)
        resumed_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming training from epoch {resumed_epoch}...")

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

    # Initialize the data module
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

    # Get the model class
    model_class, filename = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class(class_weights, DatasetConfig.num_classes, in_channels, model_weights, main_path)
    model.print_summary((in_channels, 120, 120), filename) 
    model.visualize_model((in_channels, 120, 120), filename)

    logger.info(f"Training {model_name} model with {weights} weights and '{selected_bands}' bands on {selected_dataset}.")
    epoch_end_logger_callback = LogEpochEndCallback(logger)

    # Save hyperparameters
    file_path = save_hyperparameters(ModelConfig, main_path)
    logger.info(f"Hyperparameters saved to {file_path}")

    # Save model architecture
    arch_path = save_model_architecture(model, (in_channels, 120, 120), file_path, filename=model_name)
    logger.info(f"Model architecture saved to {arch_path}")

    # Initialize callbacks
    checkpoint_dir = os.path.join(main_path, 'checkpoints')
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Model Training with custom callbacks
    trainer = pl.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=ModelConfig.num_epochs,
        logger=tb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='32',
        gradient_clip_val=100.0,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        callbacks=[
                    best_metrics_callback,
                    final_checkpoint, 
                    early_stopping,
                    epoch_end_logger_callback,
                    #GradientLoggingCallback(),
                    OnChangeLrLoggerCallback(logger),
                    EarlyStoppingLoggerCallback(logger)
                ],
    )

    if resume_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        logger.info("No checkpoint provided, starting training from scratch.")

    logger.info("Starting model training...")
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
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

    # logger.info("Shutting down the computer in 10 seconds")
    # import time
    # time.sleep(10)

    # os.system("shutdown /s /t 0")
    
    if test_variable == 'True':
        subprocess.run(['python', '../FYPProjectMultiSpectral/tester_runner.py', model_name, weights, selected_bands, selected_dataset])

if __name__ == "__main__":
    main()