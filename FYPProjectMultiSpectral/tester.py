# Standard library imports
import os
import sys
import json

# Third-party imports
import pandas as pd
import torch
import pytorch_lightning as pl

# Local application imports
from config.config import DatasetConfig
from callbacks import BestMetricsCallback
from dataloader import BigEarthNetDataLoader
from utils.data_utils import extract_number
from utils.setup_utils import set_random_seeds
from utils.model_utils import get_model_class
from utils.test_utils import calculate_metrics_and_save_results, visualize_predictions_and_heatmaps, generate_gradcam_visualizations, get_sigmoid_outputs, generate_gradcam_for_single_image
from utils.visualisation_utils import register_hooks, show_rgb_from_batch, clear_activations, visualize_activations
from utils.logging_utils import setup_logger
from models.models import *

# Test the model
def main():
    set_random_seeds()
    torch.set_float32_matmul_precision('high')

    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_dataset = sys.argv[3]
    checkpoint_path = sys.argv[4]
    in_channels = int(sys.argv[5])
    class_weights = json.loads(sys.argv[6])
    metadata_csv = pd.read_csv(sys.argv[7])
    dataset_dir = sys.argv[8]
    bands = json.loads(sys.argv[9]) 
    
    # Create the main path for the experiment
    print(f"Checkpoint Path: {checkpoint_path}")
    main_path = os.path.dirname(os.path.dirname(checkpoint_path))
    print(f"Main path: {main_path}")
    result_path = os.path.join(main_path, "results")
    print(f"Result Path: {result_path}")
    
    dataset_num = extract_number(selected_dataset)
    cache_file = f"{dataset_num}%_sample_weights.npy"
    cache_path = os.path.join(main_path, cache_file)

    # Initialize the log directories
    testing_log_path = os.path.join(main_path, 'logs', 'testing_logs')
    logger = setup_logger(log_dir=testing_log_path, file_name='testing.log')

    # Load the model from the checkpoint
    model_class, _ = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=model_weights, main_path=main_path)
    model.eval()
    register_hooks(model) # Register hooks for visualization of activations

    # Initialize the data module
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage='test')
    class_labels = DatasetConfig.class_labels

    # Initialize the BestMetricsCallback
    metrics_to_track = ['test_acc', 'test_loss', 'test_f1', 'test_f2', 'test_precision', 'test_recall','test_one_error', 'test_hamming_loss', 'test_avg_precision']
    best_test_metrics_path = os.path.join(main_path, 'results', 'best_test_metrics.json')
    best_metrics_callback = BestMetricsCallback(metrics_to_track=metrics_to_track, save_path=best_test_metrics_path)
    
    # Set up Trainer for testing
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='32',
        deterministic=True,
        callbacks=[best_metrics_callback],
        logger = False
    )

    # Run the testing phase
    logger.info("Testing the model...")
    trainer.test(model, datamodule=data_module)
    logger.info("Testing complete.")
    
    # Calculate metrics and save results
    logger.info("Calculating metrics and saving results...")
    all_preds, all_labels = calculate_metrics_and_save_results( # This saves the results as a npz file
        model=model,
        data_module=data_module,
        model_name=model_name,
        dataset_name=selected_dataset,
        class_labels=class_labels,
        result_path=result_path,
        logger = logger
    )
    logger.info("Metrics and results saved.")

    print("Computing continuous probability outputs for ROC AUC...")
    # all_probs = get_sigmoid_outputs(model, dataset_dir, metadata_csv, bands=bands)

    # Visualize predictions and results
    logger.info("Visualizing predictions and heatmaps...")
    visualize_predictions_and_heatmaps(
        model=model,
        data_module=data_module,
        in_channels=in_channels,
        predictions=all_preds,
        true_labels=all_labels,
        class_labels=class_labels,
        model_name=model_name,
        result_path=result_path,
        probs=None,
        logger = logger
    )
    logger.info("Predictions and heatmaps saved.")

    # Visualize activations
    logger.info("Visualizing activations...")
    test_loader = data_module.test_dataloader()
    example_batch = next(iter(test_loader))
    example_imgs, example_lbls = example_batch
    show_rgb_from_batch(example_imgs[0], in_channels)
    example_imgs = example_imgs.to(model.device)
    clear_activations()
    with torch.no_grad():
        _ = model(example_imgs[0].unsqueeze(0))
    visualize_activations(result_path=result_path, num_filters=16)  
    logger.info("Activations saved")
    
    # Generate Grad-CAM visualizations
    logger.info("Generating Grad-CAM visualizations...")
    generate_gradcam_visualizations(
        model=model,
        data_module=data_module,
        class_labels=class_labels,
        model_name=model_name,
        result_path=result_path,
        in_channels=in_channels,
        logger = logger
    )
    logger.info("Grad-CAM visualizations generated.")
    logger.info("Testing completed successfully")

if __name__ == "__main__":
    main()