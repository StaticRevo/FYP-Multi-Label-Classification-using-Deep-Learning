import os
os.environ['CPL_LOG_LEVEL'] = 'NONE'
os.environ['CPL_DEBUG'] = 'OFF'
import sys
import json
import pandas as pd
import torch
import pytorch_lightning as pl
import logging

# Local application imports
from config.config import DatasetConfig, ModelConfig
from callbacks import BestMetricsCallback
from dataloader import BigEarthNetDataLoader
from utils.setup_utils import set_random_seeds
from utils.model_utils import get_model_class
from utils.file_utils import initalize_paths_tester
from utils.test_utils import calculate_metrics_and_save_results, visualize_predictions_and_heatmaps, generate_gradcam_visualizations, get_sigmoid_outputs
from utils.visualisation_utils import register_hooks, show_rgb_from_batch, clear_activations, visualize_activations
from utils.logging_utils import setup_logger
from models.models import *

def main():
    print("DEBUG: tester.py has started", flush=True)
    print("DEBUG: Received sys.argv:", sys.argv, flush=True)
    set_random_seeds()
    torch.set_float32_matmul_precision('high')

    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]
    checkpoint_path = sys.argv[5]
    in_channels = int(sys.argv[6])
    class_weights = json.loads(sys.argv[7])
    metadata_csv = pd.read_csv(sys.argv[8])
    dataset_dir = sys.argv[9]
    bands = json.loads(sys.argv[10])

    print(f"Using checkpoint: {checkpoint_path}")

    # Create the main path for the experiment.
    main_path = initalize_paths_tester(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)
    print(f"Main path: {main_path}")

    model_class, _ = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        class_weights=class_weights,
        num_classes=DatasetConfig.num_classes,
        in_channels=in_channels,
        model_weights=model_weights,
        main_path=main_path
    )
    model.eval()
    register_hooks(model)

    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage='test')

    class_labels = DatasetConfig.class_labels

    # Initialize the BestMetricsCallback.
    metrics_to_track = [
        'test_acc', 'test_loss', 'test_f1', 'test_f2',
        'test_precision', 'test_recall', 'test_one_error',
        'test_hamming_loss', 'test_avg_precision'
    ]
    best_test_metrics_path = os.path.join(main_path, 'results', 'best_test_metrics.json')
    best_metrics_callback = BestMetricsCallback(metrics_to_track=metrics_to_track, save_path=best_test_metrics_path)

    # Set up Trainer for testing.
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='16-mixed',
        deterministic=True,
        callbacks=[best_metrics_callback]
    )
    
    print("Starting testing phase...")
    # Run the testing phase.
    trainer.test(model, datamodule=data_module)
    print("Testing phase complete.")

    result_path = os.path.join(main_path, "results")
    print(f"Results will be saved to: {result_path}")

    # Calculate metrics and save results.
    print("Calculating metrics and saving results...")
    all_preds, all_labels = calculate_metrics_and_save_results(
        model=model,
        data_module=data_module,
        model_name=model_name,
        dataset_name=selected_dataset,
        class_labels=class_labels,
        result_path=result_path
    )

    # Compute continuous probability outputs (using your helper function)
    print("Computing continuous probability outputs for ROC AUC...")
    all_probs = get_sigmoid_outputs(model, dataset_dir, metadata_csv, bands=bands)

    # Visualize predictions and heatmaps.
    print("Visualizing predictions, heatmaps, and ROC AUC curve...")
    visualize_predictions_and_heatmaps(
        model=model,
        data_module=data_module,
        in_channels=in_channels,
        predictions=all_preds,
        true_labels=all_labels,
        class_labels=class_labels,
        model_name=model_name,
        result_path=result_path,
        probs=all_probs
    )

    # Visualize activations
    print("Visualizing activations...")
    test_loader = data_module.test_dataloader()
    example_batch = next(iter(test_loader))
    example_imgs, _ = example_batch
    show_rgb_from_batch(example_imgs[0], in_channels)
    example_imgs = example_imgs.to(model.device)
    clear_activations()
    with torch.no_grad():
        _ = model(example_imgs[0].unsqueeze(0))
    visualize_activations(result_path=result_path, num_filters=16)

    # Generate Grad-CAM visualizations.
    print("Generating Grad-CAM visualizations...")
    generate_gradcam_visualizations(
        model=model,
        data_module=data_module,
        class_labels=class_labels,
        model_name=model_name,
        result_path=result_path,
        in_channels=in_channels
    )
    print("Experiment complete.")

if __name__ == "__main__":
    print("Starting the testing phase...")
    main()