# Standard library imports
import os
import sys
import json

# Third-party imports
import pandas as pd
import torch
import pytorch_lightning as pl

# Local application imports
from config.config import DatasetConfig, ModelConfig
from callbacks import BestMetricsCallback
from dataloader import BigEarthNetDataLoader
from utils.data_utils import extract_number
from utils.setup_utils import set_random_seeds
from utils.model_utils import get_model_class
from utils.test_utils import calculate_metrics_and_save_results, visualize_predictions_and_heatmaps, generate_gradcam_visualizations, get_sigmoid_outputs
from utils.visualisation_utils import register_hooks, show_rgb_from_batch, clear_activations, visualize_activations
from utils.logging_utils import setup_logger
from models.models import *

# Testing the model
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

    model_class, _ = get_model_class(model_name)
    model_weights = None if weights == 'None' else weights
    model = model_class.load_from_checkpoint(checkpoint_path, class_weights=class_weights, num_classes=DatasetConfig.num_classes, in_channels=in_channels, model_weights=model_weights, main_path=main_path)
    model.eval()
    register_hooks(model)

    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage='test')

    class_labels = DatasetConfig.class_labels


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