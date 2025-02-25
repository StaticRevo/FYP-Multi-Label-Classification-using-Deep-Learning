from dataloader import BigEarthNetDataLoader
from utils.setup_utils import set_random_seeds
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
import pandas as pd

def main():
    # Optionally set random seeds for reproducibility
    set_random_seeds(42)

    # Get dataset directory and metadata CSV file
    dataset_dir = DatasetConfig.dataset_paths['0.5']
    metadata_path = DatasetConfig.metadata_paths['0.5']
    metadata_csv = pd.read_csv(metadata_path)
    bands = DatasetConfig.all_bands

    # Initialize the data module
    data_module = BigEarthNetDataLoader(bands=bands, dataset_dir=dataset_dir, metadata_csv=metadata_csv)
    data_module.setup(stage=None)

    # Get the train dataloader
    train_loader = data_module.train_dataloader()

    # Iterate over one batch to verify normalization
    for batch in train_loader:
        images, labels = batch  # assumes your dataset returns (image, label)
        # Print shape to ensure it's as expected: (batch_size, channels, height, width)
        print("Batch image shape:", images.shape)

        # Compute the mean and std for each channel over the batch
        normalized_means = images.mean(dim=[0, 2, 3])
        normalized_stds = images.std(dim=[0, 2, 3])

        print("Normalized channel means:", normalized_means)
        print("Normalized channel stds:", normalized_stds)
    
        break

if __name__ == '__main__':
    main()
