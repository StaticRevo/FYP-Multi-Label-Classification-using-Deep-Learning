import torch
from torch.utils.data import Dataset
import rasterio
from pathlib import Path
import ast
from config.config import DatasetConfig
from utils.helper_functions import encode_label, get_band_indices

# Dataset class for BigEarthNet dataset
class BigEarthNetDatasetTIF(Dataset):
    def __init__(self, *, df, root_dir, transforms=None, normalisation=None, is_test=False, selected_bands=None, metadata_csv=None):
        super().__init__()
        self.df = df.reset_index(drop=True)  
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.normalisation = normalisation
        self.is_test = is_test

        # Determine which bands to use
        self.selected_bands = selected_bands if selected_bands is not None else DatasetConfig.rgb_bands
        self.selected_band_indices = get_band_indices(self.selected_bands, DatasetConfig.all_bands)

        # Create a mapping from patch_id to labels
        if metadata_csv is not None:
            self.patch_to_labels = dict(zip(metadata_csv['patch_id'], metadata_csv['labels']))
        else:
            raise ValueError("metadata_csv must be provided and contain 'patch_id' and 'labels' columns.")

        # Extract the list of patch_ids for this subset
        self.patch_ids = self.df['patch_id'].tolist()

        # Construct image paths based on patch_ids
        self.image_paths = [self.root_dir / f"{pid}.tif" for pid in self.patch_ids]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the patch_id and corresponding image path
        patch_id = self.patch_ids[idx]
        image_path = self.image_paths[idx]

        # 2. Read the raster data
        try:
            with rasterio.open(image_path) as src:
                image = src.read()  # Shape: (channels, height, width)
                # Select only the desired bands
                image = image[self.selected_band_indices, :, :]
        except Exception as e:
            print(f"Error reading {image_path}: {e}. Returning a zero tensor and zero label.")
            image = torch.zeros((len(self.selected_band_indices), DatasetConfig.image_height, DatasetConfig.image_width), dtype=torch.float32)
            label = torch.zeros(DatasetConfig.num_classes, dtype=torch.float32)
            return image, label

        # 3. Convert image to a float32 tensor
        image = torch.tensor(image, dtype=torch.float32)

        # 4. Apply transformations (e.g., augmentations)
        if self.transforms:
            image = self.transforms(image)

        # 5. Apply normalisation
        if self.normalisation:
            image = self.normalisation(image)

        # 6. Retrieve the label using patch_id
        label = self.get_label(patch_id)

        return image, label

    def get_label(self, patch_id):
        labels = self.patch_to_labels.get(patch_id, None)

        # If no labels found, return a zero vector
        if labels is None:
            print(f"No labels found for patch_id: {patch_id}. Returning zero vector.")
            return torch.zeros(DatasetConfig.num_classes, dtype=torch.float32)

        # If labels are stored as a string, parse them
        if isinstance(labels, str):
            try:
                cleaned_labels = labels.replace(" '", ", '").replace("[", "[").replace("]", "]")
                labels =  ast.literal_eval(cleaned_labels)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing labels for patch_id {patch_id}: {e}")
                return torch.zeros(DatasetConfig.num_classes, dtype=torch.float32)

        # Encode the list of labels into a multi-hot vector
        encoded = encode_label(labels)
        return encoded
