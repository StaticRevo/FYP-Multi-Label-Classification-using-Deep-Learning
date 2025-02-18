# Standard library imports
import os
import random
import shutil
import ast

# Third-party imports
import requests
import zstandard as zstd
from tqdm import tqdm
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from utils.label_utils import encode_label
from config.config_utils import clean_and_parse_labels

# Download and extract the BigEarthNet dataset
def downloadAndExtractDataset(dataset_dir):
    download_url = 'https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1'
    chunk_size = 1024
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status() 
        total = int(response.headers.get('content-length', 0))
        block_size = chunk_size

        with open(dataset_dir, 'wb') as file, tqdm(total=total, unit='iB', unit_scale=True, desc='Downloading BigEarthNet', unit_divisor=1024) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print('The dataset has been downloaded successfully')
    except requests.exceptions.RequestException as e:
        print(f"Error during dataset download: {e}")
        raise
    
    try:
        with open(dataset_dir, 'rb') as file_in:
            dctx = zstd.ZstdDecompressor()
            with open(dataset_dir[:-4] + '.tar', 'wb') as file_out:
                dctx.copy_stream(file_in, file_out)
        print(f"Decompressed dataset to: {dataset_dir[:-4] + '.tar'}")
    except Exception as e:
        print(f"Error during dataset decompression: {e}")
        raise

# Remove unnecessary files from the dataset
def removeUnnecessaryFiles(dataset_dir, unwanted_metadata_file):
    deleted_folders = 0

    metadata_df = pd.read_parquet(unwanted_metadata_file)

    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc='Removing unnecessary files'):
        patch_id = row['patch_id']
        folder_name = '_'.join(patch_id.split('_')[:2])
        dest_dir = os.path.join(dataset_dir, folder_name, patch_id)

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
            deleted_folders += 1
    
    print(f"Deleted {deleted_folders} folders")

# Create subsets of the dataset based on a percentage
def createSubsets(dataset_dir, subset_dir, metadata_df, percentage):
    metadata_subset = pd.DataFrame(columns=metadata_df.columns)

    subset_path = os.path.join(subset_dir, f'{percentage}_percent')

    if not os.path.exists(subset_path):
        os.makedirs(subset_path)
        print(f"Created subset: {subset_path}")
    else:
        print(f"Subset already exists: {subset_path}")
    
    for folder in tqdm(os.listdir(dataset_dir), desc='Creating subsets'):
        folder_path = os.path.join(dataset_dir, folder)

        if os.path.isdir(folder_path) and folder != os.path.basename(subset_dir):
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

            num_subfolders = len(subfolders)
            num_subset = min(max(1, int(num_subfolders * (percentage / 100))), num_subfolders)

            selected_subfolders = random.sample(subfolders, num_subset)

            metadata_subset = pd.concat([metadata_subset, metadata_df[metadata_df['patch_id'].isin(selected_subfolders)]])

            for selected in selected_subfolders:
                dest_path = os.path.join(subset_path, folder, selected)
                if not os.path.exists(dest_path): 
                    try:
                        shutil.copytree(os.path.join(folder_path, selected), dest_path)
                    except FileExistsError:
                        print(f"Directory already exists, skipping: {dest_path}")

    # Save metadata for the subset
    metadata_subset.to_csv(os.path.join(subset_dir, f'metadata_{percentage}_percent.csv'), index=False)

# Copy a subset of images from the original dataset to a new directory
def copy_subset_images(original_images_dir, original_metadata_path, subset_metadata_path, subset_images_dir):
    original_images_dir = Path(original_images_dir)
    subset_metadata_path = Path(subset_metadata_path)
    subset_images_dir = Path(subset_images_dir)
    subset_images_dir.mkdir(parents=True, exist_ok=True)

    # Load subset metadata
    subset_metadata = pd.read_csv(subset_metadata_path)

    # Get unique patch_ids from subset metadata
    subset_patch_ids = subset_metadata["patch_id"].unique().tolist()

    missing_files = []

    for patch_id in subset_patch_ids:
        # Construct file paths
        src_path = original_images_dir / f"{patch_id}.tif"  
        dst_path = subset_images_dir / f"{patch_id}.tif"
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            missing_files.append(patch_id)

    if missing_files: # Report missing files (if any)
        print(f"Warning: {len(missing_files)} files missing in original dataset")
        print("Missing patch_ids:", missing_files)

# Count the number of .tif images in a folder
def count_tif_images(folder_path):
    total_images = 0  # Initialize total image counter

    # Iterate through all files in the folder
    for file in tqdm(os.listdir(folder_path), desc="Counting .tif images"):
        file_path = os.path.join(folder_path, file)
        
        # Check if the current path is a file and has a .tif extension
        if os.path.isfile(file_path) and file.lower().endswith('.tif'):
            total_images += 1

    return total_images

# Display the percentage of a partial count compared to a full count
def display_percentage(partial_count, full_count, folder_name):
    percentage = (partial_count / full_count) * 100
    print(f"Folder: {folder_name} | Subfolder Count: {partial_count} | Percentage: {percentage:.2f}%")

# Resize a .tif image to a new width and height
def resizeTiffFiles(input_tiff, output_tiff, new_width, new_height):
    with rasterio.open(input_tiff) as src:
        transform, width, height = calculate_default_transform(src.crs, src.crs, new_width, new_height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': src.crs,
            'transform': transform,
            'width': new_width,
            'height': new_height
        })
        with rasterio.open(output_tiff, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )

# Process folders in a dataset directory
def process_folders(dataset_dir, combined_dir_name, combine_function, exclude_dirs=[]):
    combined_destination_dir = os.path.join(dataset_dir, combined_dir_name)

    if not os.path.exists(combined_destination_dir):
        os.makedirs(combined_destination_dir)

    folders = [folder for folder in os.listdir(dataset_dir) if folder not in exclude_dirs]

    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(dataset_dir, folder)
        dest_folder_path = os.path.join(combined_destination_dir, folder)
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

        subfolders = os.listdir(folder_path)
        for subfolder in tqdm(subfolders, desc=f"Processing subfolders in {folder}", leave=False):
            subfolder_path = os.path.join(folder_path, subfolder, subfolder)
            dest_subfolder_path = os.path.join(dest_folder_path, f"{subfolder}.tif")
            combine_function(subfolder_path, dest_subfolder_path)

# Generate paths for each band in a multi-spectral image
def generatePaths(base_path):
    bands = ['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09','B11', 'B12']
    return [f"{base_path}_{band}.tif" for band in bands]

# Combine multiple .tif files into a single multi-band .tif file
def combineTiffs(base_path, output_path):
    band_paths = generatePaths(base_path)
    # Read the first image to get metadata
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta.copy()
        meta.update(count=len(band_paths))  # Update the count to the number of bands
    # Create a new multi-band TIFF file
    with rasterio.open(output_path, 'w', **meta) as dst:
        for idx, path in enumerate(band_paths, start=1):
            with rasterio.open(path) as src:
                # Read each band and write it to the new file
                dst.write(src.read(1), idx)

# Move all images from subfolders to a single folder
def move_images_to_single_folder(source_root_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Count the total number of image files
    total_files = sum(len(files) for _, _, files in os.walk(source_root_dir) if any(file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) for file in files))
    
    with tqdm(total=total_files, desc="Moving images", unit="file") as pbar:
        for subdir, _, files in os.walk(source_root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    source_file_path = os.path.join(subdir, file)
                    target_file_path = os.path.join(target_dir, file)
                    
                    # Ensure unique filenames in the target directory
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file_path):
                        target_file_path = os.path.join(target_dir, f"{base}_{counter}{extension}")
                        counter += 1
                    
                    shutil.move(source_file_path, target_file_path)
                    pbar.update(1)

    # Remove empty directories
    for subdir, dirs, _ in os.walk(source_root_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(subdir, dir)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass 

# Move all subfolders to the main directory
def move_all_subfolders_to_main_directory(source_root_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create destination directory if it doesn't exist
    
    for root_folder in os.listdir(source_root_dir):
        root_folder_path = os.path.join(source_root_dir, root_folder)
        
        if os.path.isdir(root_folder_path):  # Ensure it's a directory
            for subfolder in os.listdir(root_folder_path):
                subfolder_path = os.path.join(root_folder_path, subfolder)
                
                if os.path.isdir(subfolder_path):  # Ensure it's a directory
                    # Define the new destination path for the subfolder
                    dest_path = os.path.join(target_dir, subfolder)
                    
                    # Move the folder
                    shutil.move(subfolder_path, dest_path)
                    print(f"Moved: {subfolder_path} -> {dest_path}")

# Move images based on a split column in a CSV file
def move_images_based_on_split(csv_file_path, source_root_dir, target_root_dir):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    splits = ['train', 'test', 'validation']
    for split in splits:
        split_dir = os.path.join(target_root_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    metadata_csv = pd.read_csv(csv_file_path)
    print(metadata_csv.columns)  # Print the columns to debug
    total_files = len(metadata_csv)
    
    with tqdm(total=total_files, desc="Moving images based on split", unit="file") as pbar:
        for _, row in metadata_csv.iterrows():
            patch_id = row['patch_id']
            split = row['split']
            if split in splits:
                source_file_path = os.path.join(source_root_dir, f"{patch_id}.jpg")
                target_file_path = os.path.join(target_root_dir, split, f"{patch_id}.jpg")
                
                if os.path.exists(source_file_path):
                    shutil.move(source_file_path, target_file_path)
                    pbar.update(1)

# Precompute sample weights for a dataset based on label frequencies
def precompute_sample_weights(metadata_csv, num_classes, label_column="labels", cache_path="subset_sample_weights.npy"):
    # Check if cached weights exist; if so, load and print a message.
    if os.path.exists(cache_path):
        cached_weights = np.load(cache_path)
        return cached_weights

    # Clean and parse the labels for each row.
    metadata_csv.loc[:, label_column] = metadata_csv[label_column].apply(clean_and_parse_labels)
    print(metadata_csv[label_column].head())

    # Initialize an array to count the occurrences of each class.
    label_counts = np.zeros(num_classes, dtype=np.float32)

    # Count occurrences of each class across all samples.
    for idx, row in metadata_csv.iterrows():
        label_list = row[label_column]
        encoded_labels = encode_label(label_list)  # Convert label names to multi-hot encoding
        label_counts += np.array(encoded_labels, dtype=np.float32)  # Count occurrences of each class

    total_rows = len(metadata_csv)
    weights = []

    # Compute a weight for each sample based on the inverse frequency of its labels.
    for idx, row in metadata_csv.iterrows():
        label_list = row[label_column]
        encoded_labels = encode_label(label_list)  # Convert to multi-hot encoding

        pos_indices = np.where(np.array(encoded_labels, dtype=np.float32) == 1)[0]
        if len(pos_indices) == 0:
            weight = 1.0  
        else:
            inv_freqs = [1.0 / label_counts[i] for i in pos_indices]
            weight = np.mean(inv_freqs)

        weights.append(weight)


    sample_weights = np.array(weights, dtype=np.float32)
    np.save(cache_path, sample_weights)

    return sample_weights

# Create a weighted sampler for a dataset based on precomputed sample weights
def create_weighted_sampler_from_csv(metadata_csv, num_classes, label_column="labels", cache_path="subset_sample_weights.npy"):
    sample_weights = precompute_sample_weights(metadata_csv, num_classes, label_column, cache_path)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


