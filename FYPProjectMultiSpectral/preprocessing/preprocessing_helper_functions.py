import requests
import zstandard as zstd
from tqdm import tqdm
import os
import shutil
import pandas as pd
import random
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio

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


def count_subfolders(base_dir, folder):
    # Dictionary to hold folder counts
    folder_counts = {}
    total_subfolders = 0  # Initialize total subfolder counter
    
    # Iterate through all folders in the base directory
    for folder in tqdm(os.listdir(base_dir), desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder)
        
        # Check if the current path is a directory
        if os.path.isdir(folder_path):
            # Count subdirectories within this folder
            subfolder_count = sum(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in os.listdir(folder_path))
            folder_counts[folder] = subfolder_count
        
            # Update total subfolder count
            total_subfolders += subfolder_count

    # Print total subfolders
    return total_subfolders, folder

def display_percentage(partial_count, full_count, folder_name):
    percentage = (partial_count / full_count) * 100
    print(f"Folder: {folder_name} | Subfolder Count: {partial_count} | Percentage: {percentage:.2f}%")

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

def generatePaths(base_path):
    bands = ['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09','B11', 'B12']
    return [f"{base_path}_{band}.tif" for band in bands]

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
