import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from config import DatasetConfig
from tqdm import tqdm
import numpy as np

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

def normalize_band(band_data, band_mean, band_std):
    normalized_data = ((band_data - band_mean) / band_std * 255).astype(np.uint8)
    return normalized_data

def combineTiffs(base_path, output_path):
    band_paths = generatePaths(base_path)

    # Read the first image to get metadata
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta.copy()
        meta.update(count=len(band_paths), dtype='uint8')  

    # Create a new multi-band TIFF file
    with rasterio.open(output_path, 'w', **meta) as dst:
        for idx, path in enumerate(band_paths, start=1):
            with rasterio.open(path) as src:
                band_data = src.read(1)
                band_name = os.path.basename(path).split('_')[-1].split('.')[0]

                print(f"Processing band: {band_name}")

                band_mean = DatasetConfig.band_stats["mean"].get(band_name, 0)
                band_std = DatasetConfig.band_stats["std"].get(band_name, 1)
                normalized_band = normalize_band(band_data, band_mean, band_std)
                # Write each band's data to the corresponding index in the new file
                dst.write(normalized_band, idx)
            
if __name__ == "__main__":
    path = r'C:\Users\isaac\Desktop\BigEarthTests\OnePBigEarthNetCopySubsets2\50_percent'
    process_folders(path, 'CombinedImages', combineTiffs, exclude_dirs=["CombinedImages"])