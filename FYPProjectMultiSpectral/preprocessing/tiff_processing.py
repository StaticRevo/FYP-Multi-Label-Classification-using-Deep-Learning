import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
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
            
if __name__ == "__main__":
    path = r'C:\Users\isaac\Desktop\BigEarthTests\5PercentBigEarthNetSubset\5_percent'
    bands_of_interest = ['B01', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for subfolder in tqdm(os.listdir(folder_path), desc='Resizing TIFF files'):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for band in bands_of_interest:
                        band_source = subfolder_path + "/" + subfolder + "_" + band + ".tif"
                        temp_tif = subfolder_path + "/" + subfolder + "_" + band + "_resized.tif"
                        new_width = 120
                        new_height = 120

                        resizeTiffFiles(band_source, temp_tif, new_width, new_height)

                        os.remove(band_source)  # Delete the original
                        os.rename(temp_tif, band_source)  # Rename the temporary file

    process_folders(path, 'CombinedImages', combineTiffs, exclude_dirs=["CombinedImages"])