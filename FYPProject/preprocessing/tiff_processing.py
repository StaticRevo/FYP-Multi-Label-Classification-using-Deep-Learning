import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from config import DatasetConfig
from tqdm import tqdm


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


if __name__ == "__main__":
    bands_of_interest = ['B01', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']
    for folder in os.listdir(DatasetConfig.dataset_dir):
        folder_path = os.path.join(DatasetConfig.dataset_dir, folder)
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