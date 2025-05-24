# Standard library imports
import random

# Third-party imports
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, DataCollection
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Load configuration
config = SHConfig()
config.sh_client_id = "0edf8a3f-2452-4feb-8d85-b33d6b7e23b6"
config.sh_client_secret = "VwSaP8ynk2KzGfcSKgWXa1xk5eyGH2Op"
config.save()

if not config.sh_client_id or not config.sh_client_secret:
    raise ValueError("Sentinel Hub credentials not set!")

# Evalscript for 12 bands with UINT16 output and proper scaling
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    output: { bands: 12, sampleType: "UINT16" }  // Request UINT16 output
  };
}
function evaluatePixel(sample) {
  // Scale reflectance to 0-10000 (Sentinel-2 L2A standard)
  return [sample.B01 * 10000, sample.B02 * 10000, sample.B03 * 10000, sample.B04 * 10000,
          sample.B05 * 10000, sample.B06 * 10000, sample.B07 * 10000, sample.B08 * 10000,
          sample.B8A * 10000, sample.B09 * 10000, sample.B11 * 10000, sample.B12 * 10000];
}
"""

def fetch_sentinel_patch(lat, lon, output_tiff=None):
    """
    Fetch a 120x120 Sentinel-2 patch at given coordinates in WGS84, matching BigEarthNet’s size and scaling.
    Args:
        lat (float): Latitude
        lon (float): Longitude
        output_tiff (str, optional): Path to save TIFF file
    Returns:
        np.ndarray: Multispectral data (120, 120, 12) in uint16
        list: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
    """
    delta = 0.0054  # 600m radius for 120x120 at 10m resolution in WGS84
    bbox_coords = [lon - delta, lat - delta, lon + delta, lat + delta]
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

    # Create request with cloud coverage filter
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2018-06-01", "2024-08-31"),
            other_args={"dataFilter": {"maxCloudCoverage": 1}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=[120, 120], 
        config=config
    )

    # Fetch data
    data = request.get_data()[0] # (120, 120, 12) in uint16, scaled to 0-10000

    if output_tiff:
        # Transpose data to (bands, height, width) for rasterio
        data_tiff = np.transpose(data, (2, 0, 1)).astype(np.uint16)  # (12, 120, 120)

        # Define metadata in WGS84 (no reprojection)
        meta = {
            'driver': 'GTiff',
            'height': 120, 
            'width': 120,  
            'count': data_tiff.shape[0],
            'dtype': 'uint16',
            'crs': 'EPSG:4326',
            'transform': from_bounds(*bbox_coords, 120, 120)  
        }

        # Save directly as multispectral TIFF in WGS84
        with rasterio.open(output_tiff, 'w', **meta) as dst:
            dst.write(data_tiff)
        print(f"Saved patch as {output_tiff} in EPSG:4326")

    return data, bbox_coords

if __name__ == "__main__":
    min_lat, max_lat = 35.8, 36.0  # Malta
    min_lon, max_lon = 14.4, 14.6
    for i in range(10):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        data, bbox = fetch_sentinel_patch(lat, lon, f"multispectral_patch_{i + 1}.tif")
        print(f"Generated patch {i + 1} with shape {data.shape}")