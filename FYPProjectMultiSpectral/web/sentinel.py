# sentinel.py
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, DataCollection
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import random

# Load configuration
config = SHConfig()
config.sh_client_id = "0edf8a3f-2452-4feb-8d85-b33d6b7e23b6"
config.sh_client_secret = "VwSaP8ynk2KzGfcSKgWXa1xk5eyGH2Op"
config.save()

if not config.sh_client_id or not config.sh_client_secret:
    raise ValueError("Sentinel Hub credentials not set!")

# Evalscript for 12 bands
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    output: { bands: 12 }
  };
}
function evaluatePixel(sample) {
  return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
          sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12];
}
"""

def fetch_sentinel_patch(lat, lon, output_tiff=None):
    """
    Fetch a 120x120 Sentinel-2 patch at given coordinates.
    Args:
        lat (float): Latitude
        lon (float): Longitude
        output_tiff (str, optional): Path to save TIFF file
    Returns:
        np.ndarray: Multispectral data (120, 120, 12)
        list: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
    """
    delta = 0.0054  # ~600m radius for 120x120 at 10m resolution
    bbox_coords = [lon - delta, lat - delta, lon + delta, lat + delta]
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

    # Create request with cloud coverage filter
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2020-03-01", "2024-03-31"),
            other_args={"dataFilter": {"maxCloudCoverage": 10}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=[120, 120],
        config=config
    )

    # Fetch data
    data = request.get_data()[0]  # (120, 120, 12)

    if output_tiff:
        # Transpose data to (bands, height, width) for rasterio
        data_tiff = np.transpose(data, (2, 0, 1))  # (12, 120, 120)

        # Define metadata for the TIFF
        meta = {
            'driver': 'GTiff',
            'height': data_tiff.shape[1],
            'width': data_tiff.shape[2],
            'count': data_tiff.shape[0],
            'dtype': data_tiff.dtype,
            'crs': 'EPSG:4326',
            'transform': from_bounds(*bbox_coords, data_tiff.shape[2], data_tiff.shape[1])
        }

        # Save as multispectral TIFF
        with rasterio.open(output_tiff, 'w', **meta) as dst:
            dst.write(data_tiff)
        print(f"Saved patch as {output_tiff}")

    return data, bbox_coords

if __name__ == "__main__":
    min_lat, max_lat = 35.8, 36.0  # Malta
    min_lon, max_lon = 14.4, 14.6
    for i in range(10):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        data, bbox = fetch_sentinel_patch(lat, lon, f"multispectral_patch_{i + 1}.tif")
        print(f"Generated patch {i + 1} with shape {data.shape}")