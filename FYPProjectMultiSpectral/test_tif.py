import os
import subprocess
import pandas as pd

metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\metadata.parquet'
unwanted_metadata_path: str = r'C:\Users\isaac\Desktop\BigEarthTests\metadata_for_patches_with_snow_cloud_or_shadow (1).parquet'
metadata_csv = pd.read_parquet(metadata_path)
unwanted_metadata = pd.read_parquet(unwanted_metadata_path)

# Print the number of records
print(f"Number of records: {metadata_csv.shape[0]}")
print(f"Number of unwanted records: {unwanted_metadata.shape[0]}")
print(f"Total number of records: {metadata_csv.shape[0] + unwanted_metadata.shape[0]}")
