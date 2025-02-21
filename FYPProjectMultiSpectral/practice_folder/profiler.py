import os
import shutil
import pandas as pd
from tqdm import tqdm

def removeUnnecessaryDataItems(dataset_dir, unwanted_metadata_file):
    deleted_items = 0

    # Read the metadata from the Parquet file.
    metadata_df = pd.read_parquet(unwanted_metadata_file)

    # Iterate over each row in the DataFrame with a progress bar.
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc='Removing unnecessary data items'):
        patch_id = row['patch_id']
        file_name = f"{patch_id}.tif"
        item_path = os.path.join(dataset_dir, file_name)

        # If the file exists, delete it
        if os.path.exists(item_path):
            os.remove(item_path)
            deleted_items += 1

    print(f"Deleted {deleted_items} data items")


# Set the maximum column width to None to display full strings
pd.set_option('display.max_colwidth', None)

unwanted_file = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
df = pd.read_parquet(unwanted_file) # Read the Parquet file into a DataFrame
print(df.head(10)) # Print the first 10 rows

dataset_dir = r'D:\100%_BigEarthNet'
removeUnnecessaryDataItems(dataset_dir, unwanted_file)
