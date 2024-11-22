import os
import pandas as pd
import shutil
from tqdm import tqdm
from FYPProjectRGB.preprocessing.config import DatasetConfig

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


if __name__ == "__main__":
    if os.path.exists(DatasetConfig.unwanted_metadata_file):
        removeUnnecessaryFiles(DatasetConfig.dataset_dir, DatasetConfig.unwanted_metadata_file)