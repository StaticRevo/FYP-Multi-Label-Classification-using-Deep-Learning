import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

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


dataset_dir = r'D:\Datasets\BigEarthNet-S2\100%BigEarthNet'
subset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\5PercentBigEarthNet'
metadata_file = r'C:\Users\isaac\Downloads\metadata.parquet'
metadata_df = pd.read_parquet(metadata_file)
percentage = 5

createSubsets(dataset_dir, subset_dir, metadata_df, percentage)