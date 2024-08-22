import os
import shutil
import pandas as pd
from tqdm import tqdm

# Load the meta data
metadata_df = pd.read_parquet(r'C:\Users\isaac\datasets\2020 - BigEarthNet-S2\metadata.parquet')

# Base directories
source_base_dir = r'C:\Users\isaac\Desktop\SampleBigEarth'
destination_base_dir = r'C:\Users\isaac\Desktop\SampleBigEarth\BigEarthNetDataset'

# Function to process each image
def process_image(image_folder_path, labels):
    try:
        # Process each label
        for label in labels:
            print(label)
            safe_label = label
            print(safe_label)
            dest_dir = os.path.join(destination_base_dir, safe_label)

            # Create the directory if it does not exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)

            # Construct the destination path for each label
            dest_folder_path = os.path.join(dest_dir, os.path.basename(image_folder_path))

            # Check if the destination folder already exists
            if not os.path.exists(dest_folder_path):
                # Copy the folder
                shutil.copytree(image_folder_path, dest_folder_path)
                print(f"Copied {image_folder_path} to {dest_folder_path}")
            else:
                print(f"Folder already exists: {dest_folder_path}")

        # Remove the source directory after copying all the labels
        shutil.rmtree(image_folder_path)
        print(f"Removed {image_folder_path}")
    except Exception as e:
        print(f"Error processing folder {image_folder_path}: {e}")

# Iterate through each date folder
for date_folder in tqdm(os.listdir(source_base_dir), desc="Processing Date Folders"):
    date_folder_path = os.path.join(source_base_dir, date_folder)
    
    if os.path.isdir(date_folder_path):
        # Iterate through each image folder within the date folder
        for image_folder in os.listdir(date_folder_path):
            image_folder_path = os.path.join(date_folder_path, image_folder)
            
            if os.path.isdir(image_folder_path):
                # Find the corresponding metadata row
                patch_id = image_folder
                row = metadata_df[metadata_df['patch_id'] == patch_id]
                
                if not row.empty:
                    labels = row.iloc[0]['labels']
                    process_image(image_folder_path, labels)
                else:
                    print(f"No data found for {patch_id}")

print("Processing complete.")