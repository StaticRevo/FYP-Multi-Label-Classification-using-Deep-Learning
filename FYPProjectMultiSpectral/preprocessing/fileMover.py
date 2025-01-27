# Standard library imports
import os
import shutil

# Third-party imports
from tqdm import tqdm
import pandas as pd

def move_images_to_single_folder(source_root_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Count the total number of image files
    total_files = sum(len(files) for _, _, files in os.walk(source_root_dir) if any(file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) for file in files))
    
    with tqdm(total=total_files, desc="Moving images", unit="file") as pbar:
        for subdir, _, files in os.walk(source_root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    source_file_path = os.path.join(subdir, file)
                    target_file_path = os.path.join(target_dir, file)
                    
                    # Ensure unique filenames in the target directory
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_file_path):
                        target_file_path = os.path.join(target_dir, f"{base}_{counter}{extension}")
                        counter += 1
                    
                    shutil.move(source_file_path, target_file_path)
                    pbar.update(1)

    # Remove empty directories
    for subdir, dirs, _ in os.walk(source_root_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(subdir, dir)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass 

def move_all_subfolders_to_main_directory(source_root_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create destination directory if it doesn't exist
    
    for root_folder in os.listdir(source_root_dir):
        root_folder_path = os.path.join(source_root_dir, root_folder)
        
        if os.path.isdir(root_folder_path):  # Ensure it's a directory
            for subfolder in os.listdir(root_folder_path):
                subfolder_path = os.path.join(root_folder_path, subfolder)
                
                if os.path.isdir(subfolder_path):  # Ensure it's a directory
                    # Define the new destination path for the subfolder
                    dest_path = os.path.join(target_dir, subfolder)
                    
                    # Move the folder
                    shutil.move(subfolder_path, dest_path)
                    print(f"Moved: {subfolder_path} -> {dest_path}")

def move_images_based_on_split(csv_file_path, source_root_dir, target_root_dir):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
    
    splits = ['train', 'test', 'validation']
    for split in splits:
        split_dir = os.path.join(target_root_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    metadata_csv = pd.read_csv(csv_file_path)
    print(metadata_csv.columns)  # Print the columns to debug
    total_files = len(metadata_csv)
    
    with tqdm(total=total_files, desc="Moving images based on split", unit="file") as pbar:
        for _, row in metadata_csv.iterrows():
            patch_id = row['patch_id']
            split = row['split']
            if split in splits:
                source_file_path = os.path.join(source_root_dir, f"{patch_id}.jpg")
                target_file_path = os.path.join(target_root_dir, split, f"{patch_id}.jpg")
                
                if os.path.exists(source_file_path):
                    shutil.move(source_file_path, target_file_path)
                    pbar.update(1)
        

if __name__ == "__main__":
    source_root_directory = r'E:\CombinedImages'
    target_directory = r'E:\CombinedImages\CombinedImages'
    move_images_to_single_folder(source_root_directory, target_directory)