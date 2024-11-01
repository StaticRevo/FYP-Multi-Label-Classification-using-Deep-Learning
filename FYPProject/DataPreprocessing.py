import os
import requests
import zstandard as zstd
from tqdm import tqdm
import pandas as pd
import random
import shutil

# Function to preprocess the BigEarthNet dataset
def BigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_df, snow_cloud_metadata_df):
    # Stage 1: Download and Extract the dataset   
    if os.path.exists(dataset_dir):
        print('The dataset has already been downloaded')
    else:
        downloadAndExtractDataset(dataset_dir)

    # Stage 2: Remove uncessary files and folders based on the provided metadata
    if os.path.exists(unwanted_metadata_file):
        removeUnnecessaryFiles(dataset_dir, unwanted_metadata_file)
    
    # Stage 3: Create subsets of the dataset and filter metadata
    subsets = {
        '50%': os.path.join(subset_dir, '50%'),
        '10%': os.path.join(subset_dir, '10%'),
        '1%': os.path.join(subset_dir, '1%')
    }
    metadata_df = pd.read_parquet(metadata_file)

    if os.path.exists(subsets['50%']) and os.path.exists(subsets['10%']) and os.path.exists(subsets['1%']):
        print('Subsets already exist')
    else:
        createSubsets(dataset_dir, subsets, metadata_df)

    full_subfolder_count, folder = count_subfolders(dataset_dir, '100%BigEarthNet')
    half_subfolder_count, folder = count_subfolders(subsets['50%'], '50%BigEarthNet' )
    tenth_subfolder_count, folder = count_subfolders(subsets['10%'], '10%BigEarthNet' )
    hundredth_subfolder_count, folder = count_subfolders(subsets['1%'], '1%BigEarthNet')

    # Display the counts and percentages for each folder
    print(f"Total subfolder count in full dataset: {full_subfolder_count}\n")
    display_percentage(half_subfolder_count, full_subfolder_count, '50%BigEarthNet')
    display_percentage(tenth_subfolder_count, full_subfolder_count, '10%BigEarthNet')
    display_percentage(hundredth_subfolder_count, full_subfolder_count, '1%BigEarthNet')

    # Stage 4: Add a binary vector to the metadata files to indicate the presence of a specific land cover class
    unique_labels = metadata_df['labels'].explode().unique()
    print(unique_labels)

    metadata_df['binary_vector'] = metadata_df['labels'].apply(lambda x: labels_to_binary_vector(x, unique_labels))

    print(metadata_df.columns, "\n")

    # Save the updated DataFrame to a new Parquet file
    updated_metadata_file = r'C:\Users\isaac\Downloads\updated_metadata.parquet'
    metadata_df.to_parquet(updated_metadata_file)

    updated_metadata_df = pd.read_parquet(updated_metadata_file)
    print(updated_metadata_df.columns, "\n")

    # Stage 5: Resize the TIFF files to the same resolution (120x120)

    # Stage 6: Combine TIFF files into a single image

    # Stage 7: Split the dataset into training, validation and testing sets

    # Stage 8: Apply normalisation and data augmentation techniques

    # Stage 9: Save the preprocessed dataset to a specified location
    

# Helper Functions  
def downloadAndExtractDataset(dataset_dir):
    download_url = 'https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1'
    chunk_size = 1024
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status() 
        total = int(response.headers.get('content-length', 0))
        block_size = chunk_size

        with open(dataset_dir, 'wb') as file, tqdm(total=total, unit='iB', unit_scale=True, desc='Downloading BigEarthNet', unit_divisor=1024) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print('The dataset has been downloaded successfully')
    except requests.exceptions.RequestException as e:
        print(f"Error during dataset download: {e}")
        raise
    
    try:
        with open(dataset_dir, 'rb') as file_in:
            dctx = zstd.ZstdDecompressor()
            with open(dataset_dir[:-4] + '.tar', 'wb') as file_out:
                dctx.copy_stream(file_in, file_out)
        print(f"Decompressed dataset to: {dataset_dir[:-4] + '.tar'}")
    except Exception as e:
        print(f"Error during dataset decompression: {e}")
        raise

def createSubsets(dataset_dir, subsets, metadata_df, subset_dir):
    metadata_50 = pd.DataFrame(columns=metadata_df.columns)
    metadata_10 = pd.DataFrame(columns=metadata_df.columns)
    metadata_1 = pd.DataFrame(columns=metadata_df.columns)

    for subset in subsets.values():
        if not os.path.exists(subset):
            os.makedirs(subset)
            print(f"Created subset: {subset}")
        else:
            print(f"Subset already exists: {subset}")
    
    for folder in tqdm(os.listdir(dataset_dir), desc='Creating subsets'):
        folder_path = os.path.join(dataset_dir, folder)

        if os.path.isdir(folder_path):
            # List all subfolders
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

            # Calculate number of folders in each subset
            num_subfolders = len(subfolders)
            num_50_percent = min(max(1, num_subfolders // 2), num_subfolders)
            num_10_percent = min(max(1, num_subfolders // 10), num_subfolders)
            num_1_percent = min(max(1, num_subfolders // 100), num_subfolders)

            # Select random subfolders for each subset
            selected_50 = random.sample(subfolders, num_50_percent)
            selected_10 = random.sample(subfolders, num_10_percent)
            selected_1 = random.sample(subfolders, num_1_percent)

            # Filter metadata for each subset
            metadata_50 = pd.concat([metadata_50, metadata_df[metadata_df['patch_id'].isin(selected_50)]])
            metadata_10 = pd.concat([metadata_10, metadata_df[metadata_df['patch_id'].isin(selected_10)]])
            metadata_1 = pd.concat([metadata_1, metadata_df[metadata_df['patch_id'].isin(selected_1)]])

            # Copy selected subfolders to each subset directory
            for selected in selected_50:
                dest_path = os.path.join(subsets['50%'], folder, selected)
                if not os.path.exists(dest_path):  # Check if the destination folder exists
                    try:
                        shutil.copytree(os.path.join(folder_path, selected), dest_path)
                    except FileExistsError:
                        print(f"Directory already exists, skipping: {dest_path}")

            for selected in selected_10:
                dest_path = os.path.join(subsets['10%'], folder, selected)
                if not os.path.exists(dest_path):  # Check if the destination folder exists
                    try:
                        shutil.copytree(os.path.join(folder_path, selected), dest_path)
                    except FileExistsError:
                        print(f"Directory already exists, skipping: {dest_path}")
            
            for selected in selected_1:
                dest_path = os.path.join(subsets['1%'], folder, selected)
                if not os.path.exists(dest_path):  # Check if the destination folder exists
                    try:
                        shutil.copytree(os.path.join(folder_path, selected), dest_path)
                    except FileExistsError:
                        print(f"Directory already exists, skipping: {dest_path}")

    # Save metadata for each subset
    metadata_50.to_csv(os.path.join(subset_dir, 'metadata_50_percent.csv'), index=False)
    metadata_10.to_csv(os.path.join(subset_dir, 'metadata_10_percent.csv'), index=False)
    metadata_1.to_csv(os.path.join(subset_dir, 'metadata_1_percent.csv'), index=False)

def count_subfolders(base_dir, folder):
    # Dictionary to hold folder counts
    folder_counts = {}
    total_subfolders = 0  # Initialize total subfolder counter
    
    # Iterate through all folders in the base directory
    for folder in tqdm(os.listdir(base_dir), desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder)
        
        # Check if the current path is a directory
        if os.path.isdir(folder_path):
            # Count subdirectories within this folder
            subfolder_count = sum(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in os.listdir(folder_path))
            folder_counts[folder] = subfolder_count
        
            # Update total subfolder count
            total_subfolders += subfolder_count

    # Print total subfolders
    return total_subfolders, folder

# Function to calculate and display subfolder count and percentage
def display_percentage(partial_count, full_count, folder_name):
    percentage = (partial_count / full_count) * 100
    print(f"Folder: {folder_name} | Subfolder Count: {partial_count} | Percentage: {percentage:.2f}%")

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

def labels_to_binary_vector(labels, unique_labels):
        binary_vector = [1 if label in labels else 0 for label in unique_labels]
        return binary_vector

############################################################################################################
if __name__ == '__main__':
    dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\OnePBigEarthNetCopy'
    subset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets'
    metadata_file = r'C:\Users\isaac\Downloads\metadata.parquet'
    unwanted_metadata_file = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    
    metadata_df = pd.read_parquet(metadata_file)
    snow_cloud_metadata_df = pd.read_parquet(unwanted_metadata_file)


    #BigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_file, unwanted_metadata_file)