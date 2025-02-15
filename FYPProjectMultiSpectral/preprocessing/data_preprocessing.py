import os
from preprocessing.preprocessing_helper_functions import *
from config.config import DatasetConfig

# Function to preprocess the BigEarthNet dataset
def bigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_df, snow_cloud_metadata_df):
    # -- Stage 1: Download and Extract the dataset --
    if os.path.exists(dataset_dir):
        print('The dataset has already been downloaded')
    else:
        downloadAndExtractDataset(dataset_dir)

    # -- Stage 2: Remove uncessary files and folders based on the provided metadata --
    if os.path.exists(snow_cloud_metadata_df):
        removeUnnecessaryFiles(dataset_dir, snow_cloud_metadata_df)
    
    # -- Stage 3: Resize the TIFF files to the same resolution (120x120) --
    bands_of_interest = ['B01', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
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

    # -- Stage 4: Combine TIFF files into a single image --
    process_folders(dataset_dir, 'CombinedImages', combineTiffs, exclude_dirs=["CombinedImages"])

    # -- Stage 5: Create subsets of the dataset and filter metadata --
    subsets = {
        '50%': os.path.join(subset_dir, '50%_BigEarthNet', 'images'),
        '10%': os.path.join(subset_dir, '10%_BigEarthNet', 'images'),
        '5%': os.path.join(subset_dir, '5%_BigEarthNet', 'images'),
        '1%': os.path.join(subset_dir, '1%_BigEarthNet', 'images'),
        '0.5%': os.path.join(subset_dir, '0.5%_BigEarthNet', 'images')
    }
    metadata_df = pd.read_csv(DatasetConfig.metadata_paths['100'])

    copy_subset_images(dataset_dir, metadata_df, DatasetConfig.metadata_paths['50'], subsets['50%'])
    copy_subset_images(dataset_dir, metadata_df, DatasetConfig.metadata_paths['10'], subsets['10%'])
    copy_subset_images(dataset_dir, metadata_df, DatasetConfig.metadata_paths['5'], subsets['5%'])
    copy_subset_images(dataset_dir, metadata_df, DatasetConfig.metadata_paths['1'], subsets['1%'])
    copy_subset_images(dataset_dir, metadata_df, DatasetConfig.metadata_paths['0.5'], subsets['0.5%'])

    full_subfolder_count, folder = count_tif_images(dataset_dir)
    half_subfolder_count, folder = count_tif_images(subsets['50%'])
    tenth_subfolder_count, folder = count_tif_images(subsets['10%'])
    fifth_subfolder_count, folder = count_tif_images(subsets['5%'])
    hundredth_subfolder_count, folder = count_tif_images(subsets['1%'])
    half_percent_subfolder_count, folder = count_tif_images(subsets['0.5%'])

    # Display the counts and percentages for each folder
    print(f"Total subfolder count in full dataset: {full_subfolder_count}\n")
    display_percentage(half_subfolder_count, full_subfolder_count, '50%BigEarthNet')
    display_percentage(tenth_subfolder_count, full_subfolder_count, '10%BigEarthNet')
    display_percentage(fifth_subfolder_count, full_subfolder_count, '5%BigEarthNet')
    display_percentage(hundredth_subfolder_count, full_subfolder_count, '1%BigEarthNet')
    display_percentage(half_percent_subfolder_count, full_subfolder_count, '0.5%BigEarthNet')

    # -- Stage 6: Pre-compute sample weights for the subsets --

def precompute_sample_weights_for_subsets(metadata_csv, subset_dirs, num_classes, label_column="labels"):
    for subset, subset_dir in subset_dirs.items():
        metadata_path = os.path.join(subset_dir, f"metadata_{subset.replace('%','')}_percent.csv")
        if not os.path.exists(metadata_path):
            print(f"[WARNING] Metadata file for subset {subset} not found at {metadata_path}. Skipping.")
            continue

        subset_df = pd.read_csv(metadata_path)
        # Define the cache file path within the subset folder.
        cache_file = os.path.join(subset_dir, f"{subset}_sample_weights.npy")
        precompute_sample_weights(subset_df, num_classes, label_column, cache_file)
        print(f"[INFO] Precomputed sample weights for subset '{subset}' saved to {cache_file}")

############################################################################################################
if __name__ == '__main__':
    # dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\OnePBigEarthNetCopySubsets2\50_percent' # Full path of the dataset
    # subset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets' # Full path of the subset directory
    # metadata_file = r'C:\Users\isaac\Downloads\metadata.parquet' # Full path of the metadata file
    # unwanted_metadata_file = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet' # Full path of the unwanted metadata
    
    # bigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_file, unwanted_metadata_file)  

    subset_dirs = {
        "0.5%": r"C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet",
        # "1%":   r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet",
        # "5%":   r"C:\Users\isaac\Desktop\BigEarthTests\5%_BigEarthNet",
        # "10%":  r"C:\Users\isaac\Desktop\BigEarthTests\10%_BigEarthNet",
        # "50%":  r"C:\Users\isaac\Desktop\BigEarthTests\50%_BigEarthNet"
    }
    
    # For example, use the full metadata (or adjust as needed).
    metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['100'])
    num_classes = DatasetConfig.num_classes
    
    # Precompute weights for each subset.
    precompute_sample_weights_for_subsets(metadata_csv, subset_dirs, num_classes, label_column="labels")