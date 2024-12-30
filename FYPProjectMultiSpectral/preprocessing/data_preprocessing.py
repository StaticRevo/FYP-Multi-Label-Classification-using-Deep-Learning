import os
from preprocessing.preprocessing_helper_functions import *

# Function to preprocess the BigEarthNet dataset
def bigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_df, snow_cloud_metadata_df):
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
        '50%': os.path.join(subset_dir, '50%_BigEarthNet'),
        '10%': os.path.join(subset_dir, '10%_BigEarthNet'),
        '5%': os.path.join(subset_dir, '5%_BigEarthNet'),
        '1%': os.path.join(subset_dir, '1%_BigEarthNet'),
        '0.5%': os.path.join(subset_dir, '0.5%_BigEarthNet')
    }
    metadata_df = pd.read_parquet(metadata_file)

    createSubsets(dataset_dir, subsets['50%'], metadata_df, 50)
    createSubsets(dataset_dir, subsets['10%'], metadata_df, 10)
    createSubsets(dataset_dir, subsets['5%'], metadata_df, 5)
    createSubsets(dataset_dir, subsets['1%'], metadata_df, 1)
    createSubsets(dataset_dir, subsets['0.5%'], metadata_df, 0.5)

    full_subfolder_count, folder = count_subfolders(dataset_dir, '100%BigEarthNet')
    half_subfolder_count, folder = count_subfolders(subsets['50%'], '50%BigEarthNet' )
    tenth_subfolder_count, folder = count_subfolders(subsets['10%'], '10%BigEarthNet' )
    fifth_subfolder_count, folder = count_subfolders(subsets['5%'], '5%BigEarthNet' )
    hundredth_subfolder_count, folder = count_subfolders(subsets['1%'], '1%BigEarthNet')
    half_percent_subfolder_count, folder = count_subfolders(subsets['0.5%'], '0.5%BigEarthNet')

    # Display the counts and percentages for each folder
    print(f"Total subfolder count in full dataset: {full_subfolder_count}\n")
    display_percentage(half_subfolder_count, full_subfolder_count, '50%BigEarthNet')
    display_percentage(tenth_subfolder_count, full_subfolder_count, '10%BigEarthNet')
    display_percentage(fifth_subfolder_count, full_subfolder_count, '5%BigEarthNet')
    display_percentage(hundredth_subfolder_count, full_subfolder_count, '1%BigEarthNet')
    display_percentage(half_percent_subfolder_count, full_subfolder_count, '0.5%BigEarthNet')

    # Stage 4: Resize the TIFF files to the same resolution (120x120)
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

    # Stage 5: Combine TIFF files into a single image
    process_folders(dataset_dir, 'CombinedImages', combineTiffs, exclude_dirs=["CombinedImages"])

    # Stage 7: Split the dataset into training, validation and testing sets
    # Stage 8: Apply normalisation and data augmentation techniques
    # Stage 9: Save the preprocessed dataset to a specified location
    

############################################################################################################
if __name__ == '__main__':
    dataset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\OnePBigEarthNetCopySubsets2\50_percent'
    subset_dir = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets'
    metadata_file = r'C:\Users\isaac\Downloads\metadata.parquet'
    unwanted_metadata_file = r'C:\Users\isaac\Downloads\metadata_for_patches_with_snow_cloud_or_shadow.parquet'
    
    bigEarthNetDataPreprocessing(dataset_dir, subset_dir, metadata_file, unwanted_metadata_file)
