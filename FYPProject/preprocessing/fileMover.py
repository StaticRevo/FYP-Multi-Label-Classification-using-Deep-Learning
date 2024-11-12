import os
import shutil
from tqdm import tqdm

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

if __name__ == "__main__":
    source_root_directory = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImages'
    target_directory = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedImagesTIF'
    move_images_to_single_folder(source_root_directory, target_directory)