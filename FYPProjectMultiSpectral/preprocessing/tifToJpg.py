import os
from PIL import Image

def convert_tif_to_jpg(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                tif_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(tif_path, root_dir)
                jpg_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.jpg')
                
                os.makedirs(os.path.dirname(jpg_path), exist_ok=True)
                
                with Image.open(tif_path) as img:
                    img.convert('RGB').save(jpg_path, 'JPEG')
               
if __name__ == "__main__":
    root_directory = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImages'
    output_directory = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImagesJPG'
    convert_tif_to_jpg(root_directory, output_directory)