import rasterio
import numpy as np
import matplotlib.pyplot as plt

def display_rgb_image(tiff_file_path):
    with rasterio.open(tiff_file_path) as src:
        # Read the red, green, and blue bands
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Normalize each band to the range 0-1
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        blue = blue.astype(np.float32)
        
        red /= np.max(red)
        green /= np.max(green)
        blue /= np.max(blue)
        
        # Stack the bands into an RGB image
        rgb = np.dstack((red, green, blue))
        
        # Display the RGB image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()

tiff_file_path = r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_68.tif'
display_rgb_image(tiff_file_path)