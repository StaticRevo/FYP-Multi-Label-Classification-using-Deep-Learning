import numpy as np

# Replace the path with your actual file path
npz_file_path = r'C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_10%_BigEarthNet_2epochs\results\test_predictions_ResNet18_10%_BigEarthNet.npz'

# Load the .npz file; this returns a dict-like object
data = np.load(npz_file_path)

# Print a header
print("Contents of the .npz file (first 5 entries per key):\n")

# Iterate over the keys in the .npz file
for key in data.files:
    array = data[key]
    print(f"Key: {key}")
    print(f"Shape: {array.shape}")
    
    # Check if the array is long enough, then print the first 5 elements.
    # If it's multidimensional, the slicing will work on the first dimension.
    print("First 5 entries:")
    print(array[:5])
    print("-" * 40)
