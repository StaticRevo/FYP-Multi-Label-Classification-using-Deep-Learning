import torch
from pprint import pprint  # For pretty-printing

# Path to the checkpoint
ckpt_path = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_50epochs\checkpoints\last.ckpt"
ckpt_path2 = r"C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\FYPProjectMultiSpectral\ensemble_results\ensemble_model.ckpt"

# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu')) 

# Inspect the checkpoint contents
print("Keys in the checkpoint:")
pprint(checkpoint.keys())  # Pretty-print the keys

# If it contains a state_dict, inspect its keys
if "state_dict" in checkpoint:
    print("\nKeys in the state_dict:")
    state_dict_keys = list(checkpoint["state_dict"].keys())  # Convert to a list for better formatting
    for key in state_dict_keys:
        print(key)

print()
# Load the checkpoint
checkpoint2 = torch.load(ckpt_path2, map_location=torch.device('cpu')) 

# Inspect the checkpoint contents
print("Keys in the checkpoint:")
pprint(checkpoint2.keys())  # Pretty-print the keys

# If it contains a state_dict, inspect its keys
if "state_dict" in checkpoint2:
    print("\nKeys in the state_dict:")
    state_dict_keys = list(checkpoint2["state_dict"].keys())  # Convert to a list for better formatting
    for key in state_dict_keys:
        print(key)