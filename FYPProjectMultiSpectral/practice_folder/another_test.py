import torch
from pprint import pprint

# Path to the checkpoint
ckpt_path3 = r"C:\Users\isaac\Desktop\experiments\CustomModelV9_None_all_bands_10%_BigEarthNet_50epochs_1\checkpoints\last.ckpt"
ckpt_path = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_50epochs\checkpoints\last.ckpt"

# Load the checkpoint
checkpoint = torch.load(ckpt_path3, map_location=torch.device('cpu'), weights_only=True)
checkpoint2 = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)

# Print all keys in the checkpoint
print("\nKeys in the checkpoint:")
pprint(checkpoint.keys())

# Print all keys in the checkpoint
print("\nKeys in the checkpoint2:")
pprint(checkpoint2.keys())

