import torch
import pandas as pd
import os
from models.models import *  # Import your model class
from config.config import DatasetConfig, calculate_class_weights

# Define the directory where you want to save the models
save_dir = r"C:\Users\isaac\Desktop\Experiment Folders"
os.makedirs(save_dir, exist_ok=True) # Create the directory if it does not exist

# Load checkpoint
checkpoint_path = r"C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_0.5%_BigEarthNet_5epochs_3\checkpoints\last.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Check available keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# Load dataset metadata for class weights
metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['0.5'])
class_weights = calculate_class_weights(metadata_csv)

# Initialize model (Manually set parameters because 'hyper_parameters' is missing)
model = ResNet18(
    class_weights=class_weights,  
    num_classes=DatasetConfig.num_classes, 
    in_channels=12,  # Ensure this matches your dataset
    model_weights=None, 
    main_path=r'C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_0.5%_BigEarthNet_5epochs_3'
)

# Load model weights, ignoring extra keys
model.load_state_dict(checkpoint["state_dict"], strict=False) 

# Set to evaluation mode
model.eval()

# Save a cleaned `.pth` model in the specified directory
pth_path = os.path.join(save_dir, "clean_model_ResNet18.pth")
torch.save(model.state_dict(), pth_path)
print(f"Cleaned model weights saved at: {pth_path}")

# Define ONNX file path in the specified directory
onnx_path = os.path.join(save_dir, "converted_model_ResNet18.onnx")

# Define a dummy input for the model (ensure correct input channels)
dummy_input = torch.randn(1, 12, 120, 120)  # Updated to match in_channels=12

# Convert to ONNX and save in the specified directory
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14  # Compatible version for Netron
)

print(f"Model successfully converted to ONNX and saved at: {onnx_path}")
