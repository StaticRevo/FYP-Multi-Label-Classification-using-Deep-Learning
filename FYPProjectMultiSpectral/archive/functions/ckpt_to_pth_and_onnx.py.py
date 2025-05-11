# Standard library imports
import os

# Third-party imports
import torch
import pandas as pd
import onnx
from onnx_tf.backend import prepare

# Local application imports
from models.models import *  
from config.config import DatasetConfig, calculate_class_weights

# Function to convert ONNX model to TensorFlow format
def convert_onnx_to_tf(onnx_model_path, tf_model_path):
    onnx_model = onnx.load(onnx_model_path) # Load ONNX model
    tf_rep = prepare(onnx_model) # Convert ONNX model to TensorFlow forma
    tf_rep.export_graph(tf_model_path)

    print(f"Model successfully converted to TensorFlow format at: {tf_model_path}")

# Function to convert PyTorch checkpoint to `.pth` and ONNX formats
def convert_ckpt_to_pth_and_onnx(checkpoint_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)


    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu")) # Load checkpoint
    print("Checkpoint keys:", checkpoint.keys()) # Check available keys in the checkpoint

    # Load dataset metadata for class weights
    metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['1'])
    class_weights = calculate_class_weights(metadata_csv)

    # Initialize model
    model = CustomWRNB0(
        class_weights=class_weights,
        num_classes=DatasetConfig.num_classes,
        in_channels=12,
        model_weights=None,
        main_path=os.path.dirname(checkpoint_path)
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False) # Load model weights, ignoring extra keys
    model.eval() # Set to evaluation mode

    # Save a cleaned `.pth` model
    pth_path = os.path.join(save_dir, "B0.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Cleaned model weights saved at: {pth_path}")

    # Define ONNX file path
    onnx_path = os.path.join(save_dir, "B0_onnx.onnx")
    dummy_input = torch.randn(1, 12, 120, 120)

    # Convert to ONNX and save
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )

    print(f"Model successfully converted to ONNX and saved at: {onnx_path}")

if __name__ == "__main__":
    onnx_model_path = r"C:\Users\isaac\Desktop\Experiment Folders\converted_model.onnx"
    tf_model_path = r"C:\Users\isaac\Desktop\Experiment Folders\tf_model"
    convert_onnx_to_tf(onnx_model_path, tf_model_path)

    checkpoint_path = r"C:\Users\isaac\Desktop\experiments\CustomWRNB0_None_all_bands_0.5%_BigEarthNet_100epochs\checkpoints\last.ckpt"
    save_dir = r"C:\Users\isaac\Desktop\Experiment Folders"
    convert_ckpt_to_pth_and_onnx(checkpoint_path, save_dir)