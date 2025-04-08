import torch
from pprint import pprint  # For pretty-printing

# Paths to the checkpoints
ckpt_path = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_50epochs\checkpoints\last.ckpt"
ckpt_path2 = r"C:\Users\isaac\OneDrive\Documents\GitHub\Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery\FYPProjectMultiSpectral\ensemble_results\resnet18_resnet50\ensemble_model_resnet18_resnet50.ckpt"
ckpt_path3 = r"C:\Users\isaac\Desktop\experiments\CustomModelV9_None_all_bands_10%_BigEarthNet_50epochs_1\checkpoints\last.ckpt"
# Function to load and inspect a checkpoint
def inspect_checkpoint(ckpt_path, name):
    print(f"\nInspecting checkpoint: {name}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # Print keys in the checkpoint
    print("\nKeys in the checkpoint:")
    pprint(checkpoint.keys())

    # Inspect state_dict keys
    if "state_dict" in checkpoint:
        print("\nKeys in the state_dict:")
        state_dict_keys = list(checkpoint["state_dict"].keys())
        print(f"Number of keys in state_dict: {len(state_dict_keys)}")
        print(f"First 10 keys: {state_dict_keys[:10]}")  # Print only the first 10 keys for brevity

    # Inspect additional metadata (if present)
    if "model_configs" in checkpoint:
        print("\nModel configurations:")
        pprint(checkpoint["model_configs"])
    if "weights" in checkpoint:
        print("\nEnsemble weights:")
        pprint(checkpoint["weights"])

    return checkpoint

# Function to compare two checkpoints
def compare_checkpoints(ckpt1, ckpt2):
    print("\n=== Comparing Checkpoints ===")
    
    # Compare keys
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())
    print("\nKeys only in first checkpoint:", keys1 - keys2)
    print("Keys only in second checkpoint:", keys2 - keys1)

    # Compare state_dict keys
    if "state_dict" in ckpt1 and "state_dict" in ckpt2:
        state_dict_keys1 = set(ckpt1["state_dict"].keys())
        state_dict_keys2 = set(ckpt2["state_dict"].keys())
        print("\nState_dict keys only in first checkpoint:", state_dict_keys1 - state_dict_keys2)
        print("State_dict keys only in second checkpoint:", state_dict_keys2 - state_dict_keys1)

        # Compare weights of a specific layer (if it exists in both)
        layer_name = "layer1.0.conv1.weight"  # Replace with a layer name from your model
        if layer_name in state_dict_keys1 and layer_name in state_dict_keys2:
            weights1 = ckpt1["state_dict"][layer_name]
            weights2 = ckpt2["state_dict"][layer_name]
            are_weights_equal = torch.equal(weights1, weights2)
            print(f"\nAre weights for {layer_name} identical? {are_weights_equal}")
            if not are_weights_equal:
                print(f"Difference in weights for {layer_name}:")
                print("First checkpoint weights:", weights1)
                print("Second checkpoint weights:", weights2)

    # Compare additional metadata
    if "model_configs" in ckpt1 and "model_configs" in ckpt2:
        print("\nComparing model configurations:")
        pprint(ckpt1["model_configs"])
        pprint(ckpt2["model_configs"])
    if "weights" in ckpt1 and "weights" in ckpt2:
        print("\nComparing ensemble weights:")
        print("First checkpoint weights:", ckpt1["weights"])
        print("Second checkpoint weights:", ckpt2["weights"])

# Inspect both checkpoints
checkpoint1 = inspect_checkpoint(ckpt_path, "Normal Checkpoint")
checkpoint2 = inspect_checkpoint(ckpt_path2, "Ensemble Checkpoint")
checkpoint3 = inspect_checkpoint(ckpt_path3, "Custom Checkpoint")

# Compare the two checkpoints
compare_checkpoints(checkpoint1, checkpoint2)