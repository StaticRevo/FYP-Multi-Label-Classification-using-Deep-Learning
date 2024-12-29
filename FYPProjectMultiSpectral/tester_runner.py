# Standard library imports
import json
import subprocess
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataclasses import dataclass, field
import torch.nn as nn
import ast
import numpy as np
from utils.helper_functions import extract_number
from config.config_utils import calculate_class_weights, clean_and_parse_labels, calculate_class_labels
from config.config import DatasetConfig, ModelConfig
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

class CheckpointSelectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Checkpoint Selector")

        # Labels and Entry fields for each checkpoint
        self.labels = {
            "best_acc": "Best Accuracy Checkpoint:",
            "best_loss": "Best Loss Checkpoint:",
            "last": "Last Checkpoint:"
        }

        self.entries = {}

        # Create GUI elements
        for idx, (key, label_text) in enumerate(self.labels.items()):
            label = tk.Label(master, text=label_text)
            label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')

            entry = tk.Entry(master, width=50)
            entry.grid(row=idx, column=1, padx=10, pady=5)
            self.entries[key] = entry

            browse_button = tk.Button(master, text="Browse", command=lambda k=key: self.browse_file(k))
            browse_button.grid(row=idx, column=2, padx=10, pady=5)

        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=len(self.labels), column=1, pady=20)

        # Initialize paths
        self.paths = {
            "best_acc": "",
            "best_loss": "",
            "last": ""
        }

    def browse_file(self, key):
        file_path = filedialog.askopenfilename(
            title=f"Select {self.labels[key]}",
            filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, file_path)
            self.paths[key] = file_path

    def submit(self):
        for key, entry in self.entries.items():
            path = entry.get().strip()
            if not path:
                messagebox.showerror("Input Error", f"Please provide a path for {self.labels[key]}")
                return
            self.paths[key] = path

        # All paths are collected, proceed to run the subprocess
        self.master.destroy()  # Close the GUI window

    def get_paths(self):
        return self.paths

# Training the model
def main():
    # Parse command-line arguments
    if len(sys.argv) < 5:
        print("Usage: python main.py <model_name> <weights> <selected_bands> <selected_dataset>")
        sys.exit(1)

    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]

    num = str(extract_number(selected_dataset))

    metadata_path = DatasetConfig.metadata_paths[num]
    metadata_csv = pd.read_csv(metadata_path)   

    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    # Determine the number of channels and selected bands
    if selected_bands == 'all_bands':
        in_channels = len(DatasetConfig.all_bands)
        bands = DatasetConfig.all_bands
    elif selected_bands == 'rgb_bands':
        in_channels = len(DatasetConfig.rgb_bands)
        bands = DatasetConfig.rgb_bands
    elif selected_bands == 'rgb_nir_bands':
        in_channels = len(DatasetConfig.rgb_nir_bands)
        bands = DatasetConfig.rgb_nir_bands
    elif selected_bands == 'rgb_swir_bands':
        in_channels = len(DatasetConfig.rgb_swir_bands)
        bands = DatasetConfig.rgb_swir_bands
    elif selected_bands == 'rgb_nir_swir_bands':
        in_channels = len(DatasetConfig.rgb_nir_swir_bands)
        bands = DatasetConfig.rgb_nir_swir_bands
    else:
        raise ValueError(f"Unknown selected_bands option: {selected_bands}")

    dataset_dir = DatasetConfig.dataset_paths[num]

    # Initialize and run the GUI
    root = tk.Tk()
    gui = CheckpointSelectorGUI(root)
    root.mainloop()

    # Retrieve the selected paths
    paths = gui.get_paths()
    best_acc_checkpoint_path = paths["best_acc"]
    best_loss_checkpoint_path = paths["best_loss"]
    last_checkpoint_path = paths["last"]

    # Validate that all checkpoint paths were selected
    if not all([best_acc_checkpoint_path, best_loss_checkpoint_path, last_checkpoint_path]):
        print("Error: All checkpoint files must be selected.")
        sys.exit(1)

    # Prepare the arguments for the subprocess
    args = [
        'python', 
        'FYPProjectMultiSpectral\\tester.py', 
        model_name, 
        weights, 
        selected_bands, 
        selected_dataset, 
        best_acc_checkpoint_path, 
        best_loss_checkpoint_path, 
        last_checkpoint_path,
        str(in_channels),
        json.dumps(class_weights.tolist()),
        metadata_path, 
        dataset_dir, 
        json.dumps(bands)
    ]

    # # Print the arguments for debugging
    # print("Arguments to subprocess.run:")
    # for arg in args:
    #     print(arg)

    # Run the subprocess
    subprocess.run(args)

if __name__ == "__main__":
    main()
