# Standard library imports
import json
import subprocess
import sys
import os

# Third-party imports
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

# Local application imports
from utils.data_utils import extract_number
from config.config_utils import calculate_class_weights
from config.config import DatasetConfig

# GUI for selecting a checkpoint file
class CheckpointSelectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Checkpoint Selector")

        # Label for the checkpoint file
        self.label_text = "Select Checkpoint File:"
        self.label = tk.Label(master, text=self.label_text)
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky='e')

        # Entry field for the checkpoint file
        self.entry = tk.Entry(master, width=50)
        self.entry.grid(row=0, column=1, padx=10, pady=5)

        # Browse button to open file dialog
        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=5)

        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=1, column=1, pady=20)

        # Initialise the checkpoint path
        self.checkpoint_path = ""

    # Open a file dialog to select a checkpoint file
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title=f"Select {self.label_text}",
            filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, file_path)
            self.checkpoint_path = file_path

    # Validate the input and close the window
    def submit(self):
        path = self.entry.get().strip()
        if not path:
            messagebox.showerror("Input Error", f"Please provide a path for {self.label_text}")
            return
        self.checkpoint_path = path
        self.master.destroy()

    def get_checkpoint(self):
        return self.checkpoint_path

def main():
    if len(sys.argv) < 5: # Check if the required arguments are provided
        print("Usage: python main.py <model_name> <weights> <selected_bands> <selected_dataset>")
        sys.exit(1)

    # Parse command-line arguments
    model_name = sys.argv[1]
    weights = sys.argv[2]
    selected_bands = sys.argv[3]
    selected_dataset = sys.argv[4]

    num = str(extract_number(selected_dataset))
    metadata_path = DatasetConfig.metadata_paths[num]
    metadata_csv = pd.read_csv(metadata_path)

    class_weights = calculate_class_weights(metadata_csv) # Calculate class weights from the metadata CSV

    # Determine the number of channels and the bands based on the selected_bands option
    if selected_bands == 'all_bands':
        in_channels = len(DatasetConfig.all_bands)
        bands = DatasetConfig.all_bands
    elif selected_bands == 'all_imp_bands':
        in_channels = len(DatasetConfig.all_imp_bands)
        bands = DatasetConfig.all_imp_bands
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

    # Initialise and run the GUI to select a single checkpoint file
    root = tk.Tk()
    gui = CheckpointSelectorGUI(root)
    root.mainloop()
    checkpoint_path = gui.get_checkpoint()

    # Validate that a checkpoint path was selected
    if not checkpoint_path:
        print("Error: A checkpoint file must be selected.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tester_script = os.path.join(script_dir, 'tester.py')

    # Prepare the arguments for the subprocess.
    args = [
        'python', 
        tester_script,
        model_name, 
        weights, 
        selected_dataset, 
        checkpoint_path, 
        str(in_channels),
        json.dumps(class_weights.tolist()),
        metadata_path, 
        dataset_dir, 
        json.dumps(bands)
    ]

    # Run the subprocess (tester.py will be called with the given arguments)
    subprocess.run(args)

if __name__ == "__main__":
    main()
