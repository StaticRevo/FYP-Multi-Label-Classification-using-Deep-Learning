import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from threading import Thread
import queue
import json
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'FYPProjectMultiSpectral'))

# Local application imports
from FYPProjectMultiSpectral.utils.data_utils import extract_number
from FYPProjectMultiSpectral.config.config_utils import calculate_class_weights
from FYPProjectMultiSpectral.config.config import DatasetConfig

# --- Constants and Selections ---
BG_COLOR = 'black'
FG_COLOR = 'white'
FONT = ("Consolas", 10)

models = {
    '1': 'custom_model',
    '2': 'ResNet18',
    '3': 'ResNet50',
    '4': 'VGG16',
    '5': 'VGG19',
    '6': 'DenseNet121',
    '7': 'EfficientNetB0',
    '8': 'EfficientNet_v2',
    '9': 'Swin-Transformer',
    '10': 'Vit-Transformer'
}

band_selection = {
    '1': 'all_bands',
    '2': 'rgb_bands',
    '3': 'rgb_nir_bands',
    '4': 'rgb_swir_bands',
    '5': 'rgb_nir_swir_bands'
}

dataset_selection = {
    '1': '100%_BigEarthNet',
    '2': '50%_BigEarthNet',
    '3': '10%_BigEarthNet',
    '4': '5%_BigEarthNet',
    '5': '1%_BigEarthNet',
    '6': '0.5%_BigEarthNet'
}


# --- Main Application Class ---
class ModelTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()
        self.process = None  
        self.checkpoint_frame = None  
        self.configure_gui()
        self.create_widgets()
        self.poll_log_queue()

    def configure_gui(self):
        self.title("Model Training Dashboard")
        self.geometry("1000x600")
        self.configure(bg=BG_COLOR)
        self.attributes('-fullscreen', True)
        self.bind("<Map>", lambda event: self.attributes('-fullscreen', True))
        
        # Configure a dark theme for ttk
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background=BG_COLOR, foreground=FG_COLOR)
        style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)
        style.configure('TRadiobutton', background=BG_COLOR, foreground=FG_COLOR)
        style.configure('TButton', background=BG_COLOR, foreground=FG_COLOR)
        style.configure('TFrame', background=BG_COLOR)

    def create_widgets(self):
        # --- Top Title Bar with minimize and close buttons ---
        title_bar = tk.Frame(self, bg=BG_COLOR)
        title_bar.pack(side=tk.TOP, anchor='ne', fill=tk.X)
        minimize_button = ttk.Button(title_bar, text="_", command=self.minimize_window)
        minimize_button.pack(side=tk.RIGHT, padx=5, pady=5)
        close_button = ttk.Button(title_bar, text="X", command=self.close_window)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # --- PanedWindow: Left for selections, Right for logs/progress ---
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(paned, bg=BG_COLOR, width=300)
        paned.add(left_frame, weight=1)
        self.right_frame = tk.Frame(paned, bg=BG_COLOR)
        paned.add(self.right_frame, weight=4)

        # --- LEFT PANEL: Selection Widgets ---
        # Model Selection
        model_frame = ttk.LabelFrame(left_frame, text="Choose a model to run", padding="10")
        model_frame.pack(padx=10, pady=10, fill=tk.X)
        self.model_var = tk.StringVar(value='1')
        for key, model in models.items():
            ttk.Radiobutton(model_frame, text=model, variable=self.model_var, value=key).pack(anchor=tk.W, padx=10, pady=2)

        # Weights Selection
        weights_frame = ttk.LabelFrame(left_frame, text="Choose the weights option", padding="10")
        weights_frame.pack(padx=10, pady=10, fill=tk.X)
        self.weights_var = tk.StringVar(value='1')
        ttk.Radiobutton(weights_frame, text="None", variable=self.weights_var, value='1').pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(weights_frame, text="DEFAULT", variable=self.weights_var, value='2').pack(anchor=tk.W, padx=10, pady=2)

        # Band Selection
        band_frame = ttk.LabelFrame(left_frame, text="Choose the band combination", padding="10")
        band_frame.pack(padx=10, pady=10, fill=tk.X)
        self.band_var = tk.StringVar(value='1')
        for key, bands in band_selection.items():
            ttk.Radiobutton(band_frame, text=bands, variable=self.band_var, value=key).pack(anchor=tk.W, padx=10, pady=2)

        # Dataset Selection
        dataset_frame = ttk.LabelFrame(left_frame, text="Choose the dataset percentage", padding="10")
        dataset_frame.pack(padx=10, pady=10, fill=tk.X)
        self.dataset_var = tk.StringVar(value='1')
        for key, dataset in dataset_selection.items():
            ttk.Radiobutton(dataset_frame, text=dataset, variable=self.dataset_var, value=key).pack(anchor=tk.W, padx=10, pady=2)

        # Train/Test Selection
        train_test_frame = ttk.LabelFrame(left_frame, text="Choose to Train or Test", padding="10")
        train_test_frame.pack(padx=10, pady=10, fill=tk.X)
        self.train_test_var = tk.StringVar(value='train')
        ttk.Radiobutton(train_test_frame, text="Train Only", variable=self.train_test_var, value='train').pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(train_test_frame, text="Train and Test", variable=self.train_test_var, value='train_test').pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(train_test_frame, text="Test Only", variable=self.train_test_var, value='test').pack(anchor=tk.W, padx=10, pady=2)

        # Run and Reset Buttons
        button_frame = tk.Frame(left_frame, bg=BG_COLOR)
        button_frame.pack(pady=10)
        self.run_button = ttk.Button(button_frame, text="Run Model", command=self.run_model)
        self.run_button.pack(side=tk.LEFT, padx=10)
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_selections)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        # --- RIGHT PANEL: Log and Progress Widgets ---
        progress_label = ttk.Label(self.right_frame, text="Training/Testing Progress", font=("Consolas", 14))
        progress_label.pack(pady=10)
        self.log_text = scrolledtext.ScrolledText(self.right_frame, bg=BG_COLOR, fg=FG_COLOR, font=FONT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.progress = ttk.Progressbar(self.right_frame, orient="horizontal", mode="indeterminate", length=400)
        self.progress.pack(padx=10, pady=10)

    def minimize_window(self):
        self.attributes('-fullscreen', False)
        self.iconify()

    def close_window(self):
        self.destroy()

    def build_command(self):
        chosen_model = models.get(self.model_var.get())
        chosen_weights = 'None' if self.weights_var.get() == '1' else f'{chosen_model}_Weights.DEFAULT'
        chosen_bands = band_selection.get(self.band_var.get())
        chosen_dataset = dataset_selection.get(self.dataset_var.get())

        if self.train_test_var.get() == 'test':
            # In test-only mode we handle checkpoint selection separately.
            return None
        else:
            chosen_test = 'True' if self.train_test_var.get() == 'train_test' else 'False'
            return [
                'python',
                'FYPProjectMultiSpectral/trainer.py',
                chosen_model, chosen_weights, chosen_bands, chosen_dataset, chosen_test
            ]

    def build_testing_command(self, selected_checkpoint):
        chosen_model = models.get(self.model_var.get())
        chosen_weights = 'None' if self.weights_var.get() == '1' else f'{chosen_model}_Weights.DEFAULT'
        chosen_bands = band_selection.get(self.band_var.get())
        chosen_dataset = dataset_selection.get(self.dataset_var.get())

        # Dynamic computations based on the selected dataset and bands
        num = str(extract_number(chosen_dataset))
        metadata_path = DatasetConfig.metadata_paths[num]
        metadata_csv = pd.read_csv(metadata_path)
        class_weights, class_weights_array = calculate_class_weights(metadata_csv)
        class_weights = class_weights_array

        # Determine in_channels and bands list based on the selected band combination
        if chosen_bands == 'all_bands':
            in_channels = len(DatasetConfig.all_bands)
            bands = DatasetConfig.all_bands
        elif chosen_bands == 'rgb_bands':
            in_channels = len(DatasetConfig.rgb_bands)
            bands = DatasetConfig.rgb_bands
        elif chosen_bands == 'rgb_nir_bands':
            in_channels = len(DatasetConfig.rgb_nir_bands)
            bands = DatasetConfig.rgb_nir_bands
        elif chosen_bands == 'rgb_swir_bands':
            in_channels = len(DatasetConfig.rgb_swir_bands)
            bands = DatasetConfig.rgb_swir_bands
        elif chosen_bands == 'rgb_nir_swir_bands':
            in_channels = len(DatasetConfig.rgb_nir_swir_bands)
            bands = DatasetConfig.rgb_nir_swir_bands
        else:
            raise ValueError(f"Unknown selected_bands option: {chosen_bands}")

        dataset_dir = DatasetConfig.dataset_paths[num]

        return [
            'python',
            'FYPProjectMultiSpectral/tester.py',
            chosen_model,
            chosen_weights,
            chosen_bands,
            chosen_dataset,
            selected_checkpoint,         # Only the chosen checkpoint is passed
            str(in_channels),
            json.dumps(class_weights.tolist()),
            metadata_path,
            dataset_dir,
            json.dumps(bands)
        ]

    def run_model(self):
        # Clear the log, start the progress bar, and disable the Run button
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Starting training/testing process...\n")
        self.progress.start()
        self.run_button.config(state=tk.DISABLED)

        if self.train_test_var.get() == 'test':
            # For test-only mode, show the checkpoint selector overlay.
            self.open_checkpoint_selector()
        else:
            cmd = self.build_command()
            self.start_subprocess(cmd)

    def open_checkpoint_selector(self):
        """Display an overlay for selecting a checkpoint."""
        self.checkpoint_frame = CheckpointSelectorFrame(
            self.right_frame,
            submit_callback=self.on_checkpoint_submit,
            bg=BG_COLOR
        )
        self.checkpoint_frame.place(relx=0.5, rely=0.5, anchor='center')

    def on_checkpoint_submit(self, selected_checkpoint):
        """Remove the overlay and start the testing process with the selected checkpoint."""
        if self.checkpoint_frame:
            self.checkpoint_frame.destroy()
            self.checkpoint_frame = None

        cmd = self.build_testing_command(selected_checkpoint)
        self.start_subprocess(cmd)

    def start_subprocess(self, cmd):
        def run_subprocess():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                # Stream output into the log queue.
                for line in iter(self.process.stdout.readline, ''):
                    self.log_queue.put(line)
                self.process.stdout.close()
                self.process.wait()
                self.log_queue.put("PROCESS_DONE")
            except Exception as e:
                self.log_queue.put(f"ERROR: {e}")

        Thread(target=run_subprocess, daemon=True).start()

    def poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "PROCESS_DONE":
                    self.progress.stop()
                    self.run_button.config(state=tk.NORMAL)
                    messagebox.showinfo("Success", "Process completed!")
                elif line.startswith("ERROR:"):
                    self.progress.stop()
                    self.run_button.config(state=tk.NORMAL)
                    messagebox.showerror("Error", line)
                else:
                    self.log_text.insert(tk.END, line)
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.poll_log_queue)

    def reset_selections(self):
        self.model_var.set('1')
        self.weights_var.set('1')
        self.band_var.set('1')
        self.dataset_var.set('1')
        self.train_test_var.set('train')
        self.log_text.delete(1.0, tk.END)
        self.run_button.config(state=tk.NORMAL)


# --- Integrated Checkpoint Selector Overlay ---
class CheckpointSelectorFrame(tk.Frame):
    def __init__(self, master, submit_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.submit_callback = submit_callback
        self.configure(bg=BG_COLOR)
        self.labels = {
            "best_acc": "Best Accuracy Checkpoint:",
            "best_loss": "Best Loss Checkpoint:",
            "last": "Last Checkpoint:"
        }
        self.entries = {}
        row = 0
        # Create entry fields and browse buttons for each checkpoint type.
        for key, text in self.labels.items():
            label = tk.Label(self, text=text, bg=BG_COLOR, fg=FG_COLOR)
            label.grid(row=row, column=0, padx=10, pady=5, sticky='e')
            entry = tk.Entry(self, width=50)
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[key] = entry
            btn = tk.Button(self, text="Browse", command=lambda k=key: self.browse_file(k))
            btn.grid(row=row, column=2, padx=10, pady=5)
            row += 1

        # Add radiobuttons to select which checkpoint to use.
        self.selected_checkpoint = tk.StringVar(value="best_acc")
        rb_frame = tk.Frame(self, bg=BG_COLOR)
        rb_frame.grid(row=row, column=0, columnspan=3, pady=10)
        tk.Label(rb_frame, text="Select checkpoint for testing:", bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w")
        tk.Radiobutton(rb_frame, text="Best Accuracy", variable=self.selected_checkpoint, value="best_acc",
                       bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(rb_frame, text="Best Loss", variable=self.selected_checkpoint, value="best_loss",
                       bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(rb_frame, text="Last", variable=self.selected_checkpoint, value="last",
                       bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=5)
        row += 1

        self.submit_button = tk.Button(self, text="Submit", command=self.on_submit)
        self.submit_button.grid(row=row, column=1, pady=20)
        self.paths = {"best_acc": "", "best_loss": "", "last": ""}

    def browse_file(self, key):
        file_path = filedialog.askopenfilename(
            title=f"Select {self.labels[key]}",
            filetypes=[("Checkpoint files", "*.ckpt"), ("All files", "*.*")]
        )
        if file_path:
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, file_path)
            self.paths[key] = file_path

    def on_submit(self):
        # Ensure that each entry has a provided path.
        for key, entry in self.entries.items():
            path = entry.get().strip()
            if not path:
                messagebox.showerror("Input Error", f"Please provide a path for {self.labels[key]}")
                return
            self.paths[key] = path
        selected_key = self.selected_checkpoint.get()
        # Pass only the selected checkpoint path to the callback.
        self.submit_callback(self.paths[selected_key])


if __name__ == "__main__":
    app = ModelTrainerApp()
    app.mainloop()
