import subprocess
import tkinter as tk
import os
from tkinter import ttk, messagebox, Toplevel
from threading import Thread  

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class ModelSelectionGUI:
    # Initialize the GUI
    def __init__(self, master): 
        self.master = master
        self.master.title("Model Selection")
        self.master.geometry("450x1050")
        self.setup_style()
        self.setup_options()
        
        # Create a scrollable frame to hold all the widgets
        self.scroll_frame = ScrollableFrame(self.master)
        self.scroll_frame.pack(fill="both", expand=True)
        
        self.create_widgets()

    # Configure common styles for the widgets
    def setup_style(self):
        self.style = ttk.Style()
        self.style.configure("TLabel", padding=5, font=('Arial', 12))
        self.style.configure("TRadiobutton", padding=5, font=('Arial', 10))
        self.style.configure("TButton", padding=5, width=15, font=('Arial', 10, 'bold'))

    # Define Options for the model, band, and dataset selection
    def setup_options(self):
        # self.models = {
        #     '1': 'CustomModel',      # Custom
        #     '2': 'CustomWRNB4ECA'
        # }
        # self.models = {
        #     '1': 'VGG16',            # 2014
        #     '2': 'VGG19',            # 2014
        #     '3': 'ResNet18',         # 2015
        #     '4': 'ResNet50',         # 2015
        #     '5': 'ResNet101',        # 2015
        #     '6': 'DenseNet121',      # 2016
        #     '7': 'EfficientNetB0',   # 2019
        #     '8': 'EfficientNet_v2',  # 2021
        #     '9': 'CustomWRNB0',     # Custom (WideResNetB0) - Papoutsis et al.
        #     '10': 'CustomWRNB4ECA', # Custom (WideResNetB4ECA) - Papoutsis et al.
        #     '11': 'CustomModel',      # Custom
        #     '12': 'Swin-Transformer',# 2021
        #     '13': 'Vit-Transformer', # 2021
        # }

        self.models = {
            '1': 'CustomModelV1',
            '2': 'CustomModelV2',
            '3': 'CustomModelV3',
            '4': 'CustomModelV4',
            '5': 'CustomModelV5',
            '6': 'CustomModelV6',
            '7': 'CustomModelV7',
            '8': 'CustomModelV8'
        }
        
        self.band_selection = {
            '1': 'all_bands',
            '2': 'all_imp_bands',
            '3': 'rgb_bands',
            '4': 'rgb_nir_bands',
            '5': 'rgb_swir_bands',
            '6': 'rgb_nir_swir_bands'
        }
        self.dataset_selection = {
            '1': '0.5%_BigEarthNet ',
            '2': '1%_BigEarthNet',
            '3': '5%_BigEarthNet',
            '4': '10%_BigEarthNet',
            '5': '50%_BigEarthNet',
            '6': '100%_BigEarthNet'
        }

    # Create the widgets for the GUI
    def create_widgets(self):
        # All widgets are added to the scrollable frame instead of self.master
        container = self.scroll_frame.scrollable_frame
        self.create_model_selection_frame(container)
        self.create_weights_selection_frame(container)
        self.create_band_selection_frame(container)
        self.create_dataset_selection_frame(container)
        self.create_train_test_frame(container)
        self.create_iteration_options_frame(container)  
        self.create_action_buttons(container)

    # Create frame for model selection
    def create_model_selection_frame(self, container):
        self.model_frame = ttk.LabelFrame(container, text="Choose a model to run", padding="10")
        self.model_frame.grid(row=0, column=0, padx=20, pady=10, sticky='ew')
        self.model_var = tk.StringVar(value='1')
        for idx, (key, model_name) in enumerate(self.models.items(), start=1):
            ttk.Radiobutton(
                self.model_frame,
                text=model_name,
                variable=self.model_var,
                value=key
            ).grid(row=idx, column=0, sticky='w', padx=10)

    # Create frame for weights selection
    def create_weights_selection_frame(self, container):
        self.weights_frame = ttk.LabelFrame(container, text="Choose the weights option", padding="10")
        self.weights_frame.grid(row=1, column=0, padx=20, pady=10, sticky='ew')
        self.weights_var = tk.StringVar(value='1')
        ttk.Radiobutton(
            self.weights_frame,
            text="None",
            variable=self.weights_var,
            value='1'
        ).grid(row=1, column=0, sticky='w', padx=10)
        ttk.Radiobutton(
            self.weights_frame,
            text="DEFAULT",
            variable=self.weights_var,
            value='2'
        ).grid(row=2, column=0, sticky='w', padx=10)

    # Create frame for band selection
    def create_band_selection_frame(self, container):
        self.band_frame = ttk.LabelFrame(container, text="Choose the band combination", padding="10")
        self.band_frame.grid(row=0, column=1, padx=20, pady=10, sticky='ew')
        self.band_var = tk.StringVar(value='1')
        for idx, (key, bands) in enumerate(self.band_selection.items(), start=1):
            ttk.Radiobutton(
                self.band_frame,
                text=bands,
                variable=self.band_var,
                value=key
            ).grid(row=idx, column=0, sticky='w', padx=10)

    # Create frame for dataset selection
    def create_dataset_selection_frame(self, container):
        self.dataset_frame = ttk.LabelFrame(container, text="Choose the dataset percentage", padding="10")
        self.dataset_frame.grid(row=1, column=1, padx=20, pady=10, sticky='ew')
        self.dataset_var = tk.StringVar(value='1')
        for idx, (key, dataset) in enumerate(self.dataset_selection.items(), start=1):
            ttk.Radiobutton(
                self.dataset_frame,
                text=dataset,
                variable=self.dataset_var,
                value=key
            ).grid(row=idx, column=0, sticky='w', padx=10)

    # Create frame for train/test selection
    def create_train_test_frame(self, container):
        self.train_test_frame = ttk.LabelFrame(container, text="Choose to Train or Train and Test", padding="10")
        self.train_test_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
        self.train_test_var = tk.StringVar(value='train')
        ttk.Radiobutton(
            self.train_test_frame,
            text="Train Only",
            variable=self.train_test_var,
            value='train'
        ).grid(row=1, column=0, sticky='w', padx=10)
        ttk.Radiobutton(
            self.train_test_frame,
            text="Train and Test",
            variable=self.train_test_var,
            value='train_test'
        ).grid(row=2, column=0, sticky='w', padx=10)
        ttk.Radiobutton(
            self.train_test_frame,
            text="Test Only",
            variable=self.train_test_var,
            value='test'
        ).grid(row=3, column=0, sticky='w', padx=10)

    # Create iteration options frame for running all models/bands/weights
    def create_iteration_options_frame(self, container):
        self.iteration_frame = ttk.LabelFrame(container, text="Iteration Options", padding="10")
        self.iteration_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
        
        self.all_models_var = tk.BooleanVar(value=False)
        self.all_bands_var = tk.BooleanVar(value=False)
        self.both_weights_var = tk.BooleanVar(value=False)
        self.all_datasets_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(
            self.iteration_frame,
            text="Run All Models",
            variable=self.all_models_var
        ).grid(row=1, column=0, sticky='w', padx=10)
        
        ttk.Checkbutton(
            self.iteration_frame,
            text="Run All Band Combinations",
            variable=self.all_bands_var
        ).grid(row=2, column=0, sticky='w', padx=10)
        
        ttk.Checkbutton(
            self.iteration_frame,
            text="Run Both Weights (None & Default)",
            variable=self.both_weights_var
        ).grid(row=3, column=0, sticky='w', padx=10)

        ttk.Checkbutton(
            self.iteration_frame,
            text="Run All Dataset Percentages",
            variable=self.all_datasets_var
        ).grid(row=4, column=0, sticky='w', padx=10)

    # Create Run and Reset buttons
    def create_action_buttons(self, container):
        self.run_button = ttk.Button(container, text="Run Model", command=self.run_model)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.reset_button = ttk.Button(container, text="Reset", command=self.reset_selections)
        self.reset_button.grid(row=5, column=0, columnspan=2, pady=10)

    # Gather selections, show a loading dialog and run the model(s) in a separate thread
    def run_model(self):
        # Determine which models to run
        if self.all_models_var.get():
            models_to_run = list(self.models.values())
        else:
            models_to_run = [self.models.get(self.model_var.get())]

        # Determine which band combinations to run
        if self.all_bands_var.get():
            bands_to_run = list(self.band_selection.values())
        else:
            bands_to_run = [self.band_selection.get(self.band_var.get())]

        # Determine which weights options to run
        if self.both_weights_var.get():
            use_both_weights = True
        else:
            weights_choice = self.weights_var.get() 
            selected_model = self.models.get(self.model_var.get())  
            weights_to_run = ['None'] if weights_choice == '1' else [f'{selected_model}_Weights.DEFAULT']
            use_both_weights = False

        # Determine which dataset percentages to run
        if self.all_datasets_var.get():
            selected_datasets = list(self.dataset_selection.values())
        else:
            dataset_choice = self.dataset_var.get()
            selected_datasets = [self.dataset_selection.get(dataset_choice)]
            
        train_test_choice = self.train_test_var.get()
        test_flag = 'True' if train_test_choice == 'train_test' else 'False'

        # Create a loading dialog
        loading_window = Toplevel(self.master)
        loading_window.title("Running Model(s)")
        loading_label = ttk.Label(
            loading_window,
            text="Running training, please wait...",
            padding="20",
            font=('Arial', 12)
        )
        loading_label.pack(padx=20, pady=20)
        progress = ttk.Progressbar(loading_window, orient="horizontal", length=300, mode="indeterminate")
        progress.pack(padx=20, pady=20)
        progress.start()
        loading_window.grab_set()

        # Run the model training/testing in a separate thread
        def model_training_thread():
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                trainer_script = os.path.join(script_dir, 'FYPProjectMultiSpectral', 'trainer.py')
                tester_script = os.path.join(script_dir, 'FYPProjectMultiSpectral', 'tester_runner.py')
                for dataset in selected_datasets:
                    if train_test_choice == 'test':
                        for model_name in models_to_run:
                            for selected_bands in bands_to_run:
                                if use_both_weights:
                                    current_weights_options = ['None', f'{model_name}_Weights.DEFAULT']
                                else:
                                    current_weights_options = weights_to_run
                                for weights in current_weights_options:
                                    print(f"Testing: model={model_name}, weights={weights}, bands={selected_bands}, dataset={dataset}")
                                    subprocess.run(['python', tester_script, model_name, weights, selected_bands, dataset])
                    else:
                        for model_name in models_to_run: 
                            if use_both_weights:
                                current_weights_options = ['None', f'{model_name}_Weights.DEFAULT']
                            else:
                                current_weights_options = weights_to_run
                            for selected_bands in bands_to_run:
                                for weights in current_weights_options:
                                    print(f"Running: model={model_name}, weights={weights}, bands={selected_bands}, dataset={dataset}")
                                    subprocess.run(['python', trainer_script, model_name, weights, selected_bands, dataset, test_flag])
                        
                messagebox.showinfo("Success", "Automatic training completed for selected combinations.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                loading_window.destroy()

        thread = Thread(target=model_training_thread)
        thread.start()

    # Reset all selection variables to their defaults
    def reset_selections(self):
        self.model_var.set('1')
        self.weights_var.set('1')
        self.band_var.set('1')
        self.dataset_var.set('1')
        self.train_test_var.set('train')
        self.all_models_var.set(False)
        self.all_bands_var.set(False)
        self.both_weights_var.set(False)
        self.all_datasets_var.set(False)

def main():
    root = tk.Tk()
    app = ModelSelectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
