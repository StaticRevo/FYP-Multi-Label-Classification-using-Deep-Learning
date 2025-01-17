import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import Toplevel
from threading import Thread  

# Function to choose and run the model
def choose_and_run_model_gui():
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

    # Create main window
    root = tk.Tk()
    root.title("Model Selection")

    # Create a style
    style = ttk.Style()
    style.configure("TLabel", padding=5, font=('Arial', 12))
    style.configure("TRadiobutton", padding=5, font=('Arial', 10))
    style.configure("TButton", padding=5, width=15, font=('Arial', 10, 'bold'))

    # Set minimum size for the window
    root.geometry("450x750")
    
    # Model selection
    model_frame = ttk.LabelFrame(root, text="Choose a model to run", padding="10")
    model_frame.grid(row=0, column=0, padx=20, pady=10, sticky='ew')
    model_var = tk.StringVar(value='1')
    for idx, (key, model) in enumerate(models.items(), start=1):
        ttk.Radiobutton(model_frame, text=model, variable=model_var, value=key).grid(row=idx, column=0, sticky='w', padx=10)

    # Weights selection
    weights_frame = ttk.LabelFrame(root, text="Choose the weights option", padding="10")
    weights_frame.grid(row=1, column=0, padx=20, pady=10, sticky='ew')
    weights_var = tk.StringVar(value='1')
    ttk.Radiobutton(weights_frame, text="None", variable=weights_var, value='1').grid(row=1, column=0, sticky='w', padx=10)
    ttk.Radiobutton(weights_frame, text="DEFAULT", variable=weights_var, value='2').grid(row=2, column=0, sticky='w', padx=10)

    # Band selection
    band_frame = ttk.LabelFrame(root, text="Choose the band combination", padding="10")
    band_frame.grid(row=0, column=1, padx=20, pady=10, sticky='ew')
    band_var = tk.StringVar(value='1')
    for idx, (key, bands) in enumerate(band_selection.items(), start=1):
        ttk.Radiobutton(band_frame, text=bands, variable=band_var, value=key).grid(row=idx, column=0, sticky='w', padx=10)

    # Dataset selection
    dataset_frame = ttk.LabelFrame(root, text="Choose the dataset percentage", padding="10")
    dataset_frame.grid(row=1, column=1, padx=20, pady=10, sticky='ew')
    dataset_var = tk.StringVar(value='1')
    for idx, (key, dataset) in enumerate(dataset_selection.items(), start=1):
        ttk.Radiobutton(dataset_frame, text=dataset, variable=dataset_var, value=key).grid(row=idx, column=0, sticky='w', padx=10)

    # Train/Test selection
    train_test_frame = ttk.LabelFrame(root, text="Choose to Train or Train and Test", padding="10")
    train_test_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky='ew')
    train_test_var = tk.StringVar(value='train')
    ttk.Radiobutton(train_test_frame, text="Train Only", variable=train_test_var, value='train').grid(row=1, column=0, sticky='w', padx=10)
    ttk.Radiobutton(train_test_frame, text="Train and Test", variable=train_test_var, value='train_test').grid(row=2, column=0, sticky='w', padx=10)
    ttk.Radiobutton(train_test_frame, text="Test Only", variable=train_test_var, value='test').grid(row=3, column=0, sticky='w', padx=10)

    # Function to run the model
    def run_model():
        model_choice = model_var.get()
        model_name = models.get(model_choice)
        
        weights_choice = weights_var.get()
        weights = 'None' if weights_choice == '1' else f'{model_name}_Weights.DEFAULT'

        band_choice = band_var.get()
        selected_bands = band_selection.get(band_choice)

        dataset_choice = dataset_var.get()
        selected_dataset = dataset_selection.get(dataset_choice)

        train_test_choice = train_test_var.get()
        test = 'True' if train_test_choice == 'train_test' else 'False'

        # Show loading dialog with progress bar
        loading_window = Toplevel(root)
        loading_window.title("Running Model")
        loading_label = ttk.Label(loading_window, text="Running the model, please wait...", padding="20", font=('Arial', 12))
        loading_label.pack(padx=20, pady=20)

        # Create a progress bar
        progress = ttk.Progressbar(loading_window, orient="horizontal", length=300, mode="indeterminate")
        progress.pack(padx=20, pady=20)
        progress.start()

        loading_window.grab_set()

        def model_training_thread():
            try:
                if train_test_choice == 'test':
                    subprocess.run(['python', 'FYPProjectMultiSpectral\\tester_runner.py', model_name, weights, selected_bands, selected_dataset])
                    messagebox.showinfo("Success", f"Testing {model_name} with {selected_bands} and {selected_dataset} dataset.")
                else:
                    subprocess.run(['python', 'FYPProjectMultiSpectral\\trainer.py', model_name, weights, selected_bands, selected_dataset, test])
                    messagebox.showinfo("Success", f"Model {model_name} with {selected_bands} and {selected_dataset} dataset is running.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                loading_window.destroy()  

        # Run the model training in a separate thread
        thread = Thread(target=model_training_thread)
        thread.start()

    # Function to reset the selections
    def reset_selections():
        model_var.set('1')
        weights_var.set('1')
        band_var.set('1')
        dataset_var.set('1')
        train_test_var.set('train')

    # Run button
    run_button = ttk.Button(root, text="Run Model", command=run_model)
    run_button.grid(row=3, column=0, columnspan=2, pady=20)

    # Reset button
    reset_button = ttk.Button(root, text="Reset", command=reset_selections)
    reset_button.grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    choose_and_run_model_gui()