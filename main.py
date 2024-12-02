import os
import subprocess
from FYPProjectMultiSpectral.config.config import DatasetConfig, ModelConfig

def choose_and_run_model():
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

    print("Please choose a model to run:")
    for key, model in models.items():
        print(f"{key}: {model}")

    choice = input("Enter the number corresponding to the model: ")
    if choice in models:
        model_name = models[choice]
        
        # Prompt for weights option
        print("Please choose the weights option:")
        print("1: None")
        print(f"2: {model_name}_Weights.DEFAULT")
        weights_choice = input("Enter the number corresponding to the weights option: ")

        if weights_choice == '1':
            weights = 'None'
        elif weights_choice == '2':
            weights = f'{model_name}_Weights.DEFAULT'
        else:
            print("Invalid choice. Please try again.")
            return

        # Prompt for band selection
        print("Please choose the band combination:")
        for key, bands in band_selection.items():
            print(f"{key}: {bands}")

        band_choice = input("Enter the number corresponding to the band combination: ")
        if band_choice in band_selection:
            selected_bands = band_selection[band_choice]
        else:
            print("Invalid choice. Please try again.")
            return

        # Prompt for dataset selection
        print("Please choose the dataset percentage:")
        for key, dataset in dataset_selection.items():
            print(f"{key}: {dataset}")

        dataset_choice = input("Enter the number corresponding to the dataset percentage: ")
        if dataset_choice in dataset_selection:
            selected_dataset = dataset_selection[dataset_choice]
        else:
            print("Invalid choice. Please try again.")
            return
        
        # Call train_tif.py with the selected choices
        subprocess.run(['python', 'FYPProjectMultiSpectral\\train_tif.py', model_name, weights, selected_bands, selected_dataset])
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    choose_and_run_model()

