import os
import subprocess

def choose_and_run_model():
    models = {
        '1': 'custom model',
        '2': 'ResNet18',
        '3': 'ResNet50',
        '4': 'VGG16',
        '5': 'VGG19',
        '6': 'DenseNet121',
        '7': 'EfficientNet',
        '8': 'EfficientNet_v2',
        '9': 'Swin-Transformer',
        '10': 'Vit-Transformer',
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

        # Run the trainer.py script with the selected model and weights
        script_path = os.path.join(os.getcwd(), 'FYPProject', 'trainer.py')
        print(f"Running {model_name} model with weights={weights}...")
        subprocess.run(['python', script_path, model_name, weights])

    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    choose_and_run_model()