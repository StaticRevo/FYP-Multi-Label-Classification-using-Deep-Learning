import os
import subprocess

def choose_and_run_model():
    models = {
        '1': 'custom model',
        '2': 'resnet18',
        '3': 'resnet50',
        '4': 'vgg16',
        '5': 'vgg19',
        '5': 'densenet121',
        '6': 'efficientnet',
        '7': 'efficientnet_v2',
        '8': 'swin-transformer',
        '9': 'vit-transformer',
    }
    print("Please choose a model to run:")
    for key, model in models.items():
        print(f"{key}: {model}")

    choice = input("Enter the number corresponding to the model: ")
    if choice in models:
        model_name = models[choice]
        if model_name == 'resnet18':
            # Run the trainer.py script for resnet18
            script_path = os.path.join(os.getcwd(), 'FYPProject', 'trainer.py')
            print(script_path)
            subprocess.run(['python', script_path])
        else:
            print(f"Running {model_name} model...")
            # Add logic to run other models if needed
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    choose_and_run_model()