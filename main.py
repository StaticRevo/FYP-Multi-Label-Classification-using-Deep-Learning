import os

def choose_and_run_model():
    models = {
        '1': 'custom model',
        '2': 'resnet18',
        '3': 'resnet50',
        '4': 'vgg16'

    }
    print("Please choose a model to run:")
    for key, model in models.items():
        print(f"{key}: {model}")

    choice = input("Enter the number corresponding to the model: ")
    if choice in models:
        model_name = models[choice]
        model_path = os.path.join('FYPProject', 'my_models', model_name, f"{model_name}.py")
        print(f"Running {model_name} model from {model_path}...")
        os.system(f"python {model_path}")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    choose_and_run_model()