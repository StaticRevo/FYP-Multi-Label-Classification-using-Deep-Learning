from models.resnet18.resnet18 import BigEarthNetResNet18Model
from config import ModelConfig, DatasetConfig
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import textwrap

def preprocess_image(image_path):
    transform = ModelConfig.test_transforms
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image_path):
    model.eval()
    image = preprocess_image(image_path).to(model.device)
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities

def get_actual_labels(image_path, metadata_csv):
    patch_id = os.path.basename(image_path).split('.')[0]
    actual_labels = metadata_csv[metadata_csv['patch_id'] == patch_id]['labels'].values[0]
    if isinstance(actual_labels, str):
        actual_labels = ast.literal_eval(actual_labels)  
    return actual_labels

def display_image_with_labels(image_path, predictions, class_labels, actual_labels):
    threshold = 0.5  
    predictions = predictions[0] 

    # Load and display the image
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # Prepare the labels
    predicted_labels = [label for i, label in enumerate(class_labels) if predictions[i] > threshold]

    # Wrap the labels to fit within the plot
    predicted_labels_str = ', '.join(predicted_labels)
    actual_labels_str = ', '.join(actual_labels)
    wrapped_predicted_labels = textwrap.fill(predicted_labels_str, width=80)
    wrapped_actual_labels = textwrap.fill(actual_labels_str, width=80)

    plt.subplots_adjust(top=0.8)

    # Display the labels
    plt.title(f"Predicted: {wrapped_predicted_labels}\nActual: {wrapped_actual_labels}", fontsize=12, color='black', loc='left', pad=20)
    plt.show()

def main():
    # Get the current directory
    current_dir = os.getcwd()

    # Load the trained model manually
    checkpoint_path = os.path.join(current_dir, 'FYPProject', 'experiments', 'checkpoints', 'best_model-ResNet18-epoch=08-val_loss=0.69.ckpt')

    # Load the model
    model = BigEarthNetResNet18Model.load_from_checkpoint(checkpoint_path)

    image_path = r'C:\Users\isaac\Desktop\BigEarthTests\Subsets\50%\CombinedRGBImagesJPG\S2A_MSIL2A_20170613T101031_N9999_R022_T34VER_01_69.jpg'  # Replace with the path to your image
    predictions = predict(model, image_path)
    print(f"Predictions for {image_path}: {predictions}")

    # Get actual labels from metadata
    actual_labels = get_actual_labels(image_path, DatasetConfig.metadata_csv)

    # Display image with labels
    display_image_with_labels(image_path, predictions, DatasetConfig.class_labels_dict.keys(), actual_labels)

if __name__ == "__main__":
    main()