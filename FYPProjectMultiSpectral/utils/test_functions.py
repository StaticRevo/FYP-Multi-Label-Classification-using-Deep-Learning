import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm
from utils.helper_functions import get_labels_for_image, display_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import random
import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

def collect_predictions_and_plot_confusion_matrix(all_preds, all_labels, DatasetConfig):
    # Compute multilabel confusion matrix
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    print("Multilabel Confusion Matrix:")
    print(mcm)

    # Aggregate confusion matrices into a single 19x19 matrix
    cm = np.zeros((DatasetConfig.num_classes, DatasetConfig.num_classes), dtype=int)
    for i in range(DatasetConfig.num_classes):
        cm[i, 0] = mcm[i, 0, 0]  # True Negative
        cm[i, 1] = mcm[i, 0, 1]  # False Positive
        cm[i, 2] = mcm[i, 1, 0]  # False Negative
        cm[i, 3] = mcm[i, 1, 1]  # True Positive

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(15, 12))  # Increase figure size for better visibility
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DatasetConfig.class_labels, yticklabels=DatasetConfig.class_labels)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0)  # Rotate y-axis labels for better visibility
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
    plt.show()

def apply_gradcam(model, input_tensor, target_layer, target_class):
    cam_extractor = GradCAM(model, target_layer) # Initialize Grad-CAM
    out = model(input_tensor) # Forward pass

    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out) # Extract CAM
    result = overlay_mask(to_pil_image(input_tensor.squeeze(0)), to_pil_image(activation_map[0], mode='F'), alpha=0.5)

    plt.imshow(result)
    plt.title(f'Grad-CAM for class {target_class}')
    plt.axis('off')
    plt.show()

def predict_and_display_random_image(model, dataset_dir, metadata_csv, bands):
    # Select a random image
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.tif')]
    random_image_file = random.choice(image_files)
    image_path = os.path.join(dataset_dir, random_image_file)

    # Load and preprocess the image
    with rasterio.open(image_path) as src:
        if max(bands) >= src.count:
            print(f"Requested band index exceeds the number of available bands ({src.count}).")
            return
        input_image = src.read(bands)
    input_tensor = torch.tensor(input_image).unsqueeze(0).float()  # Add batch dimension
    input_tensor = input_tensor / 255.0  # Normalize

    # Check number of channels
    if input_tensor.shape[1] < 4:
        print(f"Image has only {input_tensor.shape[1]} bands, cannot access bands 1, 2, 3.")
        return

    # Move the input tensor to the model's device
    input_tensor = input_tensor.to(model.device)

    # Predict labels
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    predicted_labels = (output.sigmoid() > 0.5).cpu().numpy().astype(int).squeeze()

    # Get true labels
    image_id = os.path.splitext(random_image_file)[0]
    print(f"Selected image ID: {image_id}")

    if image_id not in metadata_csv['patch_id'].values:
        print(f"Image ID {image_id} not found in metadata. Skipping.")
        return

    true_labels = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]

    # Display the image, true labels, and predicted labels
    plt.imshow(to_pil_image(input_tensor.cpu().squeeze(0)[[0, 1, 2]]))  
    plt.title(f"True Labels: {true_labels}\nPredicted Labels: {predicted_labels}")
    plt.axis('off')
    plt.show()
