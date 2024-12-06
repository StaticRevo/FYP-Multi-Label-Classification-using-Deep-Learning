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
from utils.helper_functions import decode_target
import random
import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from config.config import DatasetConfig
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.4,
    cls_labels: dict = None,
):
    result = []
    for i, x in enumerate(target):
        if x >= threshold:
            if text_labels:
                result.append(cls_labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)

def encode_label(label: list, num_classes=DatasetConfig.num_classes):
    target = torch.zeros(num_classes)
    for l in label:
        if l in DatasetConfig.class_labels_dict:
            target[DatasetConfig.class_labels_dict[l]] = 1.0
    return target

def plot_confusion_matrix(all_preds, all_labels, DatasetConfig):
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

def plot_normalized_confusion_matrix(all_preds, all_labels, DatasetConfig):
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

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix:")
    print(cm_normalized)

    # Plot normalized confusion matrix
    plt.figure(figsize=(15, 12))  # Increase figure size for better visibility
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=DatasetConfig.class_labels, yticklabels=DatasetConfig.class_labels)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.yticks(rotation=0)  # Rotate y-axis labels for better visibility
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
    plt.show()
    
def predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.6, class_labels=None):
    # Filter metadata to include only 'test' split
    test_metadata = metadata_csv[metadata_csv['split'] == 'test']
    
    # Select a random image from the test split
    test_image_files = test_metadata['patch_id'].apply(lambda x: f"{x}.tif").tolist()
    random_image_file = random.choice(test_image_files)
    image_path = os.path.join(dataset_dir, random_image_file)
    print(f"Selected random image file: {random_image_file}")

    with rasterio.open(image_path) as src:
        # Read the red, green, and blue bands
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Normalize each band to the range 0-1
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        blue = blue.astype(np.float32)
        
        red /= np.max(red)
        green /= np.max(green)
        blue /= np.max(blue)
        
        # Stack the bands into an RGB image
        rgb = np.dstack((red, green, blue))
    
    # Convert to tensor and add batch dimension
    input_image = np.stack([red, green, blue], axis=0)  # Stack bands into a single array
    input_tensor = torch.tensor(input_image).unsqueeze(0).float()
    print(f"Input tensor shape: {input_tensor.shape}")

    # Move the input tensor to the model's device
    input_tensor = input_tensor.to(model.device)
    print(f"Moved input tensor to device: {model.device}")

    # Predict labels
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Raw logits: {output}")
    sigmoid_outputs = output.sigmoid()
    print(f"Sigmoid outputs: {sigmoid_outputs}")
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()
    print(f"Predicted labels: {predicted_labels}")

    # Convert predicted labels to text
    predicted_labels_text = decode_target(predicted_labels, text_labels=True, cls_labels=class_labels)
    print(f"Predicted labels (text): {predicted_labels_text}")

    # Get true labels
    image_id = os.path.splitext(random_image_file)[0]
    print(f"Selected image ID: {image_id}")

    if image_id not in metadata_csv['patch_id'].values:
        print(f"Image ID {image_id} not found in metadata. Skipping.")
        return

    true_labels = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
    print(f"True labels: {true_labels}")

    # Display the image, true labels, and predicted labels
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(f"True Labels: {true_labels}\nPredicted Labels: {predicted_labels_text}")
    plt.axis('off')
    plt.show()

def predict_and_display_multiple_images(model, dataset_dir, metadata_csv, num_images=5, threshold=0.6, class_labels=None):
    # Filter metadata to include only 'test' split
    test_metadata = metadata_csv[metadata_csv['split'] == 'test']
    
    # Select random images from the test split
    test_image_files = test_metadata['patch_id'].apply(lambda x: f"{x}.tif").tolist()
    random_image_files = random.sample(test_image_files, num_images)
    
    # Create a grid for displaying images
    fig, axes = plt.subplots(nrows=num_images // 2 + num_images % 2, ncols=2, figsize=(15, num_images * 5))
    axes = axes.flatten()

    for idx, random_image_file in enumerate(random_image_files):
        image_path = os.path.join(dataset_dir, random_image_file)
        print(f"Selected random image file: {random_image_file}")

        with rasterio.open(image_path) as src:
            # Read the red, green, and blue bands
            red = src.read(4)
            green = src.read(3)
            blue = src.read(2)
            
            # Normalize each band to the range 0-1
            red = red.astype(np.float32)
            green = green.astype(np.float32)
            blue = blue.astype(np.float32)
            
            red /= np.max(red)
            green /= np.max(green)
            blue /= np.max(blue)
            
            # Stack the bands into an RGB image
            rgb = np.dstack((red, green, blue))
        
        # Convert to tensor and add batch dimension
        input_image = np.stack([red, green, blue], axis=0)  # Stack bands into a single array
        input_tensor = torch.tensor(input_image).unsqueeze(0).float()
        print(f"Input tensor shape: {input_tensor.shape}")

        # Move the input tensor to the model's device
        input_tensor = input_tensor.to(model.device)
        print(f"Moved input tensor to device: {model.device}")

        # Predict labels
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Raw logits: {output}")
        sigmoid_outputs = output.sigmoid()
        print(f"Sigmoid outputs: {sigmoid_outputs}")
        predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()
        print(f"Predicted labels: {predicted_labels}")

        # Convert predicted labels to text
        predicted_labels_text = decode_target(predicted_labels, text_labels=True, cls_labels=class_labels)
        print(f"Predicted labels (text): {predicted_labels_text}")

        # Get true labels
        image_id = os.path.splitext(random_image_file)[0]
        print(f"Selected image ID: {image_id}")

        if image_id not in metadata_csv['patch_id'].values:
            print(f"Image ID {image_id} not found in metadata. Skipping.")
            continue

        true_labels = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
        print(f"True labels: {true_labels}")

        # Display the image
        axes[idx].imshow(rgb)
        axes[idx].axis('off')

        # Add text for true and predicted labels
        axes[idx].text(0.5, 1.05, f"True Labels: {true_labels}", ha='center', va='center', transform=axes[idx].transAxes, fontsize=12)
        axes[idx].text(0.5, 1.15, f"Predicted Labels: {predicted_labels_text}", ha='center', va='center', transform=axes[idx].transAxes, fontsize=12)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)
    plt.show()

def display_gradcam_heatmap(model, image_path, class_labels, selected_layer, threshold=0.55):
    # Load and preprocess the image
    with rasterio.open(image_path) as src:
        red = src.read(4).astype(np.float32)
        green = src.read(3).astype(np.float32)
        blue = src.read(2).astype(np.float32)

        red /= np.max(red)
        green /= np.max(green)
        blue /= np.max(blue)

        rgb = np.dstack((red, green, blue))
        input_image = np.stack([red, green, blue], axis=0)
        input_tensor = torch.tensor(input_image).unsqueeze(0).float()

    # Move the input tensor to the model's device
    input_tensor = input_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize GradCAM
    cam_extractor = GradCAM(model, target_layer=selected_layer)

    # Perform a forward pass through the model
    with torch.no_grad():  # Disable gradients for efficiency during forward pass
        output = model(input_tensor)

    # Get the target class
    sigmoid_outputs = output.sigmoid()
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()
    target_class = int(np.argmax(predicted_labels))
    print(f"Target class for GradCAM: {class_labels[target_class]} ({target_class})")

    # Extract the activation map for the target class
    activation_map = cam_extractor(target_class, output)

    # Overlay the activation map on the image
    result = overlay_mask(to_pil_image(rgb), to_pil_image(activation_map[0].cpu(), mode='F'), alpha=0.5)

    # Display the result
    plt.imshow(result)
    plt.title(f"GradCAM Heatmap for Class: {class_labels[target_class]}")
    plt.axis('off')
    plt.show()

