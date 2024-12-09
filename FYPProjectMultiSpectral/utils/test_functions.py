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
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

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

def select_random_img(metadata_csv, split='test'):
    return random.choice(metadata_csv[metadata_csv['split'] == split]['patch_id'].apply(lambda x: f"{x}.tif").tolist())
    
def predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.6, bands=DatasetConfig.all_bands):
    # Create dictionaries for mapping between labels and indices
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict
          
    test_metadata = metadata_csv[metadata_csv['split'] == 'test']
    
    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }
    
    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]
    
    # Select a random image from the test split
    test_image_files = test_metadata['patch_id'].apply(lambda x: f"{x}.tif").tolist()
    random_image_file = random.choice(test_image_files)
    image_path = os.path.join(dataset_dir, random_image_file)
    print(f"Selected random image file: {random_image_file}")

    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read().astype(np.float32)
    
        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)
    
        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]
    
    # Stack the bands into an RGB image
    rgb = np.dstack((red, green, blue))
    
    # Read the specified bands for model input
    input_bands = np.stack([all_bands[band_indices[band]] for band in bands], axis=0)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_bands).unsqueeze(0).float()
    input_tensor = input_tensor.to(model.device)

    # Predict labels
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    sigmoid_outputs = output.sigmoid()
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()

    print("Sigmoid Outputs:", sigmoid_outputs)

    # Get true labels from the CSV
    image_id = os.path.splitext(random_image_file)[0]
    row = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
    true_labels = row if isinstance(row, list) else [row]

    # Convert true labels to indices
    true_labels_indices = [class_labels_dict[label] for label in true_labels]

    # Convert numeric predicted labels to text
    predicted_labels_indices = [idx for idx, value in enumerate(predicted_labels) if value == 1]

    # Print results
    print(f"Image ID: {image_id}")
    print(f"True Labels (Indices): {true_labels_indices}")
    print(f"Predicted Labels (Indices): {predicted_labels_indices}")

    # Display the image, true labels, and predicted labels
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(
        f"Image ID: {image_id}\n"
        f"True Labels: {true_labels_indices}\n"
        f"Predicted Labels: {predicted_labels_indices}"
    )
    plt.axis('off')
    plt.show()

def display_gradcam_heatmap(model, image_path, class_labels, selected_layer, threshold=0.55, bands=DatasetConfig.all_bands):
    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }
    
    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]

    # Load and preprocess the image
    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read().astype(np.float32)
    
        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)
    
        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]
    
    # Stack the bands into an RGB image
    rgb = np.dstack((red, green, blue))
    
    # Read the specified bands for model input
    input_bands = np.stack([all_bands[band_indices[band]] for band in bands], axis=0)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_bands).unsqueeze(0).float()
    input_tensor = input_tensor.to(model.device)

    # Set the model to evaluation mode
    model.eval()

    # Initialize GradCAM
    cam_extractor = GradCAM(model, target_layer=selected_layer)

    # Perform a forward pass through the model with gradients enabled
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

def plot_roc_auc(all_labels, all_preds, class_labels):
    # Binarize the labels for multi-label classification
    all_labels_bin = label_binarize(all_labels, classes=range(len(class_labels)))
    all_preds_bin = label_binarize(all_preds, classes=range(len(class_labels)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), all_preds_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(15, 10))
    for i in range(len(class_labels)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def predict_and_display_user_selected_image(model, image_path, metadata_csv, threshold=0.6, bands=DatasetConfig.all_bands):
    # Create dictionaries for mapping between labels and indices
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict

    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }

    # Define RGB band indices
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]

    with rasterio.open(image_path) as src:
        # Read all bands
        all_bands = src.read().astype(np.float32)

        # Normalize each band to the range 0-1
        all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)

        # Read the red, green, and blue bands specifically for display
        red = all_bands[rgb_band_indices[0]]
        green = all_bands[rgb_band_indices[1]]
        blue = all_bands[rgb_band_indices[2]]

    # Stack the bands into an RGB image
    rgb = np.dstack((red, green, blue))

    # Read the specified bands for model input
    input_bands = np.stack([all_bands[band_indices[band]] for band in bands], axis=0)

    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_bands).unsqueeze(0).float()
    input_tensor = input_tensor.to(model.device)

    # Predict labels
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    sigmoid_outputs = output.sigmoid()
    predicted_labels = (sigmoid_outputs > threshold).cpu().numpy().astype(int).squeeze()

    print("Sigmoid Outputs:", sigmoid_outputs)

    # Get true labels from the CSV
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    row = metadata_csv[metadata_csv['patch_id'] == image_id]['labels'].values[0]
    true_labels = row if isinstance(row, list) else [row]

    # Convert true labels to indices
    true_labels_indices = [class_labels_dict[label] for label in true_labels]

    # Convert numeric predicted labels to text
    predicted_labels_indices = [idx for idx, value in enumerate(predicted_labels) if value == 1]

    # Print results
    print(f"Image ID: {image_id}")
    print(f"True Labels (Indices): {true_labels_indices}")
    print(f"Predicted Labels (Indices): {predicted_labels_indices}")

    # Display the image, true labels, and predicted labels
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(
        f"Image ID: {image_id}\n"
        f"True Labels: {true_labels_indices}\n"
        f"Predicted Labels: {predicted_labels_indices}"
    )
    plt.axis('off')
    plt.show()


def predict_batch(model, dataloader, threshold=0.6, bands=DatasetConfig.all_bands):
    model.eval()
    all_preds = []
    all_true_labels = []
    class_labels_dict = DatasetConfig.class_labels_dict
    reversed_class_labels_dict = DatasetConfig.reversed_class_labels_dict

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs)
            sigmoid_outputs = outputs.sigmoid()
            preds = (sigmoid_outputs > threshold).cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_true_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_true_labels)

def display_batch_predictions(model, dataloader, threshold=0.6, bands=DatasetConfig.all_bands):
    all_preds, all_true_labels = predict_batch(model, dataloader, threshold, bands)

    for i, (pred, true) in enumerate(zip(all_preds, all_true_labels)):
        # Convert numeric predicted labels to text
        predicted_labels_indices = [idx for idx, value in enumerate(pred) if value == 1]
        true_labels_indices = [idx for idx, value in enumerate(true) if value == 1]

        # Display the image, true labels, and predicted labels
        plt.figure(figsize=(10, 10))
        image = dataloader.dataset[i][0]
        image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
        plt.imshow(image)
        plt.title(
            f"Image {i}\n"
            f"True Labels: {true_labels_indices}\n"
            f"Predicted Labels: {predicted_labels_indices}"
        )
        plt.axis('off')
        plt.show()