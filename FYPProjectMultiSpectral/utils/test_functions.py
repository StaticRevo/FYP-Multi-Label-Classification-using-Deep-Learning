import os
from config.config import DatasetConfig, ModelConfig
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.models import *
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, hamming_loss, accuracy_score, multilabel_confusion_matrix
import math
import random
import rasterio
from utils.gradcam import GradCAM, overlay_heatmap
from config.config_utils import calculate_class_weights

def calculate_metrics_and_save_results(model, data_module, model_name, dataset_name, class_labels, result_path):
    all_preds, all_labels = [], []
    test_loader = data_module.test_dataloader()

    # Iterate through batches
    for batch in tqdm(test_loader, desc="Processing Batches"):
        inputs, labels = batch
        inputs, labels = inputs.to(model.device), labels.to(model.device)

        # Generate predictions
        with torch.no_grad():
            logits = model(inputs)
            preds = torch.sigmoid(logits) > 0.5

        all_preds.extend(preds.cpu().numpy().astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))

    # Convert lists to numpy arrays
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    # Save predictions and labels
    save_path = os.path.join(result_path, f'test_predictions_{model_name}_{dataset_name}.npz')
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    return all_preds, all_labels

def visualize_predictions_and_heatmaps(model, data_module, predictions, true_labels, class_labels, model_name):
    # Display batch predictions
    display_batch_predictions(
        model, data_module.test_dataloader(), threshold=0.6, bands=DatasetConfig.all_bands
    )

    # Plot per-label confusion matrices
    plot_per_label_confusion_matrices_grid(
        true_labels, predictions, class_names=class_labels
    )

    # Compute and print aggregated metrics
    scores = compute_aggregated_metrics(true_labels, predictions)
    print("\nAggregated Metrics:\n", scores)

    # Plot co-occurrence matrix
    plot_cooccurrence_matrix(true_labels, predictions, class_names=class_labels)

def generate_gradcam_visualizations(model, data_module, class_labels, model_name, result_path):
    gradcam_save_dir = os.path.join(result_path, 'gradcam_visualizations')
    os.makedirs(gradcam_save_dir, exist_ok=True)

    if model_name == 'ResNet18':
        target_layer = model.model.layer3[-1].conv2
    elif model_name == 'ResNet50':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'VGG16':
        target_layer = model.model.features[28]
    elif model_name == 'VGG19':
        target_layer = model.model.features[34]
    elif model_name == 'EfficientNetB0':
        target_layer = model.model.features[8][0]
    elif model_name == 'EfficientNet_v2':
        target_layer = model.modelfeatures[7][4].block[3]
    elif model_name == 'Swin-Transformer':
        target_layer = model.model.stages[3].blocks[-1].norm1
    elif model_name == 'Vit-Transformer':
        target_layer = model.model.layers[-1].attention

    grad_cam = GradCAM(model, target_layer)

    test_dataset = data_module.test_dataloader().dataset
    num_images = len(test_dataset)
    target_indices = [random.randint(0, num_images - 1) for _ in range(5)]  # Select 5 random images

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(model.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(model.device)

    for idx in target_indices:
        try:
            # Retrieve the image and label from the Dataset
            img_tensor, label = test_dataset[idx]
        except IndexError:
            print(f"Index {idx} is out of bounds for the test dataset.")
            continue

        input_image = img_tensor.unsqueeze(0).to(model.device)  

        # Forward pass to get predictions
        output = model(input_image)

        # Get relevant classes for multi-label classification
        threshold = 0.5  
        target_classes = torch.where(output[0] > threshold)[0].tolist()

        # Generate heatmaps for each relevant class
        heatmaps = {}
        for target_class in target_classes:
            cam, _ = grad_cam.generate_heatmap(input_image, target_class=target_class)
            heatmaps[class_labels[target_class]] = cam

        # Convert the input tensor to a PIL image for visualization
        img = input_image.squeeze()  # Remove batch dimension
        rgb_channels = [3, 2, 1]  
        img = img[rgb_channels, :, :] 

        # Normalize each channel
        img_cpu = img.detach().cpu().numpy()
        red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
        green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
        blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)

        # Stack into an RGB image
        rgb_image = np.stack([red, green, blue], axis=-1)

        # Convert to PIL Image
        img = Image.fromarray((rgb_image * 255).astype(np.uint8))

        # Display and save heatmaps for each class
        for class_name, heatmap in heatmaps.items():
            # Overlay heatmap on image
            overlay = overlay_heatmap(img, heatmap, alpha=0.5)

            # Plot the results
            plt.figure(figsize=(15, 5))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(img)
            plt.axis('off')

            # Grad-CAM Heatmap
            plt.subplot(1, 3, 2)
            plt.title(f'Heatmap for Class: {class_name}')
            plt.imshow(heatmap, cmap='jet')
            plt.axis('off')

            # Overlayed Heatmap
            plt.subplot(1, 3, 3)
            plt.title(f'Overlay for Class: {class_name}')
            plt.imshow(overlay)
            plt.axis('off')

            # Save and display the visualization
            plt.suptitle(f'Image Index: {idx} | Class: {class_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_save_dir, f'gradcam_{idx}_{class_name}.png'))
            plt.show()

           
def predict_and_display_random_image(model, dataset_dir, metadata_csv, threshold=0.6, bands=DatasetConfig.rgb_bands):
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

    print(f"Image ID: {image_id}")
    print(f"True Labels (Indices): {true_labels_indices}")
    print(f"Predicted Labels (Indices): {predicted_labels_indices}")

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(
        f"Image ID: {image_id}\n"
        f"True Labels: {true_labels_indices}\n"
        f"Predicted Labels: {predicted_labels_indices}"
    )
    plt.axis('off')
    plt.show()

def plot_roc_auc(all_labels, all_probs, class_labels):
    num_classes = all_labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute the ROC for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        plt.plot(
            fpr[i], tpr[i],
            lw=2,
            label=f'Class {class_labels[i]} (area = {roc_auc[i]:0.2f})'
        )

    # Plot micro-average
    plt.plot(
        fpr["micro"], tpr["micro"],
        color='deeppink', linestyle=':', linewidth=4,
        label=f'Micro-average (area = {roc_auc["micro"]:0.2f})'
    )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-label ROC Curve')
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

def display_batch_predictions(model, dataloader, threshold=0.6, bands=DatasetConfig.all_bands, num_images=10):
    all_preds, all_true_labels = predict_batch(model, dataloader, threshold, bands)
    
    dataset_size = len(dataloader.dataset)
    num_images = min(num_images, dataset_size)  # Ensure we don't exceed the dataset size
    
    # Randomly select unique indices
    random_indices = random.sample(range(dataset_size), num_images)
    
    # Map band names to indices
    band_indices = {"B02": 1, "B03": 2, "B04": 3}  
    rgb_band_indices = [band_indices["B04"], band_indices["B03"], band_indices["B02"]]  # Red, Green, Blue
    
    for i in random_indices:
        pred = all_preds[i]
        true = all_true_labels[i]
        
        # Convert numeric predicted labels to text
        predicted_labels_indices = [idx for idx, value in enumerate(pred) if value == 1]
        true_labels_indices = [idx for idx, value in enumerate(true) if value == 1]
    
        # Get the image and select RGB bands for visualization
        image_tensor = dataloader.dataset[i][0]  # Assuming the first element is the image
        image_rgb = image_tensor[rgb_band_indices, :, :]  # Select RGB bands
        image_rgb = image_rgb.permute(1, 2, 0).numpy()  
    
        # Normalize RGB bands for visualization
        image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-8)
    
        # Display the image, true labels, and predicted labels
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(
            f"Image {i}\n"
            f"True Labels: {true_labels_indices}\n"
            f"Predicted Labels: {predicted_labels_indices}"
        )
        plt.axis('off')
        plt.show()


def get_sigmoid_outputs(model, dataset_dir, metadata_csv, bands=DatasetConfig.rgb_bands):
    # Create dictionaries for mapping between labels and indices
    test_metadata = metadata_csv[metadata_csv['split'] == 'test']
    
    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }
    
    sigmoid_outputs_list = []

    # Process only the first 10 test images
    for image_file in tqdm(test_metadata['patch_id'].iloc[:10].apply(lambda x: f"{x}.tif").tolist(), desc="Processing Images"):
        image_path = os.path.join(dataset_dir, image_file)
        with rasterio.open(image_path) as src:
            # Read all bands
            all_bands = src.read().astype(np.float32)
        
            # Normalize each band to the range 0-1
            all_bands /= np.max(all_bands, axis=(1, 2), keepdims=True)
        
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
        sigmoid_outputs_list.append(sigmoid_outputs.cpu().numpy().squeeze())

    return np.array(sigmoid_outputs_list)

def plot_per_label_confusion_matrices_grid(all_labels, all_preds, class_names=None, cols=4):
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    n_labels = len(mcm)

    # Determine how many rows we need
    rows = math.ceil(n_labels / cols)

    # Create a figure with (rows x cols) subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    # If there's only 1 row, axes is not a 2D array; make it a list for consistency
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
    axes = axes.flatten()  # flatten in case we have multiple rows

    for i, matrix in enumerate(mcm):
        # Flatten the 2x2 matrix into TN, FP, FN, TP
        tn, fp, fn, tp = matrix.ravel()
        label_name = class_names[i] if class_names else f"Label {i}"

        # Plot a heatmap for this label's 2x2 matrix on the i-th subplot
        ax = axes[i]
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'],
            ax=ax
        )
        ax.set_title(f'{label_name}\n(TN={tn}, FP={fp}, FN={fn}, TP={tp})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  
    plt.show()

def compute_aggregated_metrics(all_labels, all_preds):
    metrics_dict = {}
    
    # Micro-average: aggregates the contributions of all classes to compute the average metric
    metrics_dict['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    # Macro-average: computes metric independently for each class and then takes the average
    metrics_dict['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Hamming loss: fraction of labels incorrectly predicted
    metrics_dict['hamming_loss'] = hamming_loss(all_labels, all_preds)

    # Subset accuracy: only 1 if *all* labels match exactly
    metrics_dict['subset_accuracy'] = accuracy_score(all_labels, all_preds)

    return metrics_dict

def plot_cooccurrence_matrix(all_labels, all_preds, class_names=None):
    num_classes = all_labels.shape[1]
    cooccur = np.zeros((num_classes, num_classes), dtype=int)

    # For each sample
    for n in range(all_labels.shape[0]):
        # find all true labels
        true_idxs = np.where(all_labels[n] == 1)[0]
        # find all predicted labels
        pred_idxs = np.where(all_preds[n] == 1)[0]
        # increment co-occurrences
        for i in true_idxs:
            for j in pred_idxs:
                cooccur[i, j] += 1

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cooccur,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(num_classes),
        yticklabels=class_names if class_names else range(num_classes),
        cbar_kws={'shrink': 0.75}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Multi-label Co-occurrence Matrix", fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2.0)
    plt.show()

    return cooccur