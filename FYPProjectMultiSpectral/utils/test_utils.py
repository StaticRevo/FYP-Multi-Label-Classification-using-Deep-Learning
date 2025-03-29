# Standard library imports
import os
import math
import random

# Third-party imports
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, f1_score, 
    hamming_loss, accuracy_score, multilabel_confusion_matrix,
    fbeta_score, average_precision_score
)
import rasterio
from flask import url_for

# Local application imports
from config.config import DatasetConfig
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap
from utils.logging_utils import setup_logger

# Calculate metrics and save results
def calculate_metrics_and_save_results(model, data_module, model_name, dataset_name, class_labels, result_path, logger=None):
    all_preds, all_labels = [], []
    test_loader = data_module.test_dataloader()

    logger.info("Starting batch processing for metrics calculation.")
    for batch in tqdm(test_loader, desc="Processing Batches"): # Iterate through batches
        inputs, labels = batch
        inputs, labels = inputs.to(model.device), labels.to(model.device)

        with torch.no_grad(): # Generate predictions
            logits = model(inputs)
            preds = torch.sigmoid(logits) > 0.5

        all_preds.extend(preds.cpu().numpy().astype(int))
        all_labels.extend(labels.cpu().numpy().astype(int))

    all_preds, all_labels = np.array(all_preds), np.array(all_labels) # Convert lists to numpy arrays

    # Save predictions and labels
    save_path = os.path.join(result_path, f'test_predictions_{model_name}_{dataset_name}.npz')
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    logger.info(f"Predictions and labels saved to {save_path}")
    return all_preds, all_labels

# Visualize predictions and heatmaps
def visualize_predictions_and_heatmaps(model, data_module, in_channels, predictions, true_labels, class_labels, model_name, result_path, probs=None, logger=None):
    save_dir = os.path.join(result_path, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    saving_batch_predictions( # Display batch predictions
         model, data_module.test_dataloader(), in_channels, threshold=0.6, bands=DatasetConfig.all_bands, num_images=10, save_dir=save_dir, logger=logger
    )
    plot_per_label_confusion_matrices_grid( # Plot per-label confusion matrices
        true_labels, predictions, class_names=class_labels, cols=4, save_dir=save_dir, logger=logger
    )
    scores = compute_aggregated_metrics(true_labels, predictions, probs, logger) # Compute and print aggregated metrics
    print(f"Aggregated Metrics:\n{scores}")
    # Save the aggregated metrics to a text file within save_dir
    aggregated_metrics_path = os.path.join(result_path, "aggregated_metrics.txt")
    with open(aggregated_metrics_path, "w") as f:
        f.write("Aggregated Metrics:\n")
        for metric, value in scores.items():
             f.write(f"{metric}: {value}\n")
    print(f"Aggregated metrics saved to {aggregated_metrics_path}")

    plot_cooccurrence_matrix(true_labels, predictions, class_names=class_labels, save_dir=save_dir, logger=logger) # Plot co-occurrence matrix

    if probs is not None:
        plot_roc_auc(true_labels, probs, class_labels,  save_dir=save_dir, logger=logger) # Plot ROC-AUC curve
    else:
        logger.warning("Continuous probability outputs not provided. Skipping ROC-AUC plotting.")

# Get the Grad-CAM target layer
def get_target_layer(model, model_name):
    target_layer = None
    
    # Determine target layer based on model_name
    if 'CustomModel' in model_name:
        target_layer = model.block4[2].conv2  
    elif model_name == 'ResNet18':
        target_layer = model.model.layer3[-1].conv2
    elif model_name == 'ResNet50':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'ResNet101':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'ResNet152':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'VGG16':
        target_layer = model.model.features[28]
    elif model_name == 'VGG19':
        target_layer = model.model.features[34]
    elif model_name == 'EfficientNetB0':
        target_layer = model.model.features[8][0]
    elif model_name == 'EfficientNet_v2':
        target_layer = model.model.features[7][4].block[3]
    elif model_name == 'Swin-Transformer':
        target_layer = model.model.stages[3].blocks[-1].norm1
    elif model_name == 'Vit-Transformer':
        target_layer = model.model.layers[-1].attention
    elif model_name == 'DenseNet121':
        target_layer = model.model.features.norm5
    else:
        print(f"Grad-CAM not implemented for model {model_name}. Skipping visualization.")
    
    return target_layer

# Generate Grad-CAM visualizations for a select number of random images
def generate_gradcam_visualizations(model, data_module, class_labels, model_name, result_path, in_channels, logger=None):
    gradcam_save_dir = os.path.join(result_path, 'gradcam_visualizations')
    os.makedirs(gradcam_save_dir, exist_ok=True)

    # Get the target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name, logger)
    if target_layer is None:
        return

    grad_cam = GradCAM(model, target_layer) # Initialize Grad-CAM

    test_dataset = data_module.test_dataloader().dataset
    num_images = len(test_dataset)
    target_indices = [random.randint(0, num_images - 1) for _ in range(5)]  # Select 5 random images

    for idx in target_indices:
        try:
            img_tensor, label = test_dataset[idx] # Retrieve the image and label from the Dataset
        except IndexError:
            logger.warning(f"Index {idx} is out of bounds for the test dataset.")
            continue

        input_image = img_tensor.unsqueeze(0).to(model.device)  

        output = model(input_image) # Forward pass to get predictions
        threshold = 0.5   
        target_classes = torch.where(output[0] > threshold)[0].tolist() # Get relevant classes
        predicted_labels = [class_labels[i] for i in target_classes]
        actual_labels = [class_labels[i] for i, val in enumerate(label) if val == 1 or val > 0.5]

        heatmaps = {} # Generate heatmaps for each relevant class
        for target_class in target_classes:
            cam, _ = grad_cam.generate_heatmap(input_image, target_class=target_class)
            heatmaps[class_labels[target_class]] = cam

        # Convert the input tensor to a PIL image for visualization
        img = input_image.squeeze()  # Remove batch dimension
        if in_channels == 12:
            rgb_channels = [3, 2, 1]  
        else:
            rgb_channels = [2, 1, 0]
        img = img[rgb_channels, :, :] 
        
        # Normalize each channel for visualization
        img_cpu = img.detach().cpu().numpy() 
        red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
        green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
        blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)
        rgb_image = np.stack([red, green, blue], axis=-1) # Stack into an RGB image
        img = Image.fromarray((rgb_image * 255).astype(np.uint8)) # Convert to PIL Image

        # Save Grad-CAM visualizations for each class
        for class_name, heatmap in heatmaps.items(): # Display and save heatmaps for each class
            overlay = overlay_heatmap(img, heatmap, alpha=0.5) # Overlay heatmap on image

            plt.figure(figsize=(15, 5)) # Plot the results
            plt.subplot(1, 3, 1) # Original Image
            plt.title('Original Image')
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 3, 2) # Grad-CAM Heatmap
            plt.title(f'Heatmap for Class: {class_name}')
            plt.imshow(heatmap, cmap='jet')
            plt.axis('off')

            plt.subplot(1, 3, 3) # Overlayed Heatmap
            plt.title(f'Overlay for Class: {class_name}')
            plt.imshow(overlay)
            plt.axis('off')

            plt.suptitle(f'Image Index: {idx} | Class: {class_name}', fontsize=16) # Save and display the visualization
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_save_dir, f'gradcam_{idx}_{class_name}.png'))

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Image Index: {idx}', fontsize=14)
        plt.figtext(0.5, 0.01, f'Actual: {actual_labels} | Predicted: {predicted_labels}', wrap=True, horizontalalignment='center', fontsize=12)
        rgb_save_path = os.path.join(gradcam_save_dir, f'gradcam_rgb_{idx}.png')
        plt.savefig(rgb_save_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Grad-CAM visualizations saved to {gradcam_save_dir}")

# Generate Grad-CAM visualizations for a single image
def generate_gradcam_for_single_image(model, tiff_file_path, class_labels, model_name, result_path, in_channels, transforms, normalisations, logger):
    gradcam_save_dir = os.path.join(result_path, 'gradcam_visualizations')
    os.makedirs(gradcam_save_dir, exist_ok=True)

    # Get the target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name, logger)
    if target_layer is None:
        return

    grad_cam = GradCAM(model, target_layer) # Initialize Grad-CAM

    with rasterio.open(tiff_file_path) as src:
        image = src.read()

    image = torch.tensor(image, dtype=torch.float32)
    image = transforms(image)
    img_tensor = normalisations(image)

    input_tensor = img_tensor.unsqueeze(0).to(model.device)
    output = model(input_tensor)  # Forward pass to get predictions
    threshold = 0.5
    target_classes = torch.where(output[0] > threshold)[0].tolist()  # Get relevant classes
    predicted_labels = [class_labels[i] for i in target_classes]

    heatmaps = {}  # Generate heatmaps for each relevant class
    for target_class in target_classes:
        cam, _ = grad_cam.generate_heatmap(input_tensor, target_class=target_class)
        heatmaps[class_labels[target_class]] = cam

    # Convert the input tensor to a PIL image for visualization
    img = input_tensor.squeeze()  # Remove batch dimension
    if in_channels == 12:
        rgb_channels = [3, 2, 1]
    else:
        rgb_channels = [2, 1, 0]
    img = img[rgb_channels, :, :]

    # Normalize each channel for visualization
    img_cpu = img.detach().cpu().numpy()
    red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
    green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
    blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)  # Stack into an RGB image
    img = Image.fromarray((rgb_image * 255).astype(np.uint8))  # Convert to PIL Image

    # Save Grad-CAM visualizations for each class
    for class_name, heatmap in heatmaps.items():  # Display and save heatmaps for each class
        overlay = overlay_heatmap(img, heatmap, alpha=0.5)  # Overlay heatmap on image

        plt.figure(figsize=(15, 5))  # Plot the results
        plt.subplot(1, 3, 1)  # Original Image
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)  # Grad-CAM Heatmap
        plt.title(f'Heatmap for Class: {class_name}')
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')

        plt.subplot(1, 3, 3)  # Overlayed Heatmap
        plt.title(f'Overlay for Class: {class_name}')
        plt.imshow(overlay)
        plt.axis('off')

        plt.suptitle(f'Class: {class_name}', fontsize=16)  # Save and display the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_save_dir, f'gradcam_single_{class_name}.png'))

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image', fontsize=14)
    plt.figtext(0.5, 0.01, f'TIFF: {os.path.basename(tiff_file_path)} | Predicted: {predicted_labels}', wrap=True, horizontalalignment='center', fontsize=12)
    rgb_save_path = os.path.join(gradcam_save_dir, 'gradcam_rgb_single.png')
    plt.savefig(rgb_save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Grad-CAM visualizations saved to {gradcam_save_dir}')")

# Plot ROC-AUC curve
def plot_roc_auc(all_labels, all_probs, class_labels, save_dir=None, logger=None):
    logger.info("Plotting ROC-AUC curve...")
    num_classes = all_labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(num_classes): # Compute the ROC for each class
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(12, 8)) # Plot for each class
    for i in range(num_classes):
        plt.plot(
            fpr[i], tpr[i],
            lw=2,
            label=f'Class {class_labels[i]} (area = {roc_auc[i]:0.2f})'
        )

    plt.plot( # Plot micro-average
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
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "roc_auc_curve.png"))
        logger.info(f"ROC-AUC curve saved to {save_dir}")

# Predict on a batch of images
def predict_batch(model, dataloader, threshold=0.5, bands=DatasetConfig.all_bands):
    model.eval()
    all_preds = []
    all_true_labels = []

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

# Save batch predictions
def saving_batch_predictions(model, dataloader, in_channels, threshold=0.6, bands=DatasetConfig.all_bands, num_images=10, save_dir=None, logger=None):
    logger.info("Saving batch predictions...")
    all_preds, all_true_labels = predict_batch(model, dataloader, threshold, bands)
    
    dataset_size = len(dataloader.dataset)
    num_images = min(num_images, dataset_size)  
    
    # Randomly select unique indices
    random_indices = random.sample(range(dataset_size), num_images)
    
    if in_channels == 12:
        rgb_channels = [3, 2, 1]
    else:
        rgb_channels  = [2, 1, 0]

    for i in random_indices:
        pred = all_preds[i]
        true = all_true_labels[i]
        
        # Convert numeric predicted labels to text
        predicted_labels_indices = [idx for idx, value in enumerate(pred) if value == 1]
        true_labels_indices = [idx for idx, value in enumerate(true) if value == 1]
    
        # Get the image and select RGB bands for visualization
        image_tensor = dataloader.dataset[i][0]  
        img = image_tensor[rgb_channels , :, :] 

        # Normalize each channel
        img_cpu = img.detach().cpu().numpy()
        red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
        green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
        blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)

        
        rgb_image = np.stack([red, green, blue], axis=-1) # Stack into an RGB image
        image_rgb = Image.fromarray((rgb_image * 255).astype(np.uint8)) # Convert to PIL Image
    
        # Display the image, true labels, and predicted labels
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(
            f"Image {i}\n"
            f"True Labels: {true_labels_indices}\n"
            f"Predicted Labels: {predicted_labels_indices}"
        )
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"batch_prediction_{i}.png"))

        logger.info(f"Image {i}\n" f"True Labels: {true_labels_indices}\n" f"Predicted Labels: {predicted_labels_indices}")
        plt.close()

    print(f"Batch predictions saved to {save_dir}")

# Get sigmoid outputs
def get_sigmoid_outputs(model, dataset_dir, metadata_csv, bands=DatasetConfig.rgb_bands):
    test_metadata = metadata_csv[metadata_csv['split'] == 'test'] # Create dictionaries for mapping between labels and indices
    
    # Map band names to indices
    band_indices = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
        "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11
    }
    
    sigmoid_outputs_list = []

    # Process only the first 10 test images
    for image_file in tqdm(test_metadata['patch_id'].apply(lambda x: f"{x}.tif").tolist(), desc="Processing Images"):
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

# Plot the confusion matrix
def plot_per_label_confusion_matrices_grid(all_labels, all_preds, class_names=None, cols=4, save_dir=None, logger=None):
    logger.info("Plotting and Saving per-label confusion matrices...")
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    n_labels = len(mcm)

    # Determine how many rows we need
    rows = math.ceil(n_labels / cols)

    # Create a figure with (rows x cols) subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
    axes = axes.flatten()  

    for i, matrix in enumerate(mcm):
        tn, fp, fn, tp = matrix.ravel() # Flatten the 2x2 matrix into TN, FP, FN, TP
        label_name = class_names[i] if class_names else f"Label {i}"

        # Plot a heatmap for this label's 2x2 matrix on the i-th subplot
        ax = axes[i]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], ax=ax)
        ax.set_title(f'{label_name}\n(TN={tn}, FP={fp}, FN={fn}, TP={tp})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "confusion_matrices_grid.png"))
    plt.close()
    logger.info(f"Per-label confusion matrices saved to {save_dir}")

# Compute aggregated metrics
def compute_aggregated_metrics(all_labels, all_preds, all_probs=None, logger=None):
    logger.info("Computing aggregated metrics...")
    metrics_dict = {}
    
    # Micro-average: aggregates the contributions of all classes to compute the average metric
    metrics_dict['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    metrics_dict['f2_micro'] = fbeta_score(all_labels, all_preds, beta=2, average='micro', zero_division=0)

    # Macro-average: computes metric independently for each class and then takes the average
    metrics_dict['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics_dict['f2_macro'] = fbeta_score(all_labels, all_preds, beta=2, average='macro', zero_division=0)

    # Hamming loss: fraction of labels incorrectly predicted
    metrics_dict['hamming_loss'] = hamming_loss(all_labels, all_preds)
    # Subset accuracy: only 1 if all labels match exactly
    metrics_dict['subset_accuracy'] = accuracy_score(all_labels, all_preds)

    if all_probs is not None:
        # Average Precision (micro-averaged to match test_avg_precision)
        metrics_dict['avg_precision'] = average_precision_score(all_labels, all_probs, average='micro', pos_label=1)
        
        # One Error: fraction of samples where the top-ranked label is not in the true set
        top_pred_labels = np.argmax(all_probs, axis=1)  # Index of highest probability per sample
        true_positives = np.any(all_labels > 0, axis=1)  # Samples with at least one true positive label
        top_correct = np.array([all_labels[i, top_pred_labels[i]] for i in range(len(top_pred_labels))])
        if true_positives.sum() > 0:  # Avoid division by zero
            metrics_dict['one_error'] = 1 - (top_correct.sum() / true_positives.sum())
        else:
            metrics_dict['one_error'] = 0.0
            logger.warning("No positive labels in the dataset; one_error set to 0.")
    else:
        logger.warning("Probability outputs (all_probs) not provided. Skipping avg_precision and one_error.")

    return metrics_dict

# Plot and save the co-occurrence matrix
def plot_cooccurrence_matrix(all_labels, all_preds, class_names=None, save_dir=None, logger=None):
    logger.info("Plotting and saving co-occurrence matrix...")
    num_classes = all_labels.shape[1]
    cooccur = np.zeros((num_classes, num_classes), dtype=int)

    for n in range(all_labels.shape[0]):
        true_idxs = np.where(all_labels[n] == 1)[0] # find all true labels
        pred_idxs = np.where(all_preds[n] == 1)[0] # find all predicted labels
        for i in true_idxs: # increment co-occurrences
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
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "cooccurrence_matrix.png"))
    plt.close()
    logger.info(f"Co-occurrence matrix saved to {save_dir}")
    return cooccur

