import os
import subprocess
import pandas as pd
from config.config import DatasetConfig, ModelConfig
from dataloader import BigEarthNetTIFDataModule
import torch
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import *
from config.config import clean_and_parse_labels, calculate_class_weights
from utils.test_functions import *
from models.models import *
from utils.visualisation import *
import json  
from utils.gradcam import GradCAM, overlay_heatmap
from utils.test_functions import *

# Set float32 matmul precision to 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision('high')
set_random_seeds()

def test_model(
        model_name, 
        model_weights,
        selected_bands,
        selected_dataset,
        acc_checkpoint_path,
        loss_checkpoint_path,
        last_checkpoint_path, 
        in_channels,
        class_weights,
        metadata_csv, 
        dataset_dir, 
        bands
    ):
    
    model, data_module, class_labels = load_model_and_data(last_checkpoint_path, 
                                                           metadata_csv,
                                                           dataset_dir,
                                                           model_class,
                                                           model_name,
                                                           bands,
                                                           num_classes=19,
                                                           in_channels=len(bands),
                                                           model_weights=model_weights)
                                                        
    register_hooks(model)

    # Set up Trainer for testing
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision='16-mixed',
        deterministic=True,
    )

    # Run the testing phase
    trainer.test(model, datamodule=data_module)

   
    all_preds, all_labels = calculate_and_save_metrics(model, data_module, model_name, selected_dataset, class_labels)

    # Save predictions and true labels (if needed)
    save_path = f'test_predictions_{model_name}_{selected_dataset}.npz'
    np.savez(save_path, all_preds=all_preds, all_labels=all_labels)

    per_class_metrics_path = f'test_per_class_metrics_ResNet.json'
    if os.path.exists(per_class_metrics_path):
        with open(per_class_metrics_path, 'r') as f:
            per_class_metrics = json.load(f)
        
        # Print per-class metrics with class labels
        print("\nPer-Class Metrics:")
        for metric, values in per_class_metrics.items():
            if metric == 'class_labels':
                continue  # Skip class_labels key
            print(f"\n{metric.capitalize()}:")
            for i, val in enumerate(values):
                class_name = class_labels[i] if i < len(class_labels) else f"Class {i}"
                print(f"  {i} ({class_name}): {val:.4f}")
    else:
        print(f"\nPer-class metrics file not found at {per_class_metrics_path}")

    test_loader = data_module.test_dataloader()
    test_dataset = test_loader.dataset

    # Call the function to display predictions
    display_batch_predictions(model, test_loader, threshold=0.6, bands=DatasetConfig.all_bands)
    
    # Visualize activations
    test_loader = data_module.test_dataloader()
    example_batch = next(iter(test_loader))
    example_imgs, example_lbls = example_batch
    show_rgb_from_batch(example_imgs[0])
    example_imgs = example_imgs.to(model.device)
    clear_activations()
    with torch.no_grad():
        _ = model(example_imgs[0].unsqueeze(0))
    visualize_activations(num_filters=16)  

    # Plot label confusion matrices
    plot_per_label_confusion_matrices_grid(all_labels, all_preds, class_names=class_labels)

    # Plot aggregated confusion matrices
    scores = compute_aggregated_metrics(all_labels, all_preds)
    print(scores)
    
    # Plot co-occurrence matrix
    plot_cooccurrence_matrix(all_labels, all_preds, class_names=class_labels)

    model_name = model_name.toLower()
    target_layer = ModelConfig.gradcam_target_layers[model_name]

    if target_layer is None:
        raise ValueError(f"Target layer for Grad-CAM is not defined for model: {model_name}")

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    save_dir = 'gradcam_results'
    os.makedirs(save_dir, exist_ok=True)

    # Generate a random index for target_image_indices
    num_images = len(test_dataset)
    target_image_indices = [random.randint(0, num_images - 1)]

       # Define normalization parameters on the correct device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(model.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(model.device)

    for idx in target_image_indices:
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
        rgb_channels = [1, 2, 3]  
        img = img[rgb_channels, :, :] 
        img = img * std + mean  # Unnormalize
        img = torch.clamp(img, 0, 1)
        img = img.cpu()  # Move to CPU after unnormalization
        img = transforms.ToPILImage()(img)

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
            plt.savefig(os.path.join(save_dir, f'gradcam_{idx}_{class_name}.png'))
            plt.show()

if __name__ == "__main__":
    last_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\final.ckpt'
    loss_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\final.ckpt'
    acc_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\final.ckpt'
    metadata_path = r'C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\metadata_1_percent.csv'
    metadata_csv = pd.read_csv(metadata_path)
    dataset_dir = DatasetConfig.dataset_paths["1"]
    model_class = BigEarthNetResNet18ModelTIF
    model_name = "ResNet18"
    selected_dataset = "1%_BigEarthNet"
    selected_bands = 'all_bands'
    bands = DatasetConfig.all_bands
    num_classes = 19
    in_channels = 12
    model_weights = 'ResNet18_Weights.DEFAULT'

    # Calculate class weights
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    test_model(
        model_name, 
        model_weights,
        selected_bands,
        selected_dataset,
        acc_checkpoint_path,
        loss_checkpoint_path,
        last_checkpoint_path, 
        in_channels,
        class_weights,
        metadata_csv, 
        dataset_dir, 
        bands
    )
