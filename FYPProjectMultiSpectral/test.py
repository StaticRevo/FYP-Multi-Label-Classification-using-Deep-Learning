import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random

from config.config import DatasetConfig
from config.config_utils import calculate_class_weights
from utils.test_functions import *
from utils.helper_functions import *
from utils.gradcam import GradCAM, overlay_heatmap
from utils.visualisation import *
from models.models import *
from dataloader import BigEarthNetTIFDataModule
import torch.nn.functional as F

if __name__ == "__main__":
    # Paths and configurations
    metadata_path = DatasetConfig.metadata_paths["1"]
    metadata_csv = pd.read_csv(metadata_path)
    dataset_dir = DatasetConfig.dataset_paths['1']
    class_labels = DatasetConfig.class_labels
    bands = DatasetConfig.all_bands

    # Calculate class weights
    class_weights, class_weights_array = calculate_class_weights(metadata_csv)
    class_weights = class_weights_array

    num_classes = DatasetConfig.num_classes
    in_channels = len(bands)
    model_weights = 'ResNet18_Weights.DEFAULT'

    # Initialize the DataModule
    # data_module = BigEarthNetTIFDataModule(
    #     bands=bands,
    #     dataset_dir=dataset_dir, 
    #     metadata_csv=metadata_csv
    # )
    # data_module.setup(stage='test')

    # # Load the trained model
    # model_checkpoint_path = r'C:\Users\isaac\Desktop\experiments\checkpoints\ResNet18_ResNet18_Weights.DEFAULT_all_bands_1%_BigEarthNet\final.ckpt'

    # model = BigEarthNetResNet18ModelTIF.load_from_checkpoint(
    #     model_checkpoint_path,
    #     class_weights=class_weights,
    #     num_classes=num_classes, 
    #     in_channels=in_channels, 
    #     model_weights=model_weights
    # )
    # model.eval()
    # register_hooks(model)

    # # Visualize activations
    # test_loader = data_module.test_dataloader()
    # example_batch = next(iter(test_loader))
    # example_imgs, example_lbls = example_batch
    # show_rgb_from_batch(example_imgs[0])
    # example_imgs = example_imgs.to(model.device)
    # clear_activations()
    # with torch.no_grad():
    #     _ = model(example_imgs[0].unsqueeze(0))
    # visualize_activations(num_filters=16)  

    # model_name =  'ResNet18'
    # model_name = model_name.lower()
    # print(f"Model name: {model_name}")
    # target = ModelConfig.gradcam_target_layers[model_name]
    # print(f"Grad-CAM target layer: {target}")
    
    # # Identify the target convolutional layer for Grad-CAM
    # try:
    #     target_layer = model.model.layer3[-1].conv2 
    # except AttributeError:
    #     target_layer = model.layer3[-1].conv2

    # # Initialize Grad-CAM
    # grad_cam = GradCAM(model, target_layer)

    # # Access the underlying test dataset from the DataLoader
    # test_loader = data_module.test_dataloader()
    # test_dataset = test_loader.dataset

    # # Generate a random index for target_image_indices
    # num_images = len(test_dataset)
    # target_image_indices = [random.randint(0, num_images - 1)]

    # # Create directory to save Grad-CAM results
    # save_dir = 'gradcam_results'
    # os.makedirs(save_dir, exist_ok=True)

    # # Define normalization parameters on the correct device
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(model.device)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(model.device)

    # for idx in target_image_indices:
    #     try:
    #         # Retrieve the image and label from the Dataset
    #         img_tensor, label = test_dataset[idx]
    #     except IndexError:
    #         print(f"Index {idx} is out of bounds for the test dataset.")
    #         continue

    #     input_image = img_tensor.unsqueeze(0).to(model.device)  

    #     # Forward pass to get predictions
    #     output = model(input_image)

    #     # Get relevant classes for multi-label classification
    #     threshold = 0.5  
    #     target_classes = torch.where(output[0] > threshold)[0].tolist()

    #     # Generate heatmaps for each relevant class
    #     heatmaps = {}
    #     for target_class in target_classes:
    #         cam, _ = grad_cam.generate_heatmap(input_image, target_class=target_class)
    #         heatmaps[class_labels[target_class]] = cam

    #     # Convert the input tensor to a PIL image for visualization
    #     img = input_image.squeeze()  # Remove batch dimension
    #     rgb_channels = [3, 2, 1]  
    #     img = img[rgb_channels, :, :] 

    #     # Normalize each channel
    #     img_cpu = img.detach().cpu().numpy()
    #     red = (img_cpu[0] - img_cpu[0].min()) / (img_cpu[0].max() - img_cpu[0].min() + 1e-8)
    #     green = (img_cpu[1] - img_cpu[1].min()) / (img_cpu[1].max() - img_cpu[1].min() + 1e-8)
    #     blue = (img_cpu[2] - img_cpu[2].min()) / (img_cpu[2].max() - img_cpu[2].min() + 1e-8)

    #     # Stack into an RGB image
    #     rgb_image = np.stack([red, green, blue], axis=-1)

    #     # Convert to PIL Image
    #     img = Image.fromarray((rgb_image * 255).astype(np.uint8))

    #     # Display and save heatmaps for each class
    #     for class_name, heatmap in heatmaps.items():
    #         # Overlay heatmap on image
    #         overlay = overlay_heatmap(img, heatmap, alpha=0.5)

    #         # Plot the results
    #         plt.figure(figsize=(15, 5))

    #         # Original Image
    #         plt.subplot(1, 3, 1)
    #         plt.title('Original Image')
    #         plt.imshow(img)
    #         plt.axis('off')

    #         # Grad-CAM Heatmap
    #         plt.subplot(1, 3, 2)
    #         plt.title(f'Heatmap for Class: {class_name}')
    #         plt.imshow(heatmap, cmap='jet')
    #         plt.axis('off')

    #         # Overlayed Heatmap
    #         plt.subplot(1, 3, 3)
    #         plt.title(f'Overlay for Class: {class_name}')
    #         plt.imshow(overlay)
    #         plt.axis('off')

    #         # Save and display the visualization
    #         plt.suptitle(f'Image Index: {idx} | Class: {class_name}', fontsize=16)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, f'gradcam_{idx}_{class_name}.png'))
    #         plt.show()
