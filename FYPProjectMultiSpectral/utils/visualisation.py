# Standard library imports
import os
import math

# Third-party imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import rasterio

# Visualisation functions for activations within the model
activations = {}
def forward_hook(module, input, output):
    activations[module] = output

def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(forward_hook)
        
def clear_activations():
    activations.clear()

def visualize_activations(result_path, num_filters=8):
    pdf_path = os.path.join(result_path, "activations.pdf")
    with PdfPages(pdf_path) as pdf:
        for layer_module, activation in activations.items():
            act = activation.squeeze(0).detach().cpu().numpy()
            n_filters = min(num_filters, act.shape[0])
            grid_size = int(math.ceil(math.sqrt(n_filters)))

            fig, axes = plt.subplots(
                grid_size, 
                grid_size, figsize=(grid_size * 3, grid_size * 3))
            axes = axes.flatten()

            for i in range(grid_size * grid_size):
                if i < n_filters:
                    axes[i].imshow(act[i], cmap='viridis')
                    axes[i].axis('off')
                else:
                    axes[i].remove()

            plt.suptitle(f"Activations from layer: {layer_module}", fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("Activations saved to activations.pdf")

def show_rgb_from_batch(image_tensor, in_channels):
    image_cpu = image_tensor.detach().cpu().numpy()

    if in_channels == 12:
        red = image_cpu[3]
        green = image_cpu[2]
        blue = image_cpu[1]
    else:
        red = image_cpu[0]
        green = image_cpu[1]
        blue = image_cpu[2]

    red = (red - red.min()) / (red.max() - red.min() + 1e-8)
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-8)

    # Stack into an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title("RGB Visualization of Multi-Spectral Image")
    plt.axis('off')
    plt.show()

def save_tensorboard_graphs(log_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get the list of tags (metrics) in the logs
    tags = event_acc.Tags().get('scalars', [])

    # Filter tags: include tags with '_epoch' or starting with 'val_', exclude 'class' in name
    filtered_tags = [
        tag for tag in tags 
        if ('_epoch' in tag or tag.startswith('val_')) and 'class' not in tag
    ]

    # Iterate over each tag and plot the graph
    for tag in filtered_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Plot the graph
        plt.figure()
        plt.plot(steps, values, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel(tag)
        plt.title(tag.replace('_', ' ').capitalize())
        plt.grid(True)

        # Save the graph as an image
        sanitized_tag = tag.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{sanitized_tag}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Graphs saved to {output_dir}")

def get_labels_for_image(image_path, model, transform, patch_to_labels):
    with rasterio.open(image_path) as src: # Load and preprocess the image
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        image = transform(image).unsqueeze(0).to(model.device)  # Add batch dimension and move to device

    # Get the predicted labels
    model.eval()
    with torch.no_grad():
        preds = model(image).sigmoid() > 0.5  # Apply sigmoid and threshold at 0.5
        preds = preds.cpu().numpy().astype(int).flatten()

    # Get the true labels
    patch_id = os.path.basename(image_path).split('.')[0]
    true_labels = patch_to_labels[patch_id]

    return preds, true_labels, image

def display_image(image_path):
    with rasterio.open(image_path) as src:
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        plt.imshow(image)
        plt.title("Image with Bands 2, 3, and 4")
        plt.show()

def display_image_and_labels(image_path, model, transform, patch_to_labels):
    # Display the image
    display_image(image_path)

    # Get predicted and true labels
    preds, true_labels, _ = get_labels_for_image(image_path, model, transform, patch_to_labels)
    print(f"Predicted Labels: {preds}")
    print(f"True Labels: {true_labels}")

def display_rgb_image_from_tiff(tiff_file_path):
    with rasterio.open(tiff_file_path) as src:
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
        
        # Display the RGB image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()