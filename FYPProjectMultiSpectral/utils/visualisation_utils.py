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

activations = {} # Dictionary to store activations from the model

# Register forward hook for the model
def forward_hook(module, input, output):
    activations[module] = output

def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(forward_hook)

# Clear the activations        
def clear_activations():
    activations.clear()

# Visualize the activations from the model
def visualize_activations(result_path, num_filters=8):
    pdf_path = os.path.join(result_path, "activations.pdf")
    with PdfPages(pdf_path) as pdf:
        for layer_module, activation in activations.items():
            act = activation.squeeze(0).detach().cpu().numpy()
            n_filters = min(num_filters, act.shape[0])
            grid_size = int(math.ceil(math.sqrt(n_filters)))

            fig, axes = plt.subplots(
                grid_size, 
                grid_size, 
                figsize=(grid_size * 3, grid_size * 3)
            )
            
            # Ensure axes is always iterable
            if grid_size == 1:
                axes = np.array([axes])
            else:
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

# Show the RGB image from a batch of multi-spectral images
def show_rgb_from_batch(image_tensor, in_channels, save_path=None):
    image_cpu = image_tensor.detach().cpu().numpy()

    if in_channels == 12:
        red = image_cpu[3]
        green = image_cpu[2]
        blue = image_cpu[1]
    else:
        red = image_cpu[0]
        green = image_cpu[1]
        blue = image_cpu[2]

    # Normalize the bands
    red = (red - red.min()) / (red.max() - red.min() + 1e-8)
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-8)

    # Stack into an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Create the figure
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title("RGB Visualization of Multi-Spectral Image")
    plt.axis('off')

    if save_path:
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "activations_image.png")
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()  
        print(f"Saved RGB activation visualization to {save_file}")

# Save the tensorboard graphs as images
def save_tensorboard_graphs(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

    # Load the TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    tags = event_acc.Tags().get('scalars', []) # Get the list of metrics in the logs

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
        plt.xlabel('Step')
        plt.ylabel(tag)
        plt.title(tag.replace('_', ' ').capitalize())
        plt.grid(True)

        # Save the graph as an image
        sanitized_tag = tag.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{sanitized_tag}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Graphs saved to {output_dir}")