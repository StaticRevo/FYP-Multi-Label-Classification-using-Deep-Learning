import os
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

activations = {}

# Visualisation functions for activations within the model
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