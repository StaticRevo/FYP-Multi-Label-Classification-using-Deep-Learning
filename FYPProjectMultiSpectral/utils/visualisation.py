import torch
import math
import matplotlib.pyplot as plt
import numpy as np

activations = {}

def forward_hook(module, input, output):
    activations[module] = output

def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(forward_hook)

def clear_activations():
    activations.clear()

def visualize_activations(num_filters=8):
    for layer_module, activation in activations.items():
        act = activation.squeeze(0).detach().cpu().numpy()  # shape: [n_channels, height, width]

        n_filters = min(num_filters, act.shape[0])
        grid_size = int(math.ceil(math.sqrt(n_filters)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(grid_size * grid_size):
            if i < n_filters:
                axes[i].imshow(act[i], cmap='viridis')
                axes[i].axis('off')
            else:
                axes[i].remove()

        plt.suptitle(f"Activations from layer: {layer_module}", fontsize=16)
        plt.tight_layout()
        plt.show()

def show_rgb_from_batch(image_tensor):
    image_cpu = image_tensor.detach().cpu().numpy()

    red = image_cpu[3]
    green = image_cpu[2]
    blue = image_cpu[1]

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