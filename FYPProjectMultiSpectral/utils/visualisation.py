import torch
import math
import matplotlib.pyplot as plt

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
        # shape: [batch_size, n_channels, height, width]
        # For a single-image batch, squeeze out the batch dimension
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
