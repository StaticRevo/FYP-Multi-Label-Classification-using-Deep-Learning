import numpy as np
import matplotlib.pyplot as plt

# Define the hook function
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Visualize activations function
def visualize_activations(layer_names, activations):
    images_per_row = 16
    for layer_name in layer_names:
        layer_activation = activations[layer_name].squeeze().cpu().numpy()
        n_features = layer_activation.shape[0] 
        size = layer_activation.shape[1] 
        n_cols = n_features // images_per_row  
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  
            for row in range(images_per_row):
                channel_image = layer_activation[col * images_per_row + row]
                channel_image -= channel_image.mean() 
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,  
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()