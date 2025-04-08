# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# GradCAM class for generating heatmaps
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.model.eval()
        self._register_hooks()

    # Register hooks for the target layer
    def _register_hooks(self):
        def forward_hook(module, input, output): # Forward hook to capture activations
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out): # Backward hook to capture gradients
            self.gradients = grad_out[0].detach()

        # Register forward and full backward hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    # Generate the heatmap
    def generate_heatmap(self, input_image, target_class=None):
        output = self.model(input_image)  # Forward pass

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad() # Zero gradients

        # Backward pass for the target class
        loss = output[:, target_class]
        loss.backward()

        # Retrieve captured gradients and activations
        gradients = self.gradients  
        activations = self.activations  
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # Global average pooling on gradients

        # Weighted combination of activations
        weighted_activations = weights * activations  
        cam = torch.sum(weighted_activations, dim=1, keepdim=True) 
        cam = F.relu(cam) # Apply ReLU to focus on positive influences

        # Normalize the heatmap
        cam = cam - cam.min()  
        cam = cam / (cam.max() + 1e-8)  
        cam = cam.squeeze().cpu().numpy()

        return cam, target_class

# Overlay the heatmap on the image
def overlay_heatmap(img, heatmap, alpha=0.8, colormap='jet'):
    # Apply Gaussian smoothing to reduce noise
    heatmap = gaussian_filter(heatmap, sigma=1.5) 
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Re-normalize after smoothing

    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size, Image.LANCZOS)
    heatmap = np.array(heatmap, dtype=np.float32) / 255.0
    heatmap = plt.get_cmap(colormap)(heatmap)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    overlay = np.array(img) * (1 - alpha) + heatmap * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


