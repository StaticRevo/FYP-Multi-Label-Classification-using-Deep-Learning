import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# GradCAM class for generating heatmaps
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output): # Forward hook to capture activations
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out): # Backward hook to capture gradients
            self.gradients = grad_out[0].detach()

        # Register forward and full backward hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

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

        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  

        # Weighted combination of activations
        weighted_activations = weights * activations  
        cam = torch.sum(weighted_activations, dim=1, keepdim=True) 

        # Apply ReLU to focus on positive influences
        cam = F.relu(cam)

        # Normalize the heatmap
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.squeeze().cpu().numpy()

        return cam, target_class

def overlay_heatmap(img, heatmap, alpha=0.5, colormap='jet'):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size, Image.LANCZOS)
    heatmap = heatmap.convert("RGB")
    heatmap = np.array(heatmap)

    heatmap = plt.get_cmap(colormap)(heatmap[:, :, 0])[:, :, :3]  # Apply colormap
    heatmap = np.uint8(255 * heatmap)

    overlay = np.array(img) * (1 - alpha) + heatmap * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


