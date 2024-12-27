from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import torch.nn.functional as F

# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Register forward and full backward hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class=None):
        output = self.model(input_image)  # Forward pass

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()  # Zero gradients

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
        cam = cam / (cam.max() + 1e-8)
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

# Define the save_gradcam_visualizations function
def save_gradcam_visualizations(input_image, heatmaps, class_labels, idx, save_dir, mean=None, std=None):
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(input_image.device)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(input_image.device)

    # Convert tensor to PIL image
    img = input_image.squeeze(0).cpu().detach()
    img = img * std + mean  # Unnormalize
    img = torch.clamp(img, 0, 1)
    img = transforms.ToPILImage()(img)

    for class_name, heatmap in heatmaps.items():
        # Overlay heatmap on image
        overlay = GradCAM.overlay_heatmap(img, heatmap, alpha=0.5)

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
        plt.close() 