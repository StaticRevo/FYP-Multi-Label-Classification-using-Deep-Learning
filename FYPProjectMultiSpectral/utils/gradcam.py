import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward(retain_graph=True)

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)  
        cam = cam.cpu().numpy()

        # Normalize heatmap
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def visualize(self, input_image, heatmap, save_path=None):
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[1]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(input_image, 0.6, heatmap_colored, 0.4, 0)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(input_image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(heatmap, cmap="jet")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
