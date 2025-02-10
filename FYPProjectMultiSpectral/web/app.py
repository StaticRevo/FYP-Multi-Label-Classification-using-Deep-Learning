import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import rasterio
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Local application imports
from utils.model_utils import get_model_class
from config.config import DatasetConfig, ModelConfig, calculate_class_weights

app = Flask(__name__)

# Configure upload and static folders
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = os.path.join(current_dir, 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


MODEL_CONFIGS = {
    "ResNet18": {
         "checkpoint_path": r"C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_10%_BigEarthNet_2epochs\checkpoints\final.ckpt",
         "in_channels": 12,
         "weights": None
    },
    "ResNet50": {
         "checkpoint_path": r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_2epochs\checkpoints\final.ckpt",
         "in_channels": 12,
         "weights": None
    }
}
# A special key for the dropdown option "All"
MODEL_OPTIONS = list(MODEL_CONFIGS.keys()) + ["All"]

# Precompute class weights (assuming DatasetConfig.metadata_path is valid)
class_weights, class_weights_array = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))
CLASS_WEIGHTS = class_weights_array

# Load a model based on the model name
def load_model_for_name(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model configuration for {model_name} not found.")
    config = MODEL_CONFIGS[model_name]
    checkpoint_path = config["checkpoint_path"]
    in_channels = config["in_channels"]
    weights = config["weights"]
    main_path = os.path.dirname(checkpoint_path)  
    model_class, _ = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Model class for {model_name} not found!")
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        class_weights=CLASS_WEIGHTS,
        num_classes=DatasetConfig.num_classes,
        in_channels=in_channels,
        model_weights=weights,
        main_path=main_path
    )
    model.eval()
    print(f"{model_name} loaded successfully.")
    return model

# --- TIFF Preprocessing ---
def preprocess_tiff_image(file_path, selected_bands=None):
    # Read image data; shape: (channels, height, width)
    with rasterio.open(file_path) as src:
        image = src.read()

    # Select channels; if none provided, use all channels
    if selected_bands is None:
        selected_bands = list(range(image.shape[0]))
    image = image[selected_bands, :, :]

    # Convert to a float32 tensor and normalize each channel to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32)
    for i in range(image_tensor.shape[0]):
        band = image_tensor[i]
        image_tensor[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)

    if image_tensor.shape[0] > 4:     # If the image has more than 4 channels, avoid converting to PIL.
        # Add a batch dimension: shape [1, C, H, W]
        image_tensor = image_tensor.unsqueeze(0)
        # Use bilinear interpolation to resize
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # Now image_tensor is [1, C, 224, 224]
        return image_tensor
    else:
        # For images with 3 or 4 channels, you can use PIL-based resizing.
        preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = preprocess_transform(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

# --- Create an RGB Visualization ---
def create_rgb_visualization(file_path, selected_bands=None):
    with rasterio.open(file_path) as src:
        image = src.read()
    if selected_bands is None:
        if image.shape[0] >= 4:
            selected_bands = [3, 2, 1]
        else:
            selected_bands = [0, 1, 2]
    red = image[selected_bands[0]].astype(np.float32)
    green = image[selected_bands[1]].astype(np.float32)
    blue = image[selected_bands[2]].astype(np.float32)
    red = (red - red.min()) / (red.max() - red.min() + 1e-8)
    green = (green - green.min()) / (green.max() - green.min() + 1e-8)
    blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    out_filename = "result_image.png"
    out_path = os.path.join(STATIC_FOLDER, out_filename)
    Image.fromarray(rgb_image).save(out_path)
    return url_for('static', filename=out_filename)

# --- Prediction Function ---
def predict_image_for_model(model, image_tensor):
    # Get the device that the model is on (e.g., cuda:0)
    device = next(model.parameters()).device
    # Move the input tensor to the same device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        # Move the output tensor to CPU before converting to numpy
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    
    selected_classes = []
    
    # Iterate over each probability with its index
    for idx, prob in enumerate(probs):
        if prob > 0.5:
            # Retrieve the label from class_labels_dict, using a default if the key is missing
            label = DatasetConfig.class_labels_dict.get(idx, f"Class_{idx}")
            selected_classes.append(f"{label} ({prob:.3f})")
    
    return selected_classes

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Retrieve the uploaded file and selected model option
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        selected_model_option = request.form.get("model")
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Create an RGB composite visualization URL for display
        rgb_url = create_rgb_visualization(file_path)
        
        # Preprocess the TIFF file
        input_tensor = preprocess_tiff_image(file_path)
        
        # Dictionary to hold predictions per model
        predictions_dict = {}
        
        if selected_model_option == "All":
            # Run prediction for each model in the configuration
            for model_name in MODEL_CONFIGS.keys():
                model_instance = load_model_for_name(model_name)
                preds = predict_image_for_model(model_instance, input_tensor)
                predictions_dict[model_name] = np.array2string(preds, precision=3)
        else:
            # Run prediction for the selected model only
            model_instance = load_model_for_name(selected_model_option)
            preds = predict_image_for_model(model_instance, input_tensor)
            predictions_dict[selected_model_option] = np.array2string(preds, precision=3)
        
        return render_template('result.html',
                               filename=filename,
                               predictions=predictions_dict,
                               rgb_url=rgb_url)
    # GET: Render the upload form with a model selection dropdown.
    return render_template('upload.html', model_options=MODEL_OPTIONS)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
