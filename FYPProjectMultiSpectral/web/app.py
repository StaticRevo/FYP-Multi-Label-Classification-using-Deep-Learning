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

# Local application imports
from utils.model_utils import get_model_class
from config.config import DatasetConfig, ModelConfig, calculate_class_weights

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads') # Folder for temporarily saving uploaded files
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static') # Folder for static files (for saving generated images)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# --- Checkpoint and Model Settings ---
CHECKPOINT_PATH = r"C:\Users\isaac\Desktop\experiments\ResNet18_None_all_bands_10%_BigEarthNet_2epochs\checkpoints\final.ckpt"
MODEL_NAME = "ResNet18"
WEIGHTS = None      
IN_CHANNELS = 12        
MAIN_PATH = os.path.dirname(CHECKPOINT_PATH)
class_weights, class_weights_array = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))
CLASS_WEIGHTS = class_weights_array

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    try:
        model_class, _ = get_model_class(MODEL_NAME)
        if model_class is None:
            raise ValueError(f"Model class for {MODEL_NAME} not found!")
        model = model_class.load_from_checkpoint(
            CHECKPOINT_PATH,
            class_weights=CLASS_WEIGHTS,
            num_classes=DatasetConfig.num_classes,
            in_channels=IN_CHANNELS,
            model_weights=WEIGHTS,
            main_path=MAIN_PATH
        )
        model.eval()
        print("Model loaded successfully from checkpoint.")
        return model
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        exit(1)

# Load the model once when the app starts.
model = load_model()

# --- TIFF Preprocessing ---
def preprocess_tiff_image(file_path, selected_bands=None):
    with rasterio.open(file_path) as src:
        image = src.read()  # shape: (channels, height, width)
    
    # Default band selection: use bands [3, 2, 1] if available
    if selected_bands is None:
        if image.shape[0] >= 4:
            selected_bands = [3, 2, 1]
        else:
            selected_bands = [0, 1, 2]
    
    image = image[selected_bands, :, :]
    
    # Convert to float32 tensor and normalize each band to [0, 1]
    image_tensor = torch.tensor(image, dtype=torch.float32)
    for i in range(image_tensor.shape[0]):
        band = image_tensor[i]
        image_tensor[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    
    # Resize using torchvision transforms
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Adjust as needed
        transforms.ToTensor(),
    ])
    image_tensor = preprocess_transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
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
def predict_image(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output).squeeze().numpy()
    return probs

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Create an RGB composite visualization URL for display
            rgb_url = create_rgb_visualization(file_path)
            
            # Preprocess the TIFF file and run prediction
            input_tensor = preprocess_tiff_image(file_path)
            predictions = predict_image(input_tensor)
            predictions_str = np.array2string(predictions, precision=3)
            
            return render_template('result.html',
                                   filename=filename,
                                   predictions=predictions_str,
                                   rgb_url=rgb_url)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)