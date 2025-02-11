# Standard library imports
import os
import sys
import json
import secrets
import time
import subprocess

# Third-party imports
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
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
from utils.file_utils import initialize_paths
from models.models import *
from utils.gradcam import GradCAM, overlay_heatmap 
from utils.data_utils import extract_number

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

_cached_training_log = None
_last_training_log_time = 0

_cached_testing_log = None
_last_testing_log_time = 0

CACHE_DURATION = 5

# Configure upload and static folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
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
# Dropdown option "All"
MODEL_OPTIONS = list(MODEL_CONFIGS.keys()) + ["All"]

# Precompute class weights (assuming DatasetConfig.metadata_path is valid)
class_weights, class_weights_array = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))
CLASS_WEIGHTS = class_weights_array

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

# Helper function to load metrics for an experiment.
def load_experiment_metrics(experiment_name):
    experiment_path = os.path.join(EXPERIMENTS_DIR, experiment_name)
    results_path = os.path.join(experiment_path, "results")
    metrics = {}
    metrics_file = os.path.join(results_path, "best_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            metrics = {"error": f"Error loading metrics: {e}"}
    return metrics

# --- TIFF Preprocessing ---
def preprocess_tiff_image(file_path, selected_bands=None):
    with rasterio.open(file_path) as src:
        image = src.read()
    if selected_bands is None:
        selected_bands = list(range(image.shape[0]))
    image = image[selected_bands, :, :]
    image_tensor = torch.tensor(image, dtype=torch.float32)
    for i in range(image_tensor.shape[0]):
        band = image_tensor[i]
        image_tensor[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    if image_tensor.shape[0] > 4:
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return image_tensor
    else:
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
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    predictions = []
    for idx, prob in enumerate(probs):
        if prob > 0.5:
            label = DatasetConfig.reversed_class_labels_dict.get(idx, f"Class_{idx}")
            predictions.append({"label": label, "probability": prob})
    return predictions

# --- GradCAM Visualization ---
def generate_gradcam_for_single_image(model, input_tensor, model_name):
    if model_name == 'ResNet18':
        target_layer = model.model.layer3[-1].conv2
    elif model_name == 'ResNet50':
        target_layer = model.model.layer3[-1].conv3
    elif model_name == 'VGG16':
        target_layer = model.model.features[28]
    elif model_name == 'VGG19':
        target_layer = model.model.features[34]
    elif model_name == 'EfficientNetB0':
        target_layer = model.model.features[8][0]
    elif model_name == 'EfficientNet_v2':
        target_layer = model.modelfeatures[7][4].block[3]
    elif model_name == 'Swin-Transformer':
        target_layer = model.model.stages[3].blocks[-1].norm1
    elif model_name == 'Vit-Transformer':
        target_layer = model.model.layers[-1].attention
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    grad_cam = GradCAM(model, target_layer)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    probs = torch.sigmoid(output).squeeze().cpu().numpy()

    img_tensor = input_tensor[0].detach().cpu()
    if img_tensor.shape[0] >= 4:
        rgb_tensor = img_tensor[[3, 2, 1], :, :]
    else:
        rgb_tensor = img_tensor[:3]
    rgb_np = rgb_tensor.numpy()
    red   = (rgb_np[0] - rgb_np[0].min()) / (rgb_np[0].max() - rgb_np[0].min() + 1e-8)
    green = (rgb_np[1] - rgb_np[1].min()) / (rgb_np[1].max() - rgb_np[1].min() + 1e-8)
    blue  = (rgb_np[2] - rgb_np[2].min()) / (rgb_np[2].max() - rgb_np[2].min() + 1e-8)
    rgb_image = np.stack([red, green, blue], axis=-1)
    base_img = Image.fromarray((rgb_image * 255).astype(np.uint8))
    # Do not upscale base_img so that it remains clear

    gradcam_results = {}
    threshold = 0.5
    for idx, prob in enumerate(probs):
        if prob > threshold:
            cam, _ = grad_cam.generate_heatmap(input_tensor, target_class=idx)
            overlay_img = overlay_heatmap(base_img, cam, alpha=0.5)
            filename = f"gradcam_{model_name}_{idx}.png"
            out_path = os.path.join(STATIC_FOLDER, filename)
            overlay_img.save(out_path)
            class_label = DatasetConfig.reversed_class_labels_dict.get(idx, f"Class_{idx}")
            gradcam_results[class_label] = url_for('static', filename=filename)
    return gradcam_results

# --- Helper Function to Fetch Actual Labels from Metadata ---
def fetch_actual_labels(patch_id):
    import ast
    metadata_df = pd.read_csv(DatasetConfig.metadata_path)
    row = metadata_df.loc[metadata_df['patch_id'] == patch_id]
    if row.empty:
        return []
    labels_str = row.iloc[0]['labels']
    if isinstance(labels_str, str):
        try:
            # Clean the string similar to the dataset class logic
            cleaned_labels = labels_str.replace(" '", ", '").replace("[", "[").replace("]", "]")
            labels = ast.literal_eval(cleaned_labels)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing labels for patch_id {patch_id}: {e}")
            labels = []
    else:
        labels = labels_str
    return labels

# Homepage that provides options for Train, Test, or Predict
@app.route("/")
def index():
    return render_template("index.html")

# --- Predict Page ---
@app.route("/predict", methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        selected_model_option = request.form.get("model")
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        rgb_url = create_rgb_visualization(file_path)
        input_tensor = preprocess_tiff_image(file_path)
        
        predictions_dict = {}
        gradcam_dict = {}
        multiple_models = False
        
        patch_id = os.path.splitext(filename)[0]
        actual_labels = fetch_actual_labels(patch_id)

        if selected_model_option == "All":
            multiple_models = True
            for model_name in MODEL_CONFIGS.keys():
                model_instance = load_model_for_name(model_name)
                preds = predict_image_for_model(model_instance, input_tensor)
                predictions_dict[model_name] = preds
                gradcam_dict[model_name] = generate_gradcam_for_single_image(model_instance, input_tensor, model_name)
        else:
            model_instance = load_model_for_name(selected_model_option)
            preds = predict_image_for_model(model_instance, input_tensor)
            predictions_dict[selected_model_option] = preds
            gradcam_dict = generate_gradcam_for_single_image(model_instance, input_tensor, selected_model_option)

        return render_template('result.html',
                               filename=filename,
                               predictions=predictions_dict,
                               actual_labels=actual_labels,
                               rgb_url=rgb_url,
                               gradcam=gradcam_dict,
                               multiple_models=multiple_models)
    return render_template('upload.html', model_options=MODEL_OPTIONS)

# --- Logs (for Training) ---
@app.route('/logs')
def logs():
    global _cached_training_log, _last_training_log_time
    # Retrieve the main experiment path from the session.
    main_path = session.get('main_path')
    if not main_path:
        return "No training run information found in session."

    # Construct the log file path for training logs.
    log_dir = os.path.join(main_path, 'logs')
    training_log_path = os.path.join(log_dir, 'training_logs')
    log_file = os.path.join(training_log_path, 'training.log')
    
    current_time = time.time()
    # Check if we need to re-read the file (once every CACHE_DURATION seconds)
    if _cached_training_log is None or (current_time - _last_training_log_time) > CACHE_DURATION:
        try:
            with open(log_file, 'r') as f:
                _cached_training_log = f.read()
        except Exception as e:
            _cached_training_log = f"Error reading log file: {e}"
        _last_training_log_time = current_time

    return _cached_training_log

# --- Logs (for Testing) ---
@app.route('/logs_test')
def logs_test():
    global _cached_testing_log, _last_testing_log_time
    # Retrieve the main experiment path from the session.
    main_path = session.get('main_path')
    if not main_path:
        return "No testing run information found in session."

    # Construct the log file path for testing logs.
    log_dir = os.path.join(main_path, 'logs')
    testing_log_path = os.path.join(log_dir, 'testing_logs')
    log_file = os.path.join(testing_log_path, 'testing.log')
    
    current_time = time.time()
    # Check if we need to re-read the file (once every CACHE_DURATION seconds)
    if _cached_testing_log is None or (current_time - _last_testing_log_time) > CACHE_DURATION:
        try:
            with open(log_file, 'r') as f:
                _cached_testing_log = f.read()
        except Exception as e:
            _cached_testing_log = f"Error reading log file: {e}"
        _last_testing_log_time = current_time

    return _cached_testing_log

# --- Train Page ---
@app.route("/train", methods=['GET', 'POST'])
def train_page():
    if request.method == 'POST':
        model_name = request.form.get("model_name")
        weights = request.form.get("weights")
        selected_bands = request.form.get("selected_bands")
        selected_dataset = request.form.get("selected_dataset")
        test_variable = request.form.get("test_variable", "False")
        
        # Compute the main experiment path using your initialize_paths function.
        main_path = initialize_paths(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)
        
        session['main_path'] = main_path
        session['train_params'] = {
            'model_name': model_name,
            'weights': weights,
            'selected_bands': selected_bands,
            'selected_dataset': selected_dataset
        }
        
        # Build command to launch trainer.py using parent_dir
        trainer_script = os.path.join(parent_dir, "trainer.py")
        cmd = ["python", trainer_script, model_name, weights, selected_bands, selected_dataset, test_variable, main_path]
        subprocess.Popen(cmd, cwd=parent_dir)

        return render_template("train_status.html", message=f"Training for {model_name} has started.")
    
    return render_template("train.html", 
                           models=MODEL_OPTIONS, 
                           weights_options=["None", "DEFAULT"],
                           band_options=["all_bands", "rgb_bands", "rgb_nir_bands", "rgb_swir_bands", "rgb_nir_swir_bands"],
                           dataset_options=["100%_BigEarthNet", "50%_BigEarthNet", "10%_BigEarthNet", "5%_BigEarthNet", "1%_BigEarthNet", "0.5%_BigEarthNet"],
                           test_options=["False", "True"])


# --- Test Page ---
@app.route("/test", methods=['GET', 'POST'])
def test_page():
    if request.method == 'POST':
        # Extract form parameters
        model_name = request.form.get("model_name")
        weights = request.form.get("weights")
        selected_bands = request.form.get("selected_bands")
        selected_dataset = request.form.get("selected_dataset")
        checkpoint_path = request.form.get("checkpoint_path")  

        # Compute additional parameters based on selected_bands
        if selected_bands == "all_bands":
            in_channels = len(DatasetConfig.all_bands)
            bands = DatasetConfig.all_bands
        elif selected_bands == "rgb_bands":
            in_channels = len(DatasetConfig.rgb_bands)
            bands = DatasetConfig.rgb_bands
        elif selected_bands == "rgb_nir_bands":
            in_channels = len(DatasetConfig.rgb_nir_bands)
            bands = DatasetConfig.rgb_nir_bands
        elif selected_bands == "rgb_swir_bands":
            in_channels = len(DatasetConfig.rgb_swir_bands)
            bands = DatasetConfig.rgb_swir_bands
        elif selected_bands == "rgb_nir_swir_bands":
            in_channels = len(DatasetConfig.rgb_nir_swir_bands)
            bands = DatasetConfig.rgb_nir_swir_bands
        else:
            in_channels = 3  
            bands = DatasetConfig.rgb_bands

        num = str(extract_number(selected_dataset))
        dataset_dir = DatasetConfig.dataset_paths[num]
        metadata_path = DatasetConfig.metadata_paths[num]
        metadata_csv = pd.read_csv(metadata_path)

        class_weights, class_weights_array = calculate_class_weights(metadata_csv)
        class_weights = class_weights_array

        # Build command for tester.py.
        tester_script = os.path.join(parent_dir, "tester.py")
        cmd = [
            "python", tester_script,
            model_name, 
            weights, 
            selected_dataset, 
            checkpoint_path, 
            str(in_channels),
            json.dumps(class_weights.tolist()),
            metadata_path, 
            dataset_dir, 
            json.dumps(bands)
        ]
        subprocess.Popen(cmd, cwd=parent_dir)
        return render_template("test_status.html", message=f"Testing for {model_name} has started.")
    # On GET, display testing form
    return render_template("test.html", 
                           models=MODEL_OPTIONS,
                           weights_options=["None", "DEFAULT"],
                           band_options=["all_bands", "rgb_bands", "rgb_nir_bands", "rgb_swir_bands", "rgb_nir_swir_bands"],
                           dataset_options=["100%_BigEarthNet", "50%_BigEarthNet", "10%_BigEarthNet", "5%_BigEarthNet", "1%_BigEarthNet", "0.5%_BigEarthNet"])


EXPERIMENTS_DIR = r"C:\Users\isaac\Desktop\experiments"
def parse_experiment_folder(folder_name):
    parts = folder_name.split('_')
    if len(parts) == 7:
        model = parts[0]
        weights = parts[1]
        bands = parts[2] + "_" + parts[3]
        dataset = parts[4] + "_" + parts[5]
        epochs = parts[6]
    elif len(parts) == 5:
        model, weights, bands, dataset, epochs = parts
    else:
        # Fallback: if the naming is unexpected, return the folder name and empty details.
        model = folder_name
        weights = ""
        bands = ""
        dataset = ""
        epochs = ""
    return {"model": model, "weights": weights, "bands": bands, "dataset": dataset, "epochs": epochs}

@app.route("/experiments")
def experiments_overview():
    experiments = []
    if os.path.exists(EXPERIMENTS_DIR):
        for d in os.listdir(EXPERIMENTS_DIR):
            full_path = os.path.join(EXPERIMENTS_DIR, d)
            if os.path.isdir(full_path):
                parsed = parse_experiment_folder(d)
                parsed['folder_name'] = d  # store the full folder name if needed
                experiments.append(parsed)
    return render_template("experiments_overview.html", experiments=experiments)

@app.route("/experiment_file/<experiment_name>/<path:filename>")
def experiment_file(experiment_name, filename):
    experiment_path = os.path.join(EXPERIMENTS_DIR, experiment_name)
    return send_from_directory(experiment_path, filename)

@app.route("/experiment/<experiment_name>")
def experiment_detail(experiment_name):
    experiment_path = os.path.join(EXPERIMENTS_DIR, experiment_name)
    if not os.path.exists(experiment_path):
        return "Experiment not found", 404

    results = {}
    results_path = os.path.join(experiment_path, "results")
    if os.path.exists(results_path):
        # Define the expected subfolder names - note "visualizations" now uses a "z"
        expected_dirs = ["gradcam_visualisations", "tensorboard_graphs", "visualizations"]
        for ed in expected_dirs:
            ed_path = os.path.join(results_path, ed)
            if os.path.exists(ed_path) and os.path.isdir(ed_path):
                # Get the file list—even if empty
                results[ed] = os.listdir(ed_path)
            else:
                results[ed] = []  # Ensure key exists even if folder is missing

        standalone_files = []
        for item in os.listdir(results_path):
            item_path = os.path.join(results_path, item)
            if not os.path.isdir(item_path) and item not in expected_dirs:
                standalone_files.append(item)
        if standalone_files:
            results["files"] = standalone_files

    # --- Metrics ---
    metrics = {}
    metric_files = [
        "best_metrics.json",
        "best_test_metrics.json",
        "test_per_class_metrics_Sequential.json",
        "train_per_class_metrics_Sequential.json",
        "val_per_class_metrics_Sequential.json"
    ]
    for mf in metric_files:
        mf_path = os.path.join(results_path, mf)
        if os.path.exists(mf_path):
            try:
                with open(mf_path, 'r') as f:
                    metrics[mf] = json.load(f)
            except Exception as e:
                metrics[mf] = {"error": str(e)}

    return render_template("experiment_detail.html",
                           experiment_name=experiment_name,
                           results=results,
                           metrics=metrics)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
