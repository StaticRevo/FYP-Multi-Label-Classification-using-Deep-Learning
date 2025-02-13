# Standard library imports
import os
import sys

# Set up directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

import json
import secrets
import time
import subprocess
from datetime import datetime
import uuid

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

# Model Options
MODEL_OPTIONS = ["custom_model", "ResNet18", "ResNet50", "VGG16", "VGG19", "DenseNet121", "EfficientNetB0", "EfficientNet_v2", "Vit-Transformer", "Swin-Transformer"]

# Precompute class weights 
class_weights, class_weights_array = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path))
CLASS_WEIGHTS = class_weights_array

# Experiment directory
EXPERIMENTS_DIR = r"C:\Users\isaac\Desktop\experiments"

# --- Helper Functions ---
def load_model_from_experiment(experiment_name):
    # Construct the checkpoint path from the experiment folder.
    checkpoint_path = os.path.join(EXPERIMENTS_DIR, experiment_name, "checkpoints", "final.ckpt")
    
    # Parse the experiment folder name to extract model details.
    parsed = parse_experiment_folder(experiment_name)
    model_name = parsed["model"]
    
    # Set in_channels as needed (here we use 12 by default; adjust if necessary)
    in_channels = 12  
    
    main_path = os.path.dirname(checkpoint_path)
    model_class, _ = get_model_class(model_name)
    if model_class is None:
        raise ValueError(f"Model class for {model_name} not found!")
    
    # Load the model using the checkpoint from the experiment.
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        class_weights=CLASS_WEIGHTS,
        num_classes=DatasetConfig.num_classes,
        in_channels=in_channels,
        model_weights=None,  # or any additional weights if needed
        main_path=main_path
    )
    model.eval()
    print(f"Model from experiment {experiment_name} loaded successfully.")
    return model

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
    
    # Generate a unique filename for each visualization
    out_filename = f"result_image_{uuid.uuid4().hex}.png"
    out_path = os.path.join(STATIC_FOLDER, out_filename)
    Image.fromarray(rgb_image).save(out_path)
    return url_for('static', filename=out_filename)

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

def parse_experiment_folder(folder_name):
    parts = folder_name.split('_')
    if len(parts) == 7:
        model = parts[0]
        weights = parts[1]
        bands = parts[2] + "_" + parts[3]
        dataset = parts[4] + "_" + parts[5]
        epochs = parts[6]
    else:
        if any(char.isdigit() for char in parts[-1]):
            if any(char.isdigit() for char in parts[-2]):
                epochs = parts[-2] + "_" + parts[-1]
                dataset = parts[-4] + "_" + parts[-3]
                remaining = parts[:-4]
            else:
                epochs = parts[-1]
                dataset = parts[-3] + "_" + parts[-2]
                remaining = parts[:-3]
        else:
            # Fallback if last part doesn't contain digits.
            epochs = parts[-1]
            dataset = parts[-3] + "_" + parts[-2]
            remaining = parts[:-3]

        if "None" in remaining:
            w_index = remaining.index("None")
            weights = remaining[w_index]
            model = "_".join(remaining[:w_index])  
            bands = "_".join(remaining[w_index+1:])  
        else:
            # If no "None" found, fallback to defaults:
            model = remaining[0]
            weights = ""
            bands = "_".join(remaining[1:])
    return {"model": model, "weights": weights, "bands": bands, "dataset": dataset, "epochs": epochs}

# -- Routes --
# -- Home Page --
@app.route("/")
def index():
    return render_template("index.html")

# --- Train Page ---
@app.route("/train", methods=['GET', 'POST'])
def train_page():
    if request.method == 'POST':
        model_name = request.form.get("model_name")
        weights = request.form.get("weights")
        selected_bands = request.form.get("selected_bands")
        selected_dataset = request.form.get("selected_dataset")
        test_variable = request.form.get("test_variable", "False")
        
        # Compute the main experiment path 
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

# --- Logs (for Training) ---
@app.route('/logs')
def logs():
    global _cached_training_log, _last_training_log_time
    main_path = session.get('main_path') # Retrieve the main experiment path from the session.
    if not main_path:
        return "No training run information found in session."

    # Construct the log file path for training logs.
    log_dir = os.path.join(main_path, 'logs')
    training_log_path = os.path.join(log_dir, 'training_logs')
    log_file = os.path.join(training_log_path, 'training.log')
    
    current_time = time.time() # Check if we need to re-read the file (once every CACHE_DURATION seconds)
    if _cached_training_log is None or (current_time - _last_training_log_time) > CACHE_DURATION:
        try:
            with open(log_file, 'r') as f:
                _cached_training_log = f.read()
        except Exception as e:
            _cached_training_log = f"Error reading log file: {e}"
        _last_training_log_time = current_time

    return _cached_training_log

# --- Test Page ---
@app.route("/test", methods=['GET', 'POST'])
def test_page():
    if request.method == 'POST':
        experiment = request.form.get("experiment") # Extract form parameters
        checkpoint_type = request.form.get("checkpoint_type")
        
        # Build the main experiment path and determine checkpoint file based on type
        main_experiment_path = os.path.join(EXPERIMENTS_DIR, experiment)
        checkpoint_dir = os.path.join(main_experiment_path, "checkpoints")
        if checkpoint_type == "last":
            cp_file = "final.ckpt"
        elif checkpoint_type == "best_acc":
            cp_file = "best_acc.ckpt"
        elif checkpoint_type == "best_loss":
            cp_file = "best_loss.ckpt"
        else:
            cp_file = "final.ckpt"  # default fallback
        checkpoint_path = os.path.join(checkpoint_dir, cp_file)
        
        
        session['main_path'] = main_experiment_path # Save the experiment path in session 

        # Parse the experiment folder name to extract details
        experiment_details = parse_experiment_folder(experiment)
        model_name = experiment_details["model"]
        weights = experiment_details["weights"]
        selected_bands = experiment_details["bands"]
        selected_dataset = experiment_details["dataset"]

        # Set in_channels and bands based on the band combination string
        if selected_bands.lower() == "all_bands":
            in_channels = len(DatasetConfig.all_bands)
            bands = DatasetConfig.all_bands
        elif selected_bands.lower() == "rgb_bands":
            in_channels = len(DatasetConfig.rgb_bands)
            bands = DatasetConfig.rgb_bands
        elif selected_bands.lower() == "rgb_nir_bands":
            in_channels = len(DatasetConfig.rgb_nir_bands)
            bands = DatasetConfig.rgb_nir_bands
        elif selected_bands.lower() == "rgb_swir_bands":
            in_channels = len(DatasetConfig.rgb_swir_bands)
            bands = DatasetConfig.rgb_swir_bands
        elif selected_bands.lower() == "rgb_nir_swir_bands":
            in_channels = len(DatasetConfig.rgb_nir_swir_bands)
            bands = DatasetConfig.rgb_nir_swir_bands
        else:
            in_channels = 3  
            bands = DatasetConfig.rgb_bands

        # Get dataset-related parameters using the dataset percentage from the experiment name.
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
    else:
        # GET: List only the untested experiments (those without a 'visualizations' folder)
        untested_experiments = []
        if os.path.exists(EXPERIMENTS_DIR):
            for d in os.listdir(EXPERIMENTS_DIR):
                full_path = os.path.join(EXPERIMENTS_DIR, d)
                if os.path.isdir(full_path):
                    results_path = os.path.join(full_path, "results")
                    vis_path = os.path.join(results_path, "visualizations")
                    if not os.path.exists(vis_path):
                        untested_experiments.append(d)
        return render_template("test.html", 
                               untested_experiments=untested_experiments,
                               weights_options=["None", "DEFAULT"],  # no longer used, but you might keep them for consistency
                               band_options=["all_bands", "rgb_bands", "rgb_nir_bands", "rgb_swir_bands", "rgb_nir_swir_bands"],
                               dataset_options=["100%_BigEarthNet", "50%_BigEarthNet", "10%_BigEarthNet", "5%_BigEarthNet", "1%_BigEarthNet", "0.5%_BigEarthNet"])

# --- Logs (for Testing) ---
@app.route('/logs_test')
def logs_test():
    global _cached_testing_log, _last_testing_log_time
    main_path = session.get('main_path')  # Retrieve the main experiment path from the session.
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

# -- Predict Page --
@app.route("/predict", methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
 
        files = request.files.getlist('file')  # Get the list of uploaded files 
        selected_experiment = request.form.get("experiment") # Get the selected experiment from the form
        if not files or files[0].filename == '':
            return redirect(request.url)
        
        if len(files) == 1:
            file = files[0] # Single image prediction 
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            rgb_url = create_rgb_visualization(file_path)
            input_tensor = preprocess_tiff_image(file_path)
            
            model_instance = load_model_from_experiment(selected_experiment)
            preds = predict_image_for_model(model_instance, input_tensor)
            gradcam = generate_gradcam_for_single_image(model_instance, input_tensor, selected_experiment)
            
            patch_id = os.path.splitext(filename)[0] # Fetch actual labels from metadata 
            actual_labels = fetch_actual_labels(patch_id)
            
            # Parse experiment details from the folder name
            experiment_details = parse_experiment_folder(selected_experiment)
            
            return render_template('result.html',
                                   filename=filename,
                                   predictions={selected_experiment: preds},
                                   actual_labels=actual_labels,
                                   rgb_url=rgb_url,
                                   gradcam=gradcam,
                                   multiple_models=False,
                                   experiment_details=experiment_details)
        else:
            results_list = [] # Batch prediction 
            model_instance = load_model_from_experiment(selected_experiment) # Load the model once for the entire batch
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    rgb_url = create_rgb_visualization(file_path)
                    input_tensor = preprocess_tiff_image(file_path)
                    
                    predictions = predict_image_for_model(model_instance, input_tensor)
                    patch_id = os.path.splitext(filename)[0]
                    actual_labels = fetch_actual_labels(patch_id)
                    
                    results_list.append({
                        "filename": filename,
                        "predictions": predictions,
                        "rgb_url": rgb_url,
                        "actual_labels": actual_labels
                    })
            return render_template("batch_result.html", results=results_list)
    else:
        experiments = []
        if os.path.exists(EXPERIMENTS_DIR):
            for d in os.listdir(EXPERIMENTS_DIR):
                full_path = os.path.join(EXPERIMENTS_DIR, d)
                if os.path.isdir(full_path):
                    experiments.append(d)
        return render_template('upload.html', experiments=experiments)

@app.route("/experiments")
def experiments_overview():
    experiments = []
    if os.path.exists(EXPERIMENTS_DIR):
        for d in os.listdir(EXPERIMENTS_DIR):
            full_path = os.path.join(EXPERIMENTS_DIR, d)
            if os.path.isdir(full_path):
                parsed = parse_experiment_folder(d)
                parsed['folder_name'] = d  # store the full folder name 

                # Get creation time and store both timestamp and formatted date
                creation_time = os.path.getctime(full_path)
                parsed['timestamp'] = creation_time
                parsed['date_trained'] = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")

                experiments.append(parsed)

    # Get filter parameters from the query string.
    filter_model = request.args.get('model', '').lower()
    filter_weights = request.args.get('weights', '').lower()
    filter_bands = request.args.get('bands', '').lower()
    filter_dataset = request.args.get('dataset', '').lower()
    filter_epochs = request.args.get('epochs', '').lower()
    
    def matches_filter(exp):
        if filter_model and filter_model not in exp['model'].lower():
            return False
        if filter_weights and filter_weights not in exp['weights'].lower():
            return False
        if filter_bands and filter_bands not in exp['bands'].lower():
            return False
        if filter_dataset and filter_dataset not in exp['dataset'].lower():
            return False
        if filter_epochs and filter_epochs not in exp['epochs'].lower():
            return False
        return True
    
    # Apply filtering if any filter is set
    if any([filter_model, filter_weights, filter_bands, filter_dataset, filter_epochs]):
        experiments = [exp for exp in experiments if matches_filter(exp)]
    
    # Get sorting parameters
    sort_by = request.args.get('sort_by', 'date_trained')
    order = request.args.get('order', 'asc')
    
    # Apply sorting based on the chosen field and order
    if sort_by == 'date_trained':
        experiments = sorted(experiments, key=lambda x: x['timestamp'], reverse=(order == 'desc'))
    elif sort_by == 'model':
        experiments = sorted(experiments, key=lambda x: x['model'].lower(), reverse=(order == 'desc'))

    return render_template("experiments_overview.html", experiments=experiments)

# --- Experiment Detail Page ---
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
        # Define the expected subfolder names
        expected_dirs = ["gradcam_visualisations", "tensorboard_graphs", "visualizations"]
        for ed in expected_dirs:
            ed_path = os.path.join(results_path, ed)
            if os.path.exists(ed_path) and os.path.isdir(ed_path):
                results[ed] = os.listdir(ed_path) # Get the file list—even if empty
            else:
                results[ed] = []  # Ensure key exists even if folder is missing

        standalone_files = []
        for item in os.listdir(results_path):
            item_path = os.path.join(results_path, item)
            if not os.path.isdir(item_path) and item not in expected_dirs:
                standalone_files.append(item)
        if standalone_files:
            results["files"] = standalone_files

    # Metrics
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

    # Hyperparameters
    hyperparams_content = None
    hyperparams_path = os.path.join(experiment_path, "hyperparameters.txt")
    if os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams_content = f.read()
        except Exception as e:
            hyperparams_content = f"Error reading hyperparameters.txt: {e}"

    return render_template("experiment_detail.html",
                           experiment_name=experiment_name,
                           results=results,
                           metrics=metrics,
                           hyperparams_content=hyperparams_content 
                           )

# --- Inference Page ---
@app.route('/detailed_inference', methods=['GET', 'POST'])
def detailed_inference():
    if request.method == 'POST':
        selected_experiments = request.form.getlist('experiments')
        
        comparison_data = {}
        experiments_data = {}
        
        for exp in selected_experiments:
            # Load metrics from best_metrics.json
            metrics = load_experiment_metrics(exp)
            if 'best_metrics' in metrics:
                for metric, value in metrics['best_metrics'].items():
                    comparison_data.setdefault(metric, {})[exp] = value
            
            # Load hyperparameters
            exp_path = os.path.join(EXPERIMENTS_DIR, exp)
            hyperparams_path = os.path.join(exp_path, "hyperparameters.txt")
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'r') as f:
                    hyperparams = f.read()
            else:
                hyperparams = None
            
            # Load visualizations
            results_path = os.path.join(exp_path, "results")
            vis_path = os.path.join(results_path, "visualizations")
            visualizations = {}
            if os.path.exists(vis_path):
                confusion = ("results/visualizations/confusion_matrices_grid.png"
                             if os.path.exists(os.path.join(vis_path, "confusion_matrices_grid.png"))
                             else None)
                roc_auc = ("results/visualizations/roc_auc_curve.png"
                           if os.path.exists(os.path.join(vis_path, "roc_auc_curve.png"))
                           else None)
                cooccurrence = ("results/visualizations/cooccurrence_matrix.png"
                                if os.path.exists(os.path.join(vis_path, "cooccurrence_matrix.png"))
                                else None)
                visualizations = {
                    "confusion_matrices": confusion,
                    "roc_auc": roc_auc,
                    "cooccurrence": cooccurrence
                }
            else:
                visualizations = None
            
            # Load TensorBoard graphs
            tb_path = os.path.join(results_path, "tensorboard_graphs")
            if os.path.exists(tb_path):
                tb_graphs = os.listdir(tb_path)
            else:
                tb_graphs = []
            
            best_metrics = None # Load best_metrics.json contents
            bm_path = os.path.join(results_path, "best_metrics.json")
            if os.path.exists(bm_path):
                try:
                    with open(bm_path, 'r') as f:
                        best_metrics = json.load(f)
                except Exception as e:
                    best_metrics = {"error": str(e)}
            
            experiments_data[exp] = {
                "best_metrics": best_metrics,
                "hyperparams": hyperparams,
                "visualizations": visualizations,
                "tensorboard_graphs": tb_graphs
            }
        
        if "val_subset_accuracy" in comparison_data: # Exclude "val_subset_accuracy" from observations
            comparison_data.pop("val_subset_accuracy")
        
        # Define metrics that should be minimized 
        min_metrics = {"val_loss", "val_hamming_loss", "val_one_error"}
        
        # Compute best_models: for each metric, choose the best experiment (minimize or maximize as needed)
        best_models = {}
        for metric, values in comparison_data.items():
            valid_values = {k: v for k, v in values.items() if v is not None}
            if not valid_values:
                best_models[metric] = 'N/A'
            else:
                if metric in min_metrics:
                    best_models[metric] = min(valid_values, key=valid_values.get)
                else:
                    best_models[metric] = max(valid_values, key=valid_values.get)
        
        # Generate natural-language observations (insights) for each metric (except val_subset_accuracy)
        observations = []
        for metric_name, best_exp in best_models.items():
            if best_exp == 'N/A':
                continue
            best_value = comparison_data[metric_name][best_exp]
            
            if metric_name == "val_acc":
                observations.append(
                    f"{best_exp} has the highest validation accuracy ({best_value:.4f}), "
                    "indicating it generally predicts correctly more often than the others"
                )
            elif metric_name == "val_loss":
                observations.append(
                    f"{best_exp} has the lowest validation loss ({best_value:.4f}), "
                    "suggesting it fits the data with fewer overall errors"
                )
            elif metric_name == "val_f1":
                observations.append(
                    f"{best_exp} has the highest F1 score ({best_value:.4f}), "
                    "indicating a strong balance between precision and recall"
                )
            elif metric_name == "val_precision":
                observations.append(
                    f"{best_exp} has the highest precision ({best_value:.4f}), "
                    "meaning it avoids false positives effectively"
                )
            elif metric_name == "val_recall":
                observations.append(
                    f"{best_exp} has the highest recall ({best_value:.4f}), "
                    "meaning it successfully identifies more true positives"
                )
            elif metric_name == "val_hamming_loss":
                observations.append(
                    f"{best_exp} has the lowest Hamming loss ({best_value:.4f}), "
                    "indicating fewer label-wise errors"
                )
            elif metric_name == "val_one_error":
                observations.append(
                    f"{best_exp} has the lowest one-error rate ({best_value:.4f}), "
                    "meaning it has fewer top-1 misclassifications"
                )
            elif metric_name == "val_avg_precision":
                observations.append(
                    f"{best_exp} has the highest average precision ({best_value:.4f}), "
                    "indicating strong ranking performance"
                )
            elif metric_name == "val_f2":
                observations.append(
                    f"{best_exp} has the highest F2 score ({best_value:.4f}), "
                    "which places more emphasis on recall"
                )
        
        # Create colour mapping for experiments
        color_classes = [
            "bg-primary text-white",
            "bg-secondary text-white",
            "bg-success text-white",
            "bg-danger text-white",
            "bg-warning text-dark",
            "bg-info text-dark",
            "bg-light text-dark",
            "bg-dark text-white",
        ]
        exp_color_map = {}
        for i, exp in enumerate(selected_experiments):
            exp_color_map[exp] = color_classes[i % len(color_classes)]
        
        return render_template(
            'detailed_inference.html',
            selected_experiments=selected_experiments,
            comparison_data=comparison_data,
            best_models=best_models,
            experiments_data=experiments_data,
            exp_color_map=exp_color_map,
            observations=observations
        )
    else:
        available_experiments = [d for d in os.listdir(EXPERIMENTS_DIR)
                                 if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
        return render_template('select_experiments.html', experiments=available_experiments)

# -- Bubble Chart Page --
@app.route("/bubble_chart")
def bubble_chart():
    include_models = request.args.getlist("models") # Get the selected models from query parameters as a list (if any)
    if include_models:
        include_models = [m.strip().lower() for m in include_models]
    else:
        include_models = None  # No filter—include all experiments

    experiments_data = []
    if os.path.exists(EXPERIMENTS_DIR):
        for d in os.listdir(EXPERIMENTS_DIR):
            exp_path = os.path.join(EXPERIMENTS_DIR, d)
            if os.path.isdir(exp_path):
                # Look for the best_metrics.json file in the results folder
                metrics_path = os.path.join(exp_path, "results", "best_metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    except Exception as e:
                        print(f"Error loading metrics for {d}: {e}")
                        continue

                    # Check if required keys exist
                    if ("best_metrics" in metrics and 
                        "val_f2" in metrics["best_metrics"] and 
                        "training_time_sec" in metrics and 
                        "model_size_MB" in metrics):
                        
                        training_time_min = metrics["training_time_sec"] / 60.0
                        exp_details = parse_experiment_folder(d)
                        
                        # Filter based on selected models if provided
                        if include_models and exp_details.get("model", "").lower() not in include_models:
                            continue
                        
                        experiments_data.append({
                            "experiment": d,
                            "model": exp_details.get("model", "N/A"),
                            "training_time_sec": metrics["training_time_sec"],
                            "training_time_min": training_time_min,
                            "val_f2": metrics["best_metrics"]["val_f2"],
                            "model_size_MB": metrics["model_size_MB"],
                            "arch_type": exp_details.get("model", "N/A")
                        })
    # Pass both the experiments data and the available model options to the template.
    return render_template("bubble_chart.html", data=experiments_data, model_options=MODEL_OPTIONS)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
