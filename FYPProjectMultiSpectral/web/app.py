# Standard library imports
import os
import sys
import time  
import json
import secrets
import time
import subprocess
from datetime import datetime
import glob

# Directory set-up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

# Third-party imports
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, jsonify
from markupsafe import Markup
from werkzeug.utils import secure_filename
import torch
import pandas as pd
import nbformat
from nbconvert import HTMLExporter
from sentinel import fetch_sentinel_patch 
import rasterio
import matplotlib.pyplot as plt

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from utils.file_utils import initialize_paths
from models.models import *
import models.models as models
from utils.data_utils import extract_number
from utils.model_utils import get_class_names
from web_helper import *

# -- Flask app setup --
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
model_classes = get_class_names(models)

# Model Options
MODEL_OPTIONS = get_class_names(models)
CLASS_WEIGHTS = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path)) # Precompute class weights 
EXPERIMENTS_DIR = DatasetConfig.experiment_path # Experiment directory
ARCHITECTURES_DIR = os.path.join(parent_dir, "models", "Architecture")

# !-- Routes --!
# -- Home Page --
@app.route("/")
def index():
    return render_template("index.html")

# --- Train Page ---
@app.route("/train", methods=['GET', 'POST'])
def train_page():
    if request.method == 'POST':
        # Extract form parameters
        model_name = request.form.get("model_name")
        weights = request.form.get("weights")
        selected_bands = request.form.get("selected_bands")
        selected_dataset = request.form.get("selected_dataset")
        test_variable = request.form.get("test_variable", "False")
        
        # Compute the main experiment path 
        main_path = initialize_paths(model_name, weights, selected_bands, selected_dataset, ModelConfig.num_epochs)
        
        session['main_path'] = main_path
        session['train_params'] = { # Store training parameters in session
            'model_name': model_name,
            'weights': weights,
            'selected_bands': selected_bands,
            'selected_dataset': selected_dataset
        }
        
        trainer_script = os.path.join(parent_dir, "trainer.py") # Build command to launch trainer.py using parent_dir
        cmd = ["python", trainer_script, model_name, weights, selected_bands, selected_dataset, test_variable, main_path]
        subprocess.Popen(cmd, cwd=parent_dir)

        return render_template("train_status.html", message=f"Training for {model_name} has started.")
    
    # GET: Render the training form with options
    return render_template("train.html", 
                           models=MODEL_OPTIONS, 
                           weights_options=["None", "DEFAULT"],
                           band_options=["all_bands", "all_imp_bands" ,"rgb_bands", "rgb_nir_bands", "rgb_swir_bands", "rgb_nir_swir_bands"],
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
    
    current_time = time.time() # Check if need to re-read the file
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
            cp_file = "last.ckpt"
        elif checkpoint_type == "best_acc":
            cp_file = "best_acc.ckpt"
        elif checkpoint_type == "best_loss":
            cp_file = "best_loss.ckpt"
        else:
            cp_file = "last.ckpt"  # default fallback
        checkpoint_path = os.path.join(checkpoint_dir, cp_file)
        session['main_path'] = main_experiment_path # Save the experiment path in session 

        # Parse the experiment folder name to extract details
        experiment_details = parse_experiment_folder(experiment)
        model_name = experiment_details["model"]
        weights = experiment_details["weights"]
        selected_bands = experiment_details["bands"]
        selected_dataset = experiment_details["dataset"]

        # Set in_channels and bands based on the band combination string
        in_channels, bands = get_channels_and_bands(selected_bands)

        # Get dataset-related parameters using the dataset percentage from the experiment name.
        num = str(extract_number(selected_dataset))
        dataset_dir = DatasetConfig.dataset_paths[num]
        metadata_path = DatasetConfig.metadata_paths[num]
        metadata_csv = pd.read_csv(metadata_path)
        class_weights = calculate_class_weights(metadata_csv)

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
        # GET: List only the untested experiments 
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
                               weights_options=["None", "DEFAULT"],  
                               band_options=["all_bands", "all_imp_bands", "rgb_bands", "rgb_nir_bands", "rgb_swir_bands", "rgb_nir_swir_bands"],
                               dataset_options=["100%_BigEarthNet", "50%_BigEarthNet", "10%_BigEarthNet", "5%_BigEarthNet", "1%_BigEarthNet", "0.5%_BigEarthNet"])

# --- Logs (for Testing) ---
@app.route('/logs_test')
def logs_test():
    global _cached_testing_log, _last_testing_log_time
    main_path = session.get('main_path')  # Retrieve the main experiment path from the session.
    if not main_path:
        return "No testing run information found in session."

    # Construct the log file path for testing logs
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
        selected_experiment = request.form.get("experiment")  # Get the selected experiment from the form
        if not files or files[0].filename == '':
            flash("No file selected.", "error")
            return redirect(request.url)
        
        # Parse experiment details.
        experiment_details = parse_experiment_folder(selected_experiment)
        model_name = experiment_details["model"]
        selected_bands_str = experiment_details["bands"]
        
        # Determine in_channels and the default bands list based on the experiment.
        in_channels, default_bands = get_channels_and_bands(selected_bands_str)
        
        # Single image prediction branch
        if len(files) == 1:
            file = files[0]
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".tif", ".tiff"]:
                file_path = convert_image_to_tiff(file_path)

            try: # Validate image channels.
                actual_channels = validate_image_channels(file_path, in_channels)
            except ValueError as ve:
                flash(str(ve), "error")
                experiments = get_experiments_list()  # Reload experiments list for the upload page
                return render_template('upload.html', experiments=experiments)
            
            # If the image has extra bands, prompt the user to select bands
            if actual_channels > in_channels:
                return render_template('select_bands.html', 
                                       filename=filename,
                                       file_path=file_path,
                                       available_bands=list(range(1, actual_channels + 1)),
                                       expected_count=in_channels,
                                       experiment=selected_experiment)
            
            # Otherwise, proceed with prediction using the default band selection
            bands = default_bands
            return process_prediction(file_path, filename, bands, selected_experiment)
            
        else:
            results_list = []
            model_instance = load_model_from_experiment(selected_experiment)
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in [".tif", ".tiff"]:
                        file_path = convert_image_to_tiff(file_path)
                        
                    try:
                        actual_channels = validate_image_channels(file_path, in_channels)
                    except ValueError as ve:
                        flash(f"Skipping {filename}: {ve}", "error")
                        continue
                    
                    if actual_channels > in_channels:
                        flash(f"Skipping {filename}: Has extra bands. Please upload files with the expected number of bands for batch processing.", "error")
                        continue
                    
                    rgb_url = create_rgb_visualization(file_path)
                    input_tensor = preprocess_tiff_image(file_path, selected_bands=default_bands)
                    predictions = predict_image_for_model(model_instance, input_tensor)
                    patch_id = os.path.splitext(filename)[0]
                    actual_labels = fetch_actual_labels(patch_id)
                    
                    results_list.append({
                        "filename": filename,
                        "predictions": predictions,
                        "rgb_url": rgb_url,
                        "actual_labels": actual_labels,
                        
                    })

            if not results_list:
                flash("No valid files were uploaded with the required channel count.", "error")
                experiments = get_experiments_list()
                return render_template('upload.html', experiments=experiments)
                
            return render_template("batch_result.html", results=results_list, selected_experiment=selected_experiment)
    else:
        experiments = get_experiments_list()
        return render_template('upload.html', experiments=experiments)

# --- Select Bands Page ---
@app.route("/select_bands", methods=['POST'])
def select_bands():
    file_path = request.form.get("file_path")
    filename = request.form.get("filename")
    selected_experiment = request.form.get("experiment")
    selected_bands = request.form.getlist("selected_bands") 
    
    # Validate that the number of selected bands is as expected.
    experiment_details = parse_experiment_folder(selected_experiment)
    _, expected_bands = get_channels_and_bands(experiment_details["bands"])
    expected_count = len(expected_bands)
    
    if len(selected_bands) != expected_count:
        flash(f"Please select exactly {expected_count} band(s).", "error")
        try:
            with rasterio.open(file_path) as src:
                actual_channels = src.count
        except Exception as e:
            flash("Error reading image: " + str(e), "error")
            return redirect(url_for('predict_page'))
        
        return render_template('select_bands.html', 
                               filename=filename,
                               file_path=file_path,
                               available_bands=list(range(1, actual_channels + 1)),
                               expected_count=expected_count,
                               experiment=selected_experiment)
    
    selected_bands = sorted([int(b) for b in selected_bands]) # Convert band selections to integers and sort them
    
    return process_prediction(file_path, filename, selected_bands, selected_experiment) # Proceed with prediction using the user-selected bands.

# --- Batch GradCAM Page ---
@app.route("/batch_gradcam")
def batch_gradcam():
    filename = request.args.get("filename")
    experiment = request.args.get("experiment")

    if not filename or not experiment:
        return "Missing filename or experiment parameter", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return f"File {filename} not found in uploads folder.", 404

    model_instance = load_model_from_experiment(experiment)
    input_tensor = preprocess_tiff_image(file_path)
    input_tensor = input_tensor.to(next(model_instance.parameters()).device)

    with torch.no_grad():
        output = model_instance(input_tensor)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    predicted_indices = [idx for idx, prob in enumerate(probs) if prob > 0.5]

    predictions = predict_image_for_model(model_instance, input_tensor)

    experiment_details = parse_experiment_folder(experiment)
    model_name = experiment_details["model"]  
    in_channels, bands = get_channels_and_bands(experiment_details["bands"])
    gradcam_results = generate_gradcam_for_single_image(
        model=model_instance,
        img_tensor=input_tensor,
        class_labels=DatasetConfig.class_labels,
        model_name=model_name,
        in_channels=in_channels,
        predicted_indices=predicted_indices
    )
    # Generate colour-coded Grad-CAM visualization
    gradcam_colourcoded = generate_colourcoded_gradcam(
        model_instance, input_tensor,
        class_labels=DatasetConfig.class_labels,
        model_name=model_name,
        in_channels=in_channels,
        predicted_indices=predicted_indices
    )

    patch_id = os.path.splitext(filename)[0]
    actual_labels = fetch_actual_labels(patch_id)

    # Save the original image used in GradCAM to disk for display
    original_img_url = save_tensor_as_image(input_tensor.squeeze(), in_channels=12)

    return render_template(
        "batch_gradcam_result.html",
        filename=filename,
        experiment=experiment,
        gradcam=gradcam_results,
        gradcam_colourcoded=gradcam_colourcoded,
        actual_labels=actual_labels,
        original_img_url=original_img_url,
        predictions=predictions  
    )

# --- Experiment Overview Page ---
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
        "test_per_class_metrics.json",
        "train_per_class_metrics.json",
        "val_per_class_metrics.json"
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

    # Load architecture file 
    architecture_content = None
    parsed = parse_experiment_folder(experiment_name)
    model_name_from_folder = parsed.get("model")
    if model_name_from_folder:
        architecture_file = os.path.join(experiment_path, f"{model_name_from_folder}.txt")
        if os.path.exists(architecture_file):
            try:
                with open(architecture_file, 'r', encoding='utf-8') as f:
                    architecture_content = f.read()
            except Exception as e:
                architecture_content = f"Error reading {model_name_from_folder}.txt: {e}"

    # Aggregated Metrics
    aggregated_metrics = None
    aggregated_path = os.path.join(experiment_path, "results", "aggregated_metrics.txt")
    if os.path.exists(aggregated_path):
        try:
            aggregated_data = {}
            with open(aggregated_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.lower().startswith("aggregated metrics"): # Skip empty lines or header lines (if any)
                        if ":" in line:
                            key, value = line.split(":", 1)
                            aggregated_data[key.strip()] = value.strip()
            aggregated_metrics = aggregated_data
        except Exception as e:
            aggregated_metrics = {"error": str(e)}
    

    return render_template("experiment_detail.html",
                           experiment_name=experiment_name,
                           results=results,
                           metrics=metrics,
                           hyperparams_content=hyperparams_content,
                           architecture_content=architecture_content,
                           aggregated_metrics=aggregated_metrics)

# --- Inference Page --- 
@app.route('/detailed_inference', methods=['GET', 'POST'])
def detailed_inference():
    if request.method == 'POST':
        selected_experiments = request.form.getlist('experiments')
        
        comparison_data = {}
        experiments_data = {}
        testing_comparison_data = {}
        
        # Loop through each selected experiment
        for exp in selected_experiments:
            # Load validation metrics from best_metrics.json
            metrics = load_experiment_metrics(exp)
            if 'best_metrics' in metrics:
                for metric, value in metrics['best_metrics'].items():
                    comparison_data.setdefault(metric, {})[exp] = value
            
            # Load testing metrics from best_test_metrics.json
            test_metrics_path = os.path.join(EXPERIMENTS_DIR, exp, "results", "best_test_metrics.json")
            if os.path.exists(test_metrics_path):
                try:
                    with open(test_metrics_path, 'r') as f:
                        test_metrics = json.load(f)
                    if 'best_metrics' in test_metrics:
                        for metric, value in test_metrics['best_metrics'].items():
                            testing_comparison_data.setdefault(metric, {})[exp] = value
                except Exception as e:
                    for metric in ['test_acc', 'test_loss', 'test_f1', 'test_f2', 
                                   'test_precision', 'test_recall', 'test_one_error', 
                                   'test_hamming_loss', 'test_avg_precision', 'one_error']:
                        testing_comparison_data.setdefault(metric, {})[exp] = f"Error: {e}"
            
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
            
            # Also load best_metrics.json contents into experiments_data
            best_metrics = None
            bm_path = os.path.join(results_path, "best_metrics.json")
            if os.path.exists(bm_path):
                try:
                    with open(bm_path, 'r') as f:
                        best_metrics = json.load(f)
                except Exception as e:
                    best_metrics = {"error": str(e)}
            
            # Load per-class metrics (for each experiment individually)
            per_class = None
            for json_file in ["test_per_class_metrics.json", "val_per_class_metrics.json"]:
                json_path = os.path.join(EXPERIMENTS_DIR, exp, "results", json_file)
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        per_class = json.load(f)
                    break  # Use test metrics if available otherwise fall back to validation
            
            # Load aggregated metrics
            aggregated_metrics = None
            agg_path = os.path.join(results_path, "aggregated_metrics.txt")
            if os.path.exists(agg_path):
                try:
                    aggregated_data = {}
                    with open(agg_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.lower().startswith("aggregated metrics"): # Skip empty lines or header lines
                                if ":" in line:
                                    key, value = line.split(":", 1)
                                    aggregated_data[key.strip()] = value.strip()
                    aggregated_metrics = aggregated_data
                except Exception as e:
                    aggregated_metrics = {"error": str(e)}

            # Save all experiment data
            experiments_data[exp] = {
                "best_metrics": best_metrics,
                "hyperparams": hyperparams,
                "visualizations": visualizations,
                "tensorboard_graphs": tb_graphs,
                "per_class_metrics": per_class,
                "aggregated_metrics": aggregated_metrics
            }
        
        agg_keys = set()
        for exp_data in experiments_data.values():
            if exp_data.get("aggregated_metrics"):
                agg_keys.update(exp_data["aggregated_metrics"].keys())
        agg_keys = sorted(agg_keys)

        if "val_subset_accuracy" in comparison_data:
            comparison_data.pop("val_subset_accuracy")
        
        # Define metrics that should be minimized 
        min_metrics = {"val_loss", "val_hamming_loss", "val_one_error", "training_time_sec"}

        agg_min_metrics = {"hamming_loss", "one_error", "avg_hamming_loss", "subset_loss", 
                  "ranking_loss", "one_error_macro", "one_error_micro", 
                  "hamming_loss_macro", "hamming_loss_micro"}
        
        # Compute best_models: for each validation metric, choose the best experiment 
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
        
        # Generate insights for each validation metric
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
        colour_classes = [
            "bg-primary text-white",
            "bg-secondary text-white",
            "bg-success text-white",
            "bg-danger text-white",
            "bg-warning text-dark",
            "bg-info text-dark",
            "bg-light text-dark",
            "bg-dark text-white",
        ]
        exp_colour_map = {}
        for i, exp in enumerate(selected_experiments):
            exp_colour_map[exp] = colour_classes[i % len(colour_classes)]
        
        return render_template(
            'detailed_inference.html',
            selected_experiments=selected_experiments,
            comparison_data=comparison_data,
            testing_comparison_data=testing_comparison_data,  
            best_models=best_models,
            experiments_data=experiments_data,
            exp_colour_map=exp_colour_map,
            observations=observations,
            per_class_metrics=None,
            agg_keys=agg_keys
        )
    else:
        # Retrieve all experiment folder names
        available_experiments = []
        if os.path.exists(EXPERIMENTS_DIR):
            for d in os.listdir(EXPERIMENTS_DIR):
                full_path = os.path.join(EXPERIMENTS_DIR, d)
                if os.path.isdir(full_path):
                    available_experiments.append(d)
        
        # Extract filter parameters from the query string
        filter_model = request.args.get('model', '').lower()
        filter_weights = request.args.get('weights', '').lower()
        filter_bands = request.args.get('bands', '').lower()
        filter_dataset = request.args.get('dataset', '').lower()
        filter_epochs = request.args.get('epochs', '').lower()
        
        # Function to check if an experiment meets the filter criteria
        def matches_filter(exp):
            parsed = parse_experiment_folder(exp)
            if filter_model and filter_model not in parsed.get('model', '').lower():
                return False
            if filter_weights and filter_weights not in parsed.get('weights', '').lower():
                return False
            if filter_bands and filter_bands not in parsed.get('bands', '').lower():
                return False
            if filter_dataset and filter_dataset not in parsed.get('dataset', '').lower():
                return False
            if filter_epochs and filter_epochs not in parsed.get('epochs', '').lower():
                return False
            return True
        
        # Apply filtering if any filter is set
        if any([filter_model, filter_weights, filter_bands, filter_dataset, filter_epochs]):
            available_experiments = [exp for exp in available_experiments if matches_filter(exp)]
        
        # Extract sorting parameters
        sort_by = request.args.get('sort_by', 'date_trained')
        order = request.args.get('order', 'asc')
        
        # Create a list of experiment details 
        experiments_details = []
        for exp in available_experiments:
            parsed = parse_experiment_folder(exp)
            full_path = os.path.join(EXPERIMENTS_DIR, exp)
            creation_time = os.path.getctime(full_path)
            parsed['timestamp'] = creation_time
            parsed['folder_name'] = exp
            experiments_details.append(parsed)
        
        # Apply sorting based on the selected field and order
        if sort_by == 'date_trained':
            experiments_details = sorted(
                experiments_details,
                key=lambda x: x['timestamp'],
                reverse=(order == 'desc')
            )
        elif sort_by == 'model':
            experiments_details = sorted(
                experiments_details,
                key=lambda x: x.get('model', '').lower(),
                reverse=(order == 'desc')
            )
        
        # Prepare a sorted list of experiment folder names to pass to the template
        sorted_experiments = [exp['folder_name'] for exp in experiments_details]
        
        return render_template('select_experiments.html', experiments=sorted_experiments)

# -- Bubble Chart Page --
@app.route("/bubble_chart")
def bubble_chart():
    include_models = request.args.getlist("models")  # Get the selected models from query parameters as a list 
    if include_models:
        include_models = [m.strip().lower() for m in include_models]
    else:
        include_models = None  # No filter—include all experiments

    experiments_data = []
    if os.path.exists(EXPERIMENTS_DIR):
        for d in os.listdir(EXPERIMENTS_DIR):
            exp_path = os.path.join(EXPERIMENTS_DIR, d)
            if os.path.isdir(exp_path):
                # Define paths for both files in the results folder
                results_dir = os.path.join(exp_path, "results")
                metrics_path = os.path.join(results_dir, "best_metrics.json")
                aggregated_metrics_path = os.path.join(results_dir, "aggregated_metrics.txt")

                f2_score = None # Initialize F2 score

                # Check for best_metrics.json first 
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    except Exception as e:
                        print(f"Error loading metrics for {d}: {e}")
                        continue

                    # Check if required keys exist in best_metrics.json
                    if ("best_metrics" in metrics and 
                        "val_f2" in metrics["best_metrics"] and 
                        "training_time_sec" in metrics and 
                        "model_size_MB" in metrics):
                        f2_score = metrics["best_metrics"]["val_f2"] # Default to val_f2 from best_metrics.json
                    else:
                        continue  # Skip if required keys are missing

                    # Override f2_score with f2_micro if aggregated_metrics.txt exists
                    if os.path.exists(aggregated_metrics_path):
                        try:
                            with open(aggregated_metrics_path, "r") as f:
                                lines = f.readlines()
                            # Parse the file for f2_micro
                            for line in lines:
                                if "f2_micro" in line:
                                    f2_score = float(line.split(":")[1].strip())
                                    break
                        except Exception as e:
                            print(f"Error loading aggregated metrics for {d}: {e}") # If parsing fails, stick with val_f2 from best_metrics.json
                            
                    # Process the experiment with the determined f2_score
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
                        "val_f2": f2_score,  # Use f2_micro if available, otherwise val_f2
                        "model_size_MB": metrics["model_size_MB"],
                        "arch_type": exp_details.get("model", "N/A")
                    })

    # Pass both the experiments data and the available model options to the template
    return render_template("bubble_chart.html", data=experiments_data, model_options=MODEL_OPTIONS)

# -- Architecture Page --
@app.route("/architectures")
def architectures():
    architectures_list = []
    if os.path.exists(ARCHITECTURES_DIR):
        for folder in os.listdir(ARCHITECTURES_DIR):
            folder_path = os.path.join(ARCHITECTURES_DIR, folder)
            if os.path.isdir(folder_path):
                files_data = []
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    content = None
                    is_text = False
                    if file.endswith('.txt'):
                        is_text = True
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as e:
                            content = f"Error reading file: {e}"
                    files_data.append({
                        "name": file,
                        "content": content,
                        "is_text": is_text
                    })
                architectures_list.append({
                    "name": folder,
                    "files": files_data
                })
    else:
        flash("Architectures directory not found.", "error")
    return render_template("architectures.html", architectures=architectures_list)

@app.route("/architecture_file/<architecture>/<filename>")
def architecture_file(architecture, filename):
    folder_path = os.path.join(ARCHITECTURES_DIR, architecture) # Build the full path to the file
    if not os.path.exists(os.path.join(folder_path, filename)):
        return f"File {filename} not found in {architecture} folder.", 404
    return send_from_directory(folder_path, filename)

# -- Visualize Model Page --
@app.route("/visualize_model")
def visualize_model():
    return render_template("visualize.html")

# -- Data Exploration Page --
@app.route("/data_exploration")
def data_exploration():
    notebook_path = os.path.join(parent_dir, "notebooks", "data_exploration.ipynb")
    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return f"Error reading notebook: {e}"

    html_exporter = HTMLExporter()
    (body, _) = html_exporter.from_notebook_node(nb)
    
    return render_template("notebook_view.html", notebook_content=Markup(body))

# -- Map Page --
@app.route("/map_page", methods=['GET', 'POST'])
def map_page():
    print("Map page accessed")
    experiments = get_experiments_list()  # Fetch the list of experiments 
    return render_template('map_page.html', experiments=experiments)

# -- Get Image from Map Page ---
@app.route('/get_image', methods=['POST'])
def get_image():
    lat = float(request.form['lat'])  # Get coordinates from the map click
    lon = float(request.form['lon'])

    # Generate unique filename with timestamp
    timestamp = int(time.time())
    tiff_path = os.path.join('static', 'images', f'multispectral_patch_{timestamp}.tif')

    # Fetch Sentinel-2 patch and save as TIFF
    data, bbox_coords = fetch_sentinel_patch(lat, lon, output_tiff=tiff_path)

    # Scale reflectance values to [0, 1] 
    data = data / 10000.0

    # Extract RGB bands (B04, B03, B02)
    red = data[:, :, 3]   # B04
    green = data[:, :, 2] # B03
    blue = data[:, :, 1]  # B02

    # Ensure float32 for processing
    red = red.astype(np.float32)
    green = green.astype(np.float32)
    blue = blue.astype(np.float32)

    # Percentile clipping for contrast enhancement (2nd and 98th percentiles)
    def clip_band(band, lower_percentile=2, upper_percentile=98):
        lower, upper = np.percentile(band, [lower_percentile, upper_percentile])
        return np.clip(band, lower, upper)

    red = clip_band(red)
    green = clip_band(green)
    blue = clip_band(blue)

    # Normalize to [0, 1] after clipping, preserving relative colour balance
    rgb = np.dstack((red, green, blue))
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:  # Avoid division by zero
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb = np.zeros_like(rgb)  # Fallback if image is uniform 

    image_path = os.path.join('static', 'images', f'rgb_patch_{timestamp}.png') # Save RGB as PNG
    plt.imsave(image_path, rgb)

    # Return image URL, bounds, and TIFF filename for the map
    bounds = [[bbox_coords[1], bbox_coords[0]], [bbox_coords[3], bbox_coords[2]]]
    return jsonify({
        'image_url': f'/static/images/rgb_patch_{timestamp}.png',
        'bounds': bounds,
        'tiff_file': f'multispectral_patch_{timestamp}.tif'
    })

# -- Predict from Map Page ---
@app.route('/predict_from_map', methods=['POST'])
def predict_from_map():
    selected_experiment = request.form['experiment']

    tiff_dir = os.path.join('static', 'images')
    tiff_files = glob.glob(os.path.join(tiff_dir, 'multispectral_patch_*.tif'))

    if not tiff_files:
        return jsonify({'error': 'No TIFF files found in static/images.'}), 404

    latest_tiff = max(tiff_files, key=os.path.getctime)
    file_path = latest_tiff
    filename = os.path.basename(latest_tiff)

    try:
        experiment_details = parse_experiment_folder(selected_experiment)
        model_name = experiment_details["model"]
        selected_bands_str = experiment_details["bands"]
        in_channels, default_bands = get_channels_and_bands(selected_bands_str)

        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".tif", ".tiff"]:
            raise ValueError("Latest file is not a TIFF.")

        actual_channels = validate_image_channels(file_path, in_channels)

        if actual_channels < in_channels:
            raise ValueError(f"Image has only {actual_channels} bands, but model requires {in_channels}.")
        elif actual_channels > in_channels:
            print(f"Warning: Image has {actual_channels} bands, model expects {in_channels}. Using first {in_channels} bands from default list: {default_bands[:in_channels]}")
            bands = default_bands[:in_channels]
        else:
            bands = default_bands

        rgb_url = create_rgb_visualization(file_path)
        input_tensor = preprocess_tiff_image(file_path, selected_bands=bands)
        model_instance = load_model_from_experiment(selected_experiment)

        input_tensor = input_tensor.to(next(model_instance.parameters()).device)

        # Get Predictions 
        preds = predict_image_for_model(model_instance, input_tensor)
        with torch.no_grad():
            output = model_instance(input_tensor)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()
        predicted_indices = [idx for idx, prob in enumerate(probs) if prob > 0.5]

        preds_list = preds  # preds is a list of dicts with float32 probabilities

        print(f"preds type: {type(preds)}, value: {preds}")

        gradcam = generate_gradcam_for_single_image(
            model_instance, input_tensor,
            class_labels=DatasetConfig.class_labels,
            model_name=model_name,
            in_channels=len(bands),
            predicted_indices=predicted_indices
        )
        gradcam_colourcoded = generate_colourcoded_gradcam(
            model_instance, input_tensor,
            class_labels=DatasetConfig.class_labels,
            model_name=model_name,
            in_channels=len(bands),
            predicted_indices=predicted_indices
        )

        # Helper function to convert NumPy types to JSON-serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):  # Handle float32 and float64
                return float(obj)
            if isinstance(obj, np.ndarray):  # Handle NumPy arrays
                return obj.tolist()
            if isinstance(obj, dict):  # Recursively process dictionaries
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):  # Recursively process lists
                return [convert_to_serializable(item) for item in obj]
            return obj

        # Prepare the response data
        response_data = {
            'success': True,
            'filename': filename,
            'predictions': {selected_experiment: preds_list},
            'rgb_url': rgb_url,
            'gradcam': gradcam,
            'gradcam_colourcoded_': gradcam_colourcoded,
            'experiment_details': experiment_details,
            'selected_experiment': selected_experiment
        }

        # Convert all non-serializable types
        serializable_response = convert_to_serializable(response_data)

        return jsonify(serializable_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)


