# Standard library imports
import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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

# Third-party imports
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from markupsafe import Markup
from werkzeug.utils import secure_filename
import torch
import pandas as pd
import nbformat
from nbconvert import HTMLExporter

# Local application imports
from config.config import DatasetConfig, ModelConfig, calculate_class_weights
from utils.file_utils import initialize_paths
from models.models import *
from utils.data_utils import extract_number
from web_helper import *

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
MODEL_OPTIONS = ["CustomModel", "ResNet18", "ResNet50", "VGG16", "VGG19", "DenseNet121", "EfficientNetB0", "EfficientNet_v2", "Vit-Transformer", "Swin-Transformer"]
CLASS_WEIGHTS = calculate_class_weights(pd.read_csv(DatasetConfig.metadata_path)) # Precompute class weights 
EXPERIMENTS_DIR = DatasetConfig.experiment_path # Experiment directory
ARCHITECTURES_DIR = os.path.join(parent_dir, "models", "Architecture")

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
        
        trainer_script = os.path.join(parent_dir, "trainer.py") # Build command to launch trainer.py using parent_dir
        cmd = ["python", trainer_script, model_name, weights, selected_bands, selected_dataset, test_variable, main_path]
        subprocess.Popen(cmd, cwd=parent_dir)

        return render_template("train_status.html", message=f"Training for {model_name} has started.")
    
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
            json.dumps(class_weights),
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
                experiments = get_experiments_list()  # Reload experiments list for the upload page.
                return render_template('upload.html', experiments=experiments)
            
            # If the image has extra bands, prompt the user to select bands.
            if actual_channels > in_channels:
                return render_template('select_bands.html', 
                                       filename=filename,
                                       file_path=file_path,
                                       available_bands=list(range(1, actual_channels + 1)),
                                       expected_count=in_channels,
                                       experiment=selected_experiment)
            
            # Otherwise, proceed with prediction using the default band selection.
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
                        "actual_labels": actual_labels
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

    patch_id = os.path.splitext(filename)[0]
    actual_labels = fetch_actual_labels(patch_id)

    # Save the original image used in GradCAM to disk for display
    original_img_url = save_tensor_as_image(input_tensor.squeeze(), in_channels=12)

    return render_template(
        "batch_gradcam_result.html",
        filename=filename,
        experiment=experiment,
        gradcam=gradcam_results,
        actual_labels=actual_labels,
        original_img_url=original_img_url  # pass it to the template
    )

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

    return render_template("experiment_detail.html",
                           experiment_name=experiment_name,
                           results=results,
                           metrics=metrics,
                           hyperparams_content=hyperparams_content,
                           architecture_content=architecture_content)

# --- Inference Page --- 
@app.route('/detailed_inference', methods=['GET', 'POST'])
def detailed_inference():
    if request.method == 'POST':
        selected_experiments = request.form.getlist('experiments')
        
        comparison_data = {}
        experiments_data = {}
        testing_comparison_data = {}
        
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
                    # If there's an error, set a default value or error message
                    for metric in ['test_acc', 'test_loss', 'test_f1', 'test_f2', 
                                   'test_precision', 'test_recall', 'test_one_error', 
                                   'test_hamming_loss', 'test_avg_precision']:
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
            
            experiments_data[exp] = {
                "best_metrics": best_metrics,
                "hyperparams": hyperparams,
                "visualizations": visualizations,
                "tensorboard_graphs": tb_graphs
            }
        
        # Optionally, remove any unwanted key from validation metrics
        if "val_subset_accuracy" in comparison_data:
            comparison_data.pop("val_subset_accuracy")
        
        # Define metrics that should be minimized 
        min_metrics = {"val_loss", "val_hamming_loss", "val_one_error"}
        
        # Compute best_models: for each validation metric, choose the best experiment (minimize or maximize as needed)
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
        
        # Generate natural-language observations (insights) for each validation metric
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
            testing_comparison_data=testing_comparison_data,  # Pass testing metrics
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

