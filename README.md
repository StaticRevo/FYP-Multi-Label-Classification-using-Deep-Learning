# Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery
![Image](https://github.com/user-attachments/assets/6684f6aa-fbad-4860-a31d-10a0e79a8195)

## Overview
This project is my Final Year Project for the course of Software Development. It aims to classify land cover types from satellite imagery using Convolutional Neural Networks (CNNs). By making use of deep learning techniques, the goal is to accurately **identify and categorise** the various types of land cover such as forests, urban areas and pasture simultaneously. This classification supports various application such as environmental monitoring, urban planning and disaster management.

Satellite imagery can help us find solutions to the growing number of environmental problems that humans face today. It allows us to not only get a bird’s eye view of what’s around us, but also uncovers parts of the world that are rarely seen. Tapping into the potential of categorizing land cover and land use around the world means that humans can more efficiently make use of natural resources, hopefully lowering cases of waste and deprivation. However, despite its potential to be incredibly useful, satellite data is massive and complex, requiring sophisticated analysis to make sense of it.

This project aims to address this issue by developing a CNN model capable of classifying land cover types in satellite imagery. This capability can benefit a variety of stakeholders, including conservationists, urban planners, and environmental scientists, by helping them **survey and identify patterns in land use**. This allows for the detection of natural areas under threat or the identification of regions suitable for urban development.

## Project Goals
- Develop a Custom CNN that is capable of accurately classify multiple land types based on Sentinel-2 multispectral satellite imagery.
- Utilise State-Of-The-Art (SOTA) models to leverage existing knowledge and serve as benchmarks for comparison
- Implement Data preprocessing and augmentation techniques to handle datasets such as BigEarthNet
- Incorporate various visualisation techniques to further ensure transparency and trust in the model's prediction
- Create a web-based interface to enable user-friendly interaction with the classification system, supporting training, testing, real-time predictions, and interactive visualisations of model outputs.

## Getting Started
Follow these steps to set up and run the project locally.
### Prerequisites
- **Python 3.11+** – Ensure you have Python installed. [Download Python](https://www.python.org/downloads/).
- **Git** – Required to clone the repository. [Download Git](https://git-scm.com/downloads).
- **CUDA (Optional)** – If you plan to run the model on a GPU, ensure CUDA is installed along with the necessary drivers.
- **Anaconda** - To install all the requirements from the project you need to create an anaconda environment. [Download Anaconda](https://www.anaconda.com/products/distribution)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery.git
   cd Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery
   
2. **Create and activate conda environment from [environment.yml](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/tree/main/environment.yml)**
   ```bash
   conda env create -f environment.yml

3. **Activate the Conda environment**
   ```bash
   conda activate Fyp311

4. **Install any additional or missing dependencies from [requirements.txt](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/tree/main/requirements.txt)**
   ```bash
   pip install -r requirements.txt


## Configuration Setup
Before running either the command-line interface (`main.py`) or the web application (`app.py`), you must update the configuration file: [`config.py`](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/tree/main/FYPProjectMultiSpectral/config/config.py).

This file contains critical settings such as:
- **Experiment paths:** Where model outputs and logs will be stored.
- **Dataset paths:** Locations of the BigEarthNet dataset.
- **Metadata paths:** Locations of metadata files for the dataset.
Update these paths to match your local environment to ensure the project runs correctly.

## Dataset Setup
![Image](https://github.com/user-attachments/assets/b60ae138-de33-40fd-8720-722f39fe80df)

The dataset used within the project is **BigEarthNet-S2** which is a large-scale multi-label remote sensing dataset consisting of: **590,326 patches**, **19 land cover types** and **12 spectral bands at 10m,20m and 60m per pixel resolution** Official Source: [BigEarthNet](https://bigearth.net/)

To automatically download and pre-process the dataset run:

      python FYPProjectMultiSpectral/preprocessing/data_preprocessing.py

The script will:
- **Download** the dataset from the official website.
- **Extract** all necessary files.
- **Clean and organise** the dataset by removing noisy samples, ensuring consistent resolution, and converting it into a suitable deep learning format.
- **Set up stratified subsets** at 0.5%, 1%, 5%, 10%, and 50%.


## Running the Project - Command Line (Training and Testing Models)
To run the project, simply run:

     python main.py

This will launch the model selection interface, allowing users to configure and run experiments based on their desired model, dataset percentage, band combinations, and training options.

After running the script users can:
- Select a **Model** from: CustomModel, ResNet18, ResNet50, VGG16, VGG19, DenseNet121, Swin-Transformer and ViT-Transformer
- Choose a **Dataset percentage**: 100%, 50%, 10%, 5%, 1%, 0.5%
- Pick a **Band Combination**: All_Bands, RGB_Bands, RGB_NIR_Bands, RGB_SWIR_Bands, RGB_NIR_SWIR_Bands
- Specify **Weights**: Pre-trained or Not
- Select **Training Option**: Train Only, Train and Test, Test Only

Users are also encouraged to experiment with different hyperparameters through the [config.py](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/blob/main/FYPProjectMultiSpectral/config/config.py) file. - It is important to note that the experiment paths, dataset and metadata paths need to be updated through 'config/config.py'

## Running the Web Application
For a more interactive experience with additional features such as prediction and inference, you can run the web application by navigating to the web folder and executing:
    
    cd FYPProjectMultiSpectral/web
      python app.py

This will start a Flask-based web server (by default at http://127.0.0.1:5000) that provides:
- **Training and Testing:** Configure and run training/testing experiments through a web interface with direct logging capabilities.
- **Prediction:** Upload Sentinel-2 imagery or fetch patches from an interactive map to classify land cover types using trained models.
- **Inference and Comparison:** Compare model performance across experiments with detailed metrics and visualisations.
- **Data Exploration:** Interactive charts and visualizations of the BigEarthNet dataset.
- **Experiment Management**: View, filter, and analyze past experiments with detailed logs, metrics, and visualizations.
- **Model Visualization:** Explore model architectures and Grad-CAM heatmaps for interpretability.

The web application extends the functionality of main.py by providing a user-friendly interface and additional capabilities for real-time inference and visualization.

## Experiment Tracking and Reproducibility
The project follows automated experiment logging in a structured directory format:

      experiments/
      ├── results/                         # Stores evaluation metrics, logs, and visualizations
      │   ├── tensorboard_graphs/          # TensorBoard visualizations saved as images    
      │   ├── visualizations/              # Confusion matrices and Label Co-occurrence images
      │   ├── gradcam_visualizations/      # Grad-CAM heatmaps
      │   ├── aggregated_metrics.txt       # Stores testing aggregated metrics  
      │   ├── per_category_metrics.txt     # Stores testing per-class metrics   
      │   ├── best_metrics.json            # Stores best validation metrics
      │   ├── train_per_class_metrics.json # Stores the per-class metrics for training
      │   ├── val_per_class_metrics.json   # Stores the per-class metrics for validation
      │   ├── predictions.npz              # Model predictions for analysis
      │   ├── activations.pdf              # Activations of the model
      ├── checkpoints/                     # Stores trained models
      │   ├── final.pth                    # Final model after all epochs
      ├── logs/                            # Logs related to model testing and evaluation
      │   ├── lightning_logs/              # Log file managed by PyTorch Lightning
      │   ├── testing_logs/                # Logs related to model testing and evaluation
      │       ├── testing.txt              # Main log file for test runs
      │   ├── training_logs/               # Logs related to model training
      │       ├── training.txt             # Main log file for training runs
      ├── architecture.txt                 # Description of the model architecture used in the experiment
      ├── hyperparameters.txt              # Text file listing major hyperparameters for reproducibility

Such a structured format ensures that evaluation can be performed efficiently. Besides that to ensure conistent results the project implements **Fixed Random Seeds**, **Logged Model HyperParameters** and also **Command-line Arguments for Custom Runs**

To reproduce an experiment run:

    python trainer.py <model_name> <weights> <selected_bands> <selected_dataset> <enable_test>

## Evaluation and Performance Metrics
The models are evaluated using a combination of aggregated and per-class metrics to ensure a detailed performance assessment. The following are the metrics that were computed during training, validation and testing:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **F2-Score**
- **Hamming Loss**
- **One Error**
- **Mean Average Precision (mAP)**

Beyond the above standard metrics, the project also makes use of additional Evaluation techniques to provide even deeper insights on the models performance such as:
- **Confusion Matrics (Per-Class and Aggregated)**
- **ROC and AUC Curves**
- **Label Co-occurrence Analysis**
- **Grad-CAM and Activation Maps**

The results are stored within experiment/results folder.

Note: While *torchmetrics* was used during training for internal monitoring, *sklearn* was used for final evaluation and is reflected in the shared metrics and report.

## Final Experiment Results (10% BigEarthNet Subset)

The directory `FYPProjectMultiSpectral/models/final_experiment_results/` contains the results of experiments conducted on the **10% subset** of the BigEarthNet dataset. Each model (e.g., `ResNet50`, `CustomModelV6`, etc.) has its own folder with the following structure:

      model_name/
      ├── logs/                  # Contains Logs related to model testing and evaluation
      ├── checkpoints/           # Contains the final trained model (`last.ckpt`)
      ├── metrics/               # Includes aggregated and per-class evaluation metrics 
      ├── visualizations/        # Includes sample batch predictions, confusion matrices and tensorboard graphs
      ├── architecture.txt       # Describes the model's architecture and configuration
      ├── test_predictions.npz   # Contains predicted and actual labels for the test set


This layout ensures consistent tracking of model performance and interpretability. It is important to note that the results in this repository are not **raw outputs** directly produced by the code. Instead, they have been organised to enhance readability and reduce confusion during analysis.

> **Note:**  
> Due to GitHub's [file size restrictions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and issues encountered with Git LFS, the following model checkpoints have **not been included** in this repository:

- **ResNet18**
- **ResNet50**
- **ResNet101**
- **EfficientNet_V2**
- **Vision Transformer (ViT)**
- **Swin Transformer**
- **VGG16**
- **VGG19**

These models exceeded GitHub’s 100MB file size limit or caused LFS bandwidth issues. Their results can be provided externally upon request.

## Code Quality and Readability

Utmost care was taken to follow clean coding practices throughout the project. The codebase is structured in a **modular way**, with each component responsible for a specific task. Comments follow an **imperative tone** and are **concise and descriptive**, aiming to enhance readability and make the code easier to understand and maintain. This consistent style should support any future development, debugging, and collaboration.

## Contact

For questions, feedback, or access to full experiment results and model checkpoints not included in this repository due to GitHub size limitations, feel free to reach out:

**Name:** Isaac Attard  
**Email:** isaacattard@hotmail.com / isaac.attard.22@um.edu.mt
**GitHub:** [StaticRevo](https://github.com/StaticRevo)

Alternatively, open an issue on the repository if it's project-related.

