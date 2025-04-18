# Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery
![lcc_global_2048](https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery/assets/116385849/66458441-3032-439f-81a4-75b43a13d21e)

## Overview
This project is my Final Year Project for the course of Software Development. It aims to classify land cover types from satellite imagery using Convolutional Neural Networks (CNNs). By making use of deep learning techniques, the goal is to accurately **identify and categorise** the various types of land cover such as forests, urban areas and pasture simultanously. This classification supports various application such as environmental monitoring, urban planning and disaster management.

Satellite imagery can help us find solutions to the growing number of environmental problems that humans face today. It allows us to not only get a bird’s eye view of what’s around us, but also uncovers parts of the world that are rarely seen. Tapping into the potential of categorizing land cover and land use around the world means that humans can more efficiently make use of natural resources, hopefully lowering cases of waste and deprivation. However, despite its potential to be incredibly useful, satellite data is massive and complex, requiring sophisticated analysis to make sense of it.

This project aims to address this issue by developing a CNN model capable of classifying land cover types in satellite imagery. This capability can benefit a variety of stakeholders, including conservationists, urban planners, and environmental scientists, by helping them survey and identify patterns in land use. This allows for the detection of natural areas under threat or the identification of regions suitable for urban development.

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
   
2. **Create and activate conda environment from [requirements.txt](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/blob/main/requirements.txt)**
   ```bash
   conda env create -f environment.yml

3. **Activate the Conda environment**
   ```bash
   conda activate Fyp311

## Dataset Setup
![Image](https://github.com/user-attachments/assets/b60ae138-de33-40fd-8720-722f39fe80df)

The dataset used within the project is **BigEarthNet-S2** which is a large-scale multi-label remote sensing dataset consisting of: **590,326 patches**, **19 land cover types** and **12 spectral bands at 10m,20m and 60m per pixel resolution**

Official Source: [BigEarthNet](https://bigearth.net/)

To automatically download and pre-process the dataset run:

      python FYPProjectMultiSpectral/preprocessing/data_preprocessing.py

The Script will:
- **Download** the Dataset
- **Extract** All Necessary Files
- **Clean and Organise** the Dataset

## Configuration Setup
Before running either the command-line interface (main.py) or the web application (app.py), you must update the configuration paths in FYPProjectMultiSpectral/config/config.py. This file contains critical settings such as:
- **Experiment paths:** Where model outputs and logs will be stored.
- **Dataset paths:** Locations of the BigEarthNet dataset.
- **Metadata paths:** Locations of metadata files for the dataset.
Update these paths to match your local environment to ensure the project runs correctly.

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

Users are also encouraged to experiment with different hyperparamters through the [config.py](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/blob/main/FYPProjectMultiSpectral/config/config.py) file. - It is important to note that the experiment paths, dataset and metadata paths need to be updated through 'config/config.py'

## Running the Web Application
For a more interactive experience with additional features such as prediction and inference, you can run the web application by navigating to the web folder and executing:
    
    cd FYPProjectMultiSpectral/web
      python app.py

This will start a Flask-based web server (by default at http://127.0.0.1:5000) that provides:
- **Training and Testing:** Configure and run training/testing experiments through a web interface with direct logging capabilities.
- **Prediction:** Upload Sentinel-2 imagery or fetch patches from an interactive map to classify land cover types using trained models.
- **Inference and Comparison:** Compare model performance across experiments with detailed metrics and visualisations.
- **Data Exploration:** Interactive charts and visualizations of the BigEarthNet dataset.
- **Experiment Management**: View, filter, and analyze past experiments with detailed logs, metrics, and visualszations.
- **Model Visualization:** Explore model architectures and Grad-CAM heatmaps for interpretability.

The web application extends the functionality of main.py by providing a user-friendly interface and additional capabilities for real-time inference and visualization.

## Experiment Tracking and Reporoducability
The project follows automated experiment logging in a structured directory format:

      experiments/
      ├── results/                         # Stores evaluation metrics, logs, and visualizations
      │   ├── best_metrics.json            # Stores best validation metrics
      │   ├── best_test_metrics.json       # Stores best test metrics
      │   ├── train_per_class_metrics.json # Stores the per-class metrics for training
      │   ├── val_per_class_metrics.json   # Stores the per-class metrics for validation
      │   ├── test_per_class_metrics.json  # Stores the per-class metrics for testing
      │   ├── tensorboard_graphs/          # TensorBoard visualizations saved as images
      │   ├── predictions.npz              # Model predictions for analysis
      │   ├── visualizations/              # Confusion matrices and Label Co-occurance images
      │   ├── gradcam_visualizations/      # Grad-CAM heatmaps
      │   ├── activations.pdf/             # Activations of the model
      ├── checkpoints/                     # Stores trained models
      │   ├── best_acc.pth                 # Best model based on validation accuracy
      │   ├── best_loss.pth                # Best model based on validation loss
      │   ├── final.pth                    # Final model after all epochs
      ├── logs/                            # Lightning Logs 

Such a structured format ensures that evaluation can be performed efficiently. Besides that to ensure conistent results the project implementes **Fixed Random Seeds**, **Logged Model HyperParameters** and also **Command-line Arguments for Custom Runs**

To reproduce an experiment run:

    python train_runner.py <model_name> <weights> <band_combination> <dataset> <enable_test>

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
