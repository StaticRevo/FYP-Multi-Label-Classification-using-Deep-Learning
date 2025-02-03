# Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery
![lcc_global_2048](https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery/assets/116385849/66458441-3032-439f-81a4-75b43a13d21e)

## Overview
This project is my Final Year Project for the course of Software Development. It aims to classify land cover types from satellite imagery using Convolutional Neural Networks (CNNs). By making use of deep learning techniques, the goal is to accurately **identify and categorise** the various types of land cover such as forests, urban areas and pasture simultanously. This classification supports various application such as evnvironmental monitoring, urban plannning and distaster management.

Satellite imagery can help us find solutions to the growing number of environmental problems that humans face today. It allows us to not only get a bird’s eye view of what’s around us, but also uncovers parts of the world that are rarely seen. Tapping into the potential of categorizing land cover and land use around the world means that humans can more efficiently make use of natural resources, hopefully lowering cases of waste and deprivation. However, despite its potential to be incredibly useful, satellite data is massive and complex, requiring sophisticated analysis to make sense of it.

This project aims to address this issue by developing a CNN model capable of classifying land cover types in satellite imagery. This capability can benefit a variety of stakeholders, including conservationists, urban planners, and environmental scientists, by helping them survey and identify patterns in land use. This allows for the detection of natural areas under threat or the identification of regions suitable for urban development.

## Project Goals
- Develop a Custom CNN that is capable of accurately classify multiple land types based on Sentinel-2 multispectral satellite imagery.
- Utilise pre-trained models to leverage existing knowledge and serve as benchmarks for comparison
- Implement Data preprocessing and augmentation techniques to handle datasets such as BigEarthNet
- Incorporate various visualisation techniques to further ensure transparency and trust in the model's prediction

## Getting Started
Follow these steps to set up and run the project locally.
### Prerequisites
- **Python 3.11+** – Ensure you have Python installed. [Download Python](https://www.python.org/downloads/).
- **Git** – Required to clone the repository. [Download Git](https://git-scm.com/downloads).
- **CUDA (Optional)** – If you plan to run the model on a GPU, ensure CUDA is installed along with the necessary drivers.
- **Anaconda** - To install all the requirements from the project you need to create an anaconda enviroment. [Download Anaconda](https://www.anaconda.com/products/distribution)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery.git
   cd Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery

2. **Create and activate conda environment from [requirements.txt](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/blob/main/requirements.txt)**
   ```bash
   conda env create -f enviroment.yml

3. **Activate the Conda enviroment**
   ```bash
   conda activate Fyp311

### Dataset Setup
The dataset used within the project is **BigEarthNet-S2** which is a large-scale multi-label remote sensing dataset consisting of 590,326 patches, 19 land cover types and 12 spectral bands at 10m,20m and 60m per pixel resolution.

Official Source: [BigEarthNet](https://bigearth.net/)

By running the [data_preprocessing.py](https://github.com/StaticRevo/FYP-Multi-Label-Classification-using-Deep-Learning/blob/main/FYPProjectMultiSpectral/preprocessing/data_preprocessing.py) script within the preprocessing folder, the script will
- Download the Dataset
- Extract All Necessary Files
- Clean and Organise the Dataset
### Running the Project
