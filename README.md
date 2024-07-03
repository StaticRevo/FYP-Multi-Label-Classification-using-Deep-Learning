# Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery
![lcc_global_2048](https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery/assets/116385849/66458441-3032-439f-81a4-75b43a13d21e)

## Overview
This project is my Final Year Project for the course of Software Development. It aims to classify land cover types from satellite imagery using Convolutional Neural Networks (CNNs). By making use of deep learning techniques, the goal is to accurately **identify and categorise** the various types of land cover such as forests, urban areas and pasture. This classification supports various application such as evnvironmental monitoring, urban plannning and distaster management.

Satellite imagery can help us find solutions to the growing number of environmental problems that humans face today. It allows us to not only get a bird’s eye view of what’s around us, but also uncovers parts of the world that are rarely seen. Tapping into the potential of categorizing land cover and land use around the world means that humans can more efficiently make use of natural resources, hopefully lowering cases of waste and deprivation. However, despite its potential to be incredibly useful, satellite data is massive and complex, requiring sophisticated analysis to make sense of it.

This project aims to address this issue by developing a CNN model capable of classifying land cover types in satellite imagery. This capability can benefit a variety of stakeholders, including conservationists, urban planners, and environmental scientists, by helping them survey and identify patterns in land use. This allows for the detection of natural areas under threat or the identification of regions suitable for urban development.

## Table Of Contents
- Overview
- Project Goals
- Dataset
- Model Architecture
- Training
- Eveluation
- Results
- Usage and Installation
- License

## Project Goals
## Dataset: EuroSAT Sentinel-2
![EUROSAT](https://github.com/StaticRevo/Deep-Learning-Based-Land-Use-Classification-Using-Sentinel-2-Imagery/assets/116385849/139d7b76-b898-460e-93c1-13536c6c0726)
### Description
The EuroSAT Sentinel-2 dataset is a large collection of satellite images that were captured by the Sentinel-2 satellite, which is part of the European Space Agency's Compernicus Program. This dataset was created to facilitate reseach in remote sensing and computer vision especially in the context of land use classification as done in my project. The Sentinel-2 dataset was also created for training machine learning models. The dataset contains **27,000 labeled and geo-refenced images** accross **10 different classes** which cover **13 spectral bands** including Visible (RGB), Infrared (NIR) and shortwave infrared (SWIR) bands among many others. One of the major strengths of such dataset is that the images presented have a spatial resolution of 10 meters per pixel making the images high in resolution and assisting in providing specific and detailed information on Earth's surface. Another ket feature feature of the dataset is that the images have been **pre-processed and augmented** to ensure consistency and quality. Finally as mentioned already, the EuroSAT dataset includes 10 classe which cover multiple land cover types. This diversity was cruicial for training and evaluating land classification models.

### Data Structure
The Dataset is organised into 10 main folders which reesent the various land cover types around the world. Additionaliy for the purpose of crating and training a Convolutional Neural Network (CNN) another folder was created which contains three additional folders where created for **training, testing and validtion** of the CNN. Each of these added folders contain subfolders corresponding to the 10 land cover classes. The structure of my dataset is as follows:

**EuroSAT Dataset**

**datasets/eurosat-dataset/**
- AnnulCrop/
- Forest/
- HerbaceousVegetation/
- Highway/
- Industrial/
- Pasture/
- PermanentCrop/
- Residential/
- River/
- SeaLake/

**Additional Folders**

**datasets/eurosat-dataset/dataset-splits/test**
- AnnulCrop/
- Forest/
- HerbaceousVegetation/
- Highway/
- Industrial/
- Pasture/
- PermanentCrop/
- Residential/
- River/
- SeaLake/

**datasets/eurosat-dataset/dataset-splits/train**
- AnnulCrop/
- Forest/
- HerbaceousVegetation/
- Highway/
- Industrial/
- Pasture/
- PermanentCrop/
- Residential/
- River/
- SeaLake/

**datasets/eurosat-dataset/dataset-splits/valid**
- AnnulCrop/
- Forest/
- HerbaceousVegetation/
- Highway/
- Industrial/
- Pasture/
- PermanentCrop/
- Residential/
- River/
- SeaLake/

### Source
The EuroSAT Sentinel-2 dataset is publicly available and can be accessed by anyone via the following the link: [EuroSAT](https://github.com/phelber/EuroSAT). 

The dataset was created by Patrick Helber et al. and is associated with the 'EuroSAT: A Novel Dataset and Deep learning Benchmark for Land Use and Land Cover Classification' paper. The dataset is distributed under a permissive license that allws for **academic and research use**. 

## Model Architecture
### Overview
### Architecture Details
### Model Diagram

## Training 
### Training Process
### Training Code
### Training Results

## Eveluation 

## Conclusion 

