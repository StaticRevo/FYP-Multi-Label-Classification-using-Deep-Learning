# Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery
## Overview
## Table Of Contents
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
The EuroSAT Sentinel-2 dataset is publicly available and can be accessed by anyone via the following the link:[EuroSAT](https://github.com/phelber/EuroSAT).

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

