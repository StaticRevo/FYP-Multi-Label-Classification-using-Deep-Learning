# Project Timeline and Development Overview
![Image](https://github.com/user-attachments/assets/1f876d6f-ef78-4eae-a43a-e5b9793f91de)

This document outlines the high-level development timeline and key stages involved in building the Final Year Project **"Deep Learning-Based Land Use Classification Using Sentinel-2 Imagery."** The project spanned over several months, evolving through a sequence of well-defined phases, from initial research to final system deployment.

## Research & Planning
- **Deep Paper Analysis:**  
  *June 2024 – August 2024* (150 hrs)  
  Conducted a comprehensive literature review on deep learning techniques, multi-label classification, and satellite imagery analysis. Identified key gaps and feasible methodologies.

## Data Setup
- **Data Download & Exploration:**  
  *September 2024* (80 hrs)  
  Downloaded BigEarthNet dataset using a Python script, explored patch structure, metadata, and label distribution. Identified class imbalance and label overlap issues.

- **Data Preparation Pipeline:**  
  *September – October 2024* (160 hrs)  
  Designed static and on-the-fly preprocessing:  
  - Removed noisy samples using metadata  
  - Resampled bands to 120×120  
  - Stacked into multi-spectral `.tif`  
  - Applied z-score normalisation  
  - Created stratified subsets (0.5% – 50%)  
  - Implemented transformation and normalisation classes

## System Development
- **Training Pipeline:**  
  *November – December 2024* (220 hrs)  
  Built modular PyTorch Lightning training script with:
  - Custom model/weights/band selection
  - Logging, checkpointing, metric tracking
  - Support for resume training
  - Integration with configuration files

- **SOTA Model Adaptation:**  
  *November 2024* (30 hrs)  
  Adapted ResNet, VGG, DenseNet, EfficientNet, Swin, and ViT to accept 12-band inputs. All trained using the same pipeline and from scratch for fair benchmarking.

- **Evaluation Pipeline:**  
  *December 2024 – January 2025* (200 hrs)  
  Developed full evaluation module:
  - Per-class & aggregated metrics 
  - Grad-CAM heatmaps
  - Confusion and co-occurrence matrices
  - Batch predictions, activations, ROC curves

- **Custom Model Architecture (CustomModelV9):**  
  *January – March 2025* (450 hrs)  
  Iteratively developed a custom CNN architecture:
  - Depthwise Separable Convs, Residual Blocks, WideBottlenecks  
  - Spectral, Coordinate, SE, ECA, and CBAM Attention  
  - Multi-scale blocks and attention-weighted fusion  

## Website Development
- **Web Interface (Flask):**  
  *March – April 2025* (120 hrs)  
  Built a fully-featured web system:
  - Train/Test via UI  
  - Upload & predict on patches (single/batch)  
  - Live map-based Sentinel-2 patch prediction  
  - Grad-CAM viewer  
  - Interactive inference dashboard with model comparison

## Iterative Enhancements
- *April – May 2025* (150 hrs)  
  - Rebalanced dataset splits (70/15/15)  
  - Added gradient clipping & logging improvements  
  - Expanded evaluation metrics (SKLearn)  
  - Refactored training/evaluation utilities for reusability  
  - Polished model architecture and visualisation tools

## Report & Documentation
- **Documentation and Report Writing:**  
  *September 2024 – May 2025* (300 hrs)  
  The report writing phase was carried out in parallel with system development. Each major chapter was aligned with progress in the pipeline:

  - **Introduction:** Drafted early to frame the motivation, goals, and scope of the project.
  - **Background & Literature Review:** Developed during and after deep paper analysis, covering key methodologies and existing research in multi-label classification and remote sensing.
  - **Specification & Design:** Initiated alongside the data preparation and model planning phases. Included system requirements, data flow diagrams, and architecture overviews.
  - **Implementation:** Written during hands-on development of the training, evaluation, and model pipelines. This chapter described the design decisions and integration efforts across all modules.
  - **Evaluation:** Added once testing pipelines and performance results were finalised. Included analysis of metrics, visualisations, and comparison with SOTA models.
  - **Future Work & Conclusion:** Drafted near the final stages, summarising achievements and proposing follow-up improvements.

  Throughout the process, emphasis was placed on clarity and consistency. Diagrams were refined across iterations, and figures such as confusion matrices, Grad-CAM overlays, and performance charts were generated directly from experimental logs. Regular proof-reading and restructuring ensured alignment between system functionality and report content.

  Additionally, appendices were used to document architectural iterations of the custom models, detailed design and visual breakdowns of attention mechanisms, training hyperparameters, web application interfaces, system execution workflows, dataset characteristics and distribution statistics, as well as extended evaluation outputs.








