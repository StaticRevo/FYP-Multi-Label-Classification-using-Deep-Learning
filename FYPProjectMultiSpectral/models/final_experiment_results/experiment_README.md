# Final Experiment Results

This folder contains the final set of trained models and their corresponding evaluation results used for analysis in the dissertation report.

## Experiment Configuration

All experiments were conducted on the **10% subset of BigEarthNet**, using all **12 Sentinel-2 spectral bands**, resized to **120x120** resolution.

### Dataset Settings
- Dataset subset: **10% BigEarthNet**
- Image size: **120x120**
- Number of classes: **19**
- Spectral bands used: **All 12 Sentinel-2 bands**
- Training/Validation/Testing split: **70% / 15% / 15%**
- Pre-trained weights: **None** (all models trained from scratch)

### Training Settings
- `num_epochs`    : 50  
- `batch_size`    : 256  
- `learning_rate` : 0.001  
- `weight_decay`  : 0.01  
- `device`        : cuda  
- Optimizer: `AdamW` (momentum not used)

### Learning Rate Scheduler
- Strategy: `ReduceLROnPlateau`  
- `lr_factor`     : 0.5  
- `lr_patience`   : 4  

### Early Stopping
- `patience`      : 10  

### Loss Function
- `CombinedFocalLossWithPosWeight`
- `focal_alpha`   : 0.5  
- `focal_gamma`   : 3.0  

### Dropout Usage
- **Standard models** (e.g., ResNet50, VGG, DenseNet): *Unmodified*, used as-is with their default architectures.
- **Custom models** (e.g., `CustomModelV6`, `CustomModelV9`): Employed **progressive dropout**, increasing dropout rates across layers (e.g. `0.1`, `0.15`, `0.2`) to improve regularization.

### Model Blocks (Custom Model Only)
- Bottleneck block expansion ratio: `2`
- Channel reduction in attention modules: `16`
- Spatial attention ratio: `8`
- Activation functions: Primarily `GELU` (used throughout most custom layers)

## Hardware & Runtime Context

Training was performed on the following local hardware:
- **GPU**: NVIDIA RTX 3050 (8GB)
- **CPU**: Intel Core i5-12400F

Due to hardware limitations, all models were trained sequentially and subject to early stopping criteria. No distributed or cloud-based training was used. Training time varied depending on the architecture and batch size.

## Folder Contents

Each subfolder includes:
- `checkpoints/` - Final model checkpoint(s) (only if < 100MB)
- `metrics/`: 
  - `aggregated_metrics.txt` – Micro/macro scores and overall stats  
  - `per_class_metrics.txt` – Per-class precision, recall, F1, F2, accuracy
- `logs/`:
  - `training/` – Training loop outputs  
  - `testing/` – Test time logs 
  - `lightning_logs/` – PyTorch Lightning formatted logs
- `visualizations/`:
  - `tensorboard_graphs/` – Training & validation metrics over epochs
  - `confusion_matrices_grid.png`
  - `cooccurrence_matrix.png`
- `architecture.txt` - Model layer-by-layer summary
- `predictions.npz` – Saved binary predictions and ground truth labels

## Notes on Metrics
- **Training and Validation Metrics**:  
  During training and validation, metrics such as F1, F2, Precision, Recall, Accuracy, Hamming Loss, and One Error were computed using **TorchMetrics** (logged via PyTorch Lightning). These were used primarily to monitor model learning trends and overfitting. They are visualised via TensorBoard graphs and shown in the `visualizations/tensorboard_graphs/` folder.

- **Testing Metrics**:  
  Final performance metrics, used for formal evaluation and comparison in the dissertation, were calculated using **scikit-learn**. These metrics include:
  - Aggregated metrics (micro/macro F1, F2, precision, recall)
  - Per-class precision, recall, F1, F2, accuracy
  - Hamming Loss, Subset Accuracy, One Error
  - Average Precision Score

- **Important Distinction**:  
  The metrics shown during training (TensorBoard curves) **do not reflect** the final evaluation results presented in the report. All reported results in the evaluation chapter are derived from the **sklearn-calculated testing metrics**.

## Disclaimer
Only selected model checkpoints are included to reduce file size. All logs and metrics have been retained to ensure reproducibility and to support comparative analysis.

For clarity and readability, the results presented here have been deliberately organized and summarized to avoid excessive clutter and emphasize key evaluation outcomes. The complete raw experimental outputs are preserved locally and are available upon request.