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
- The final trained model checkpoint (regulated by **early stopping** and **ReduceLROnPlateau**)
- Aggregated evaluation metrics
- Per-class performance metrics
- A `.txt` file describing the model architecture

These results were used in the evaluation and comparative analysis chapter of the dissertation.
