# Custom Models Change Log

## Version 1: Base Network
**This model serves as the foundational architecture with residual blocks and diverse attention mechanisms as the starting point.**  
This initial model is a basic sequential convolutional network with four blocks, each using `Conv2d`, `BatchNorm2d`, `ReLU`, and `ResidualBlock`, with channels growing from 32 to 64, 128, and 256 via stride=2 downsampling, and featuring varied attention mechanisms (`SpectralAttention` in Block 1, `ECA` in Block 2, `SE` in Block 3, `DualAttention` in Block 4), concluding with a classifier (`AdaptiveAvgPool2d(1)` → `Flatten` → `Dropout` → `Linear(256 -> num_classes)`).  
- **Parameter Count**: 1,975,770  
- **Forward/Backward Pass Size**: 41.48 MB  
- **Params Size**: 7.90 MB  

## Version 2: Spectral Efficiency Network
**This version adds spectral mixing and switches to depthwise separable convolutions for efficiency.**  
Building on Version 1, this version introduces spectral mixing with a lightweight `Conv2d(in_channels -> 16, kernel_size=1)` → `BatchNorm2d` → `ReLU` as an initial step, replaces standard `Conv2d` with `DepthwiseSeparableConv` across all blocks for computational efficiency, adds `MultiScaleBlock` in Blocks 2 and 3 for multi-scale feature extraction, and incorporates `CoordinateAttention` in Block 1 for spatial awareness, while maintaining the four-block structure and classifier.  
- **Parameter Count**: 1,804,410  
- **Forward/Backward Pass Size**: 628.74 MB  
- **Params Size**: 7.22 MB  

## Version 3: Structured Skip Network
**This version shifts to a class-based design with a skip connection and transformer for enhanced feature integration.**  
Transitioning from Version 2, this model adopts a class-based structure with a custom `forward` method, modifies spectral mixing to `Conv2d(in_channels -> 32)` → `BatchNorm2d` → `GELU` → `MaxPool2d(stride=2)` for richer initial features, introduces a skip connection from Block 1 to Block 3 (adjusted via `Conv2d(64 -> 256)` and fused by addition), adds a `TransformerModule` after Block 3 for global context, increases depth to five blocks with channels progressing to 512, and updates the classifier to handle 512 input features.  
- **Parameter Count**: 8,517,006  
- **Forward/Backward Pass Size**: 59.22 MB  
- **Params Size**: 33.02 MB  

## Version 4: High-Capacity Bottleneck Network
**This version adopts bottleneck blocks and removes the transformer, boosting capacity with dilation.**  
Evolving from Version 3, this version replaces `ResidualBlock` with `Bottleneck` blocks (e.g., 64 -> 128 in Block 1) to improve efficiency and capacity, removes the transformer to reduce complexity, adjusts spectral mixing to `Conv2d(in_channels -> 64)` → `BatchNorm2d` → `GELU` → `MaxPool2d`, introduces dilated convolution (dilation=2) in Block 4 for a larger receptive field, enhances the skip connection with `Conv2d(128 -> 512)` → `AvgPool2d`, increases channels up to 1024, and modifies the classifier with an additional `Conv2d(1024 -> 1024)` before pooling.  
- **Parameter Count**: 12,650,898  
- **Forward/Backward Pass Size**: 97.85 MB  
- **Params Size**: 50.60 MB  

## Version 5: Wide Regularized Network
**This version introduces wide bottleneck blocks and dropout for improved generalization.**  
Advancing from Version 4, this model switches `Bottleneck` to `WideBottleneck` (widen_factor=4, e.g., 32 -> 64 in Block 1) for broader feature diversity, adds dropout layers after attention modules and in the classifier for regularization, replaces `DualAttention` in Block 4 with `CBAM` for combined channel and spatial attention, enriches spectral mixing with two convolutions (`Conv2d(in_channels -> 64)` → `BatchNorm2d` → `GELU` → `Conv2d(64 -> 32)` → `BatchNorm2d`), reduces Block 4 output to 512, and adjusts the classifier with `Conv2d(512 -> 256)` before pooling.  
- **Parameter Count**: 1,716,240  
- **Forward/Backward Pass Size**: 64.60 MB  
- **Params Size**: 6.86 MB  

## Version 6: Multi-Scale Enhanced Network
**This version substitutes the dilated convolution with a multi-scale block in Block 4.**  
Refining Version 5, this version replaces the dilated `DepthwiseSeparableConv` in Block 4 with `MultiScaleBlock(256 -> 256)` for explicit multi-scale feature extraction, while retaining the spectral mixing (two `Conv2d` layers), wide bottleneck blocks (e.g., 32 -> 64, 64 -> 128, 128 -> 256, 256 -> 512), dropout layers, `CBAM` in Block 4, single skip connection from Block 1 to Block 3 (fused by addition), and classifier structure (`Conv2d(512 -> 256)` → `AdaptiveAvgPool2d` → `Dropout` → `Linear`).  
- **Parameter Count**: 2,048,528  
- **Forward/Backward Pass Size**: 66.90 MB  
- **Params Size**: 8.19 MB  

## Version 7: Multi-Skip Fusion Network
**This version adds multiple skip connections and a learned fusion mechanism.**  
Progressing from Version 6, this model reduces channel sizes for efficiency (32 -> 48 in Block 1, 48 -> 96 in Block 2, 96 -> 168 in Block 3, 168 -> 232 in Block 4), expands skip connections to three (Block 1 to Block 3 via `48 -> 168`, Block 2 to Block 4 via `96 -> 232`, Block 3 to Block 4 via `168 -> 232`), introduces a learned fusion stage (`fusion_conv`, `336 -> 2`) to dynamically weight Block 3 and Block 1 skip features, keeps spectral mixing, wide bottleneck blocks, `MultiScaleBlock` in Block 4, `CBAM`, dropout, and adjusts the classifier to process 232 input features (`Conv2d(232 -> 128)`).  
- **Parameter Count**: 950,761  
- **Forward/Backward Pass Size**: 56.68 MB  
- **Params Size**: 3.80 MB  
- **Comparison with CustomWRNB4ECA**:  
  - **Parameters**: 950,761 (CustomModel) vs. 985,837 (CustomWRNB4ECA), which means that the CustomModel has slightly fewer parameters, ensuring a fair comparison while introducing advanced features.  
  - **Forward/Backward Pass Size**: 56.68 MB (CustomModel) vs. 76.61 MB (CustomWRNB4ECA). CustomModel is more memory-efficient, allowing for larger batch sizes during training.  
  - **Total Mult-Adds**: 0.67 GB (CustomModel) vs. 2.14 GB (CustomWRNB4ECA). CustomModel is significantly more computationally efficient, enabling faster training and inference.  
  - **Strengths**: CustomModel incorporates multi-scale features (`MultiScaleBlock`), diverse attention mechanisms (`SpectralAttention`, `CoordinateAttention`, `ECA`, `SE`, `CBAM`), and skip connections with learned fusion, addressing CustomWRNB4ECA’s lack of multi-scale processing, limited attention variety (only `ECA`), and absence of skip connections for feature integration.  

## Version 8: Comprehensive Fusion Network
**This version introduces a spectral skip and dual learned fusion for comprehensive feature integration.**  
Enhancing Version 7, this version adds `SpectralAttention(32)` to spectral mixing before `MaxPool2d` for refined early features, introduces a spectral skip connection (`32 -> 232`) from spectral mixing to a second fusion stage, implements dual learned fusion with Fusion 1 (`fusion_conv`, `336 -> 2`) weighting Block 3 and Block 1 skip, and Fusion 2 (`fusion_conv2`, `928 -> 4`) weighting Block 4, Block 2 skip, Block 3 skip, and spectral skip, while retaining the reduced channel sizes (48, 96, 168, 232), wide bottleneck blocks, `MultiScaleBlock`, `CBAM`, dropout, and classifier (`Conv2d(232 -> 128)`).  
- **Parameter Count**: 962,328  
- **Forward/Backward Pass Size**: 63.61 MB  
- **Params Size**: 3.85 MB  
- **Comparison with CustomWRNB4ECA**:  
  - **Parameters**: 962,328 (CustomModel) vs. 985,837 (CustomWRNB4ECA). Still very close in parameter count, maintaining a fair comparison.  
  - **Forward/Backward Pass Size**: 63.61 MB (CustomModel) vs. 76.61 MB (CustomWRNB4ECA). CustomModel remains more memory-efficient, though the gap narrows due to additional skip connections and fusion stages.  
  - **Total Mult-Adds**: 0.67 GB (CustomModel) vs. 2.14 GB (CustomWRNB4ECA). CustomModel retains its computational efficiency advantage.  
  - **Strengths**: CustomModel further enhances feature integration with a spectral skip and dual learned fusion, building on Version 7’s strengths (multi-scale features, diverse attention, skip connections). CustomWRNB4ECA lacks these advanced integration mechanisms, relying solely on `ECA` and a simpler residual structure.  

## Version 9: Optimised Fusion Network
**This version enhances efficiency with grouped convolutions and improves fusion with normalised weights.** 

Advancing from Version 8, this model refines spectral mixing by increasing output channels (`32 -> 36`) and introducing grouped convolutions (`groups=4`) in the second `Conv2d(64 -> 36)` for parameter efficiency (reducing params from 18,432 to 5,184 in this layer), slightly scales channel sizes across blocks (`48 -> 52` in Block 1, `96 -> 104` in Block 2, `168 -> 172` in Block 3, `232 -> 236` in Block 4) for richer feature representation, adds `BatchNorm2d` to all skip adapters (`52 -> 172`, `36 -> 236`, `104 -> 236`, `172 -> 236`) and fusion layers (`344 -> 2`, `944 -> 4`) for training stability, incorporates `groups=4` in Block 4’s `MultiScaleBlock` to reduce parameters while retaining multi-scale capability, and optimizes Fusion 2 by normalizing weights (`w_high / sum(weights)`) for balanced feature integration across Block 4, Block 2 skip, Block 3 skip, and spectral skip outputs. It retains the dual learned fusion structure (Fusion 1: `344 -> 2` weighting Block 3 and Block 1 skip, Fusion 2: `944 -> 4`), wide bottleneck blocks, diverse attention mechanisms (`SpectralAttention`, `CoordinateAttention`, `ECA`, `SE`, `CBAM`), and classifier (`Conv2d(236 -> 128)` → `AdaptiveAvgPool2d` → `Dropout` → `Linear(128 -> 19)`).
- **Parameter Count**: 939,604 (approximately 22,724 fewer than Version 8 due to grouped convolutions in spectral mixing and Block 4, despite added normalization)  
- **Forward/Backward Pass Size**: 80.61 MB (increased from Version 8’s 63.61 MB due to higher channel counts and additional normalization layers)  
- **Params Size**: 3.76 MB (reduced from Version 8’s 3.85 MB, reflecting parameter efficiency)  
- **Total Mult-Adds**: 479.63M (slightly higher than Version 8’s ~0.67 GB due to increased channels, but still computationally efficient)  
- **Comparison with CustomWRNB4ECA**:  
  - **Parameters**: 939,604 (CustomModelV9) vs. 985,837 (CustomWRNB4ECA). CustomModelV9 achieves a lower parameter count while enhancing capacity and efficiency, offering around 46,233 fewer parameters than CustomWRNB4ECA.  
  - **Forward/Backward Pass Size**: 80.61 MB (CustomModelV9) vs. 76.61 MB (CustomWRNB4ECA). CustomModelV9 uses slightly more memory due to increased channels and normalization, but remains competitive.  
  - **Total Mult-Adds**: 479.63M (CustomModelV9) vs. 2.14G (CustomWRNB4ECA). CustomModelV9 is significantly more computationally efficient, requiring up to 4.5x fewer multiply-add operations, enabling faster training and inference.  
  - **Strengths**: CustomModelV9 builds on Version 8’s dual fusion and spectral skip by introducing grouped convolutions for parameter efficiency (e.g., spectral mixing: 5,184 params vs. 18,432 in V8; Block 4 MultiScaleBlock optimized with `groups=4`), adds normalization for stability, and refines Fusion 2 with normalized weights for robust feature integration. Compared to CustomWRNB4ECA’s simpler `ECA`-only design with no multi-scale processing or advanced skip-connection fusion, V9 offers superior feature extraction, diverse attention mechanisms, and computational efficiency at a lower parameter cost.