# Custom Models Change Log

## Version 1: Base Network
**This model serves as the foundational architecture with residual blocks and diverse attention mechanisms as the starting point.**  
This initial model is a basic sequential convolutional network with four blocks, each using `Conv2d`, `BatchNorm2d`, `ReLU`, and `ResidualBlock`, with channels growing from 32 to 64, 128, and 256 via stride=2 downsampling, and featuring varied attention mechanisms (`SpectralAttention` in Block 1, `ECA` in Block 2, `SE` in Block 3, `DualAttention` in Block 4), concluding with a classifier (`AdaptiveAvgPool2d(1)` → `Flatten` → `Dropout` → `Linear(256 -> num_classes)`).

## Version 2: Spectral Efficiency Network
**This version adds spectral mixing and switches to depthwise separable convolutions for efficiency.**  
Building on Version 1, this version introduces spectral mixing with a lightweight `Conv2d(in_channels -> 16, kernel_size=1)` → `BatchNorm2d` → `ReLU` as an initial step, replaces standard `Conv2d` with `DepthwiseSeparableConv` across all blocks for computational efficiency, adds `MultiScaleBlock` in Blocks 2 and 3 for multi-scale feature extraction, and incorporates `CoordinateAttention` in Block 1 for spatial awareness, while maintaining the four-block structure and classifier.

## Version 3: Structured Skip Network
**This version shifts to a class-based design with a skip connection and transformer for enhanced feature integration.**  
Transitioning from Version 2, this model adopts a class-based structure with a custom `forward` method, modifies spectral mixing to `Conv2d(in_channels -> 32)` → `BatchNorm2d` → `GELU` → `MaxPool2d(stride=2)` for richer initial features, introduces a skip connection from Block 1 to Block 3 (adjusted via `Conv2d(64 -> 256)` and fused by addition), adds a `TransformerModule` after Block 3 for global context, increases depth to five blocks with channels progressing to 512, and updates the classifier to handle 512 input features.

## Version 4: High-Capacity Bottleneck Network
**This version adopts bottleneck blocks and removes the transformer, boosting capacity with dilation.**  
Evolving from Version 3, this version replaces `ResidualBlock` with `Bottleneck` blocks (e.g., 64 -> 128 in Block 1) to improve efficiency and capacity, removes the transformer to reduce complexity, adjusts spectral mixing to `Conv2d(in_channels -> 64)` → `BatchNorm2d` → `GELU` → `MaxPool2d`, introduces dilated convolution (dilation=2) in Block 4 for a larger receptive field, enhances the skip connection with `Conv2d(128 -> 512)` → `AvgPool2d`, increases channels up to 1024, and modifies the classifier with an additional `Conv2d(1024 -> 1024)` before pooling.

## Version 5: Wide Regularized Network
**This version introduces wide bottleneck blocks and dropout for improved generalization.**  
Advancing from Version 4, this model switches `Bottleneck` to `WideBottleneck` (widen_factor=4, e.g., 32 -> 64 in Block 1) for broader feature diversity, adds dropout layers after attention modules and in the classifier for regularization, replaces `DualAttention` in Block 4 with `CBAM` for combined channel and spatial attention, enriches spectral mixing with two convolutions (`Conv2d(in_channels -> 64)` → `BatchNorm2d` → `GELU` → `Conv2d(64 -> 32)` → `BatchNorm2d`), reduces Block 4 output to 512, and adjusts the classifier with `Conv2d(512 -> 256)` before pooling.

## Version 6: Multi-Scale Enhanced Network
**This version substitutes the dilated convolution with a multi-scale block in Block 4.**  
Refining Version 5, this version replaces the dilated `DepthwiseSeparableConv` in Block 4 with `MultiScaleBlock(256 -> 256)` for explicit multi-scale feature extraction, while retaining the spectral mixing (two `Conv2d` layers), wide bottleneck blocks (e.g., 32 -> 64, 64 -> 128, 128 -> 256, 256 -> 512), dropout layers, `CBAM` in Block 4, single skip connection from Block 1 to Block 3 (fused by addition), and classifier structure (`Conv2d(512 -> 256)` → `AdaptiveAvgPool2d` → `Dropout` → `Linear`).

## Version 7: Multi-Skip Fusion Network
**This version adds multiple skip connections and a learned fusion mechanism.**  
Progressing from Version 6, this model reduces channel sizes for efficiency (32 -> 48 in Block 1, 48 -> 96 in Block 2, 96 -> 168 in Block 3, 168 -> 232 in Block 4), expands skip connections to three (Block 1 to Block 3 via `48 -> 168`, Block 2 to Block 4 via `96 -> 232`, Block 3 to Block 4 via `168 -> 232`), introduces a learned fusion stage (`fusion_conv`, `336 -> 2`) to dynamically weight Block 3 and Block 1 skip features, keeps spectral mixing, wide bottleneck blocks, `MultiScaleBlock` in Block 4, `CBAM`, dropout, and adjusts the classifier to process 232 input features (`Conv2d(232 -> 128)`).

## Version 8: Comprehensive Fusion Network
**This version introduces a spectral skip and dual learned fusion for comprehensive feature integration.**  
Enhancing Version 7, this version adds `SpectralAttention(32)` to spectral mixing before `MaxPool2d` for refined early features, introduces a spectral skip connection (`32 -> 232`) from spectral mixing to a second fusion stage, implements dual learned fusion with Fusion 1 (`fusion_conv`, `336 -> 2`) weighting Block 3 and Block 1 skip, and Fusion 2 (`fusion_conv2`, `928 -> 4`) weighting Block 4, Block 2 skip, Block 3 skip, and spectral skip, while retaining the reduced channel sizes (48, 96, 168, 232), wide bottleneck blocks, `MultiScaleBlock`, `CBAM`, dropout, and classifier (`Conv2d(232 -> 128)`).