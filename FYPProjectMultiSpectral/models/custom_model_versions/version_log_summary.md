# Custom Models V1-V9 Change Log
This section documents the latest nine versions of custom CNN architectures developed for multi-label classification on satellite imagery from BigEarthNet. While numerous early prototypes and experimental iterations were tested during development, **Versions 1 to 9 represent the final documented models** that underwent structured evaluation and comparison.

Each version builds on the lessons of the previous, introducing changes such as spectral mixing, attention mechanisms, skip connections, multi-scale feature extraction, and dynamic fusion layers. The goal throughout was to improve classification accuracy and efficiency across complex LULC classification.

Below is a breakdown of each model version, including its architectural focus and core statistics.

## Version 1: Base Network
**This model serves as the foundational architecture with residual blocks and diverse attention mechanisms.**  
This base model includes four sequential blocks, each combining `Conv2d`, `BatchNorm2d`, and `ReLU`, followed by a `ResidualBlock`. Channel sizes progressively increase (32 -> 256). Each block integrates a distinct attention mechanism: `SpectralAttention`, `ECA`, `SE`, and `DualAttention`. The classifier consists of global pooling followed by a linear layer.
- **Parameter Count**: 1,975,770  
- **Forward/Backward Pass Size**: 41.48 MB  
- **Params Size**: 7.90 MB  

## Version 2: Spectral Efficiency Network
**Adds spectral mixing and replaces standard convolutions with depthwise separable convolutions.**  
This version introduces an initial spectral mixing layer to better capture inter-band relationships. It replaces standard convolutions with `DepthwiseSeparableConv` for efficiency, integrates `CoordinateAttention` in the early stages, and adds `MultiScaleBlock` modules in Blocks 2 and 3 for multi-scale representation.
- **Parameter Count**: 1,804,410  
- **Forward/Backward Pass Size**: 628.74 MB  
- **Params Size**: 7.22 MB

## Version 3: Structured Skip Network
**Transitions to a class-based design and integrates transformers and skip connections.**  
This version restructures the model with a custom `forward()` method. It enhances spectral mixing, introduces a skip connection from Block 1 to Block 3 (fused via a `Conv2d` adapter), and adds a `TransformerModule` after Block 3 to capture global context.
- **Parameter Count**: 8,517,006  
- **Forward/Backward Pass Size**: 59.22 MB  
- **Params Size**: 33.02 MB  

## Version 4: High-Capacity Bottleneck Network
**Switches to bottleneck blocks and removes the transformer for efficiency.**  
`Bottleneck` blocks replace residuals to increase depth and efficiency. The transformer is removed to simplify the architecture, and dilated convolutions (dilation=2) are added in Block 4 to increase receptive field. Channel depth scales up to 1024.
- **Parameter Count**: 12,650,898  
- **Forward/Backward Pass Size**: 97.85 MB  
- **Params Size**: 50.60 MB   

## Version 5: Wide Regularized Network
**Uses wide bottlenecks and adds dropout layers for better generalisation.**  
This update swaps `Bottleneck` for `WideBottleneck` blocks (widen_factor=4) to increase representational width. Dropout is added after attention layers and in the classifier. `CBAM` replaces `DualAttention` in Block 4, and spectral mixing becomes a two-step convolutional process.
- **Parameter Count**: 1,716,240  
- **Forward/Backward Pass Size**: 64.60 MB  
- **Params Size**: 6.86 MB  

## Version 6: Multi-Scale Enhanced Network
**Introduces `MultiScaleBlock` in place of dilated convolutions.**  
Refining Version 5, Block 4's dilated convolution is replaced with a `MultiScaleBlock` to better capture patterns across different spatial scales. Other components such as wide bottlenecks, `CBAM`, spectral mixing, and the skip connection are retained.
- **Parameter Count**: 2,048,528  
- **Forward/Backward Pass Size**: 66.90 MB  
- **Params Size**: 8.19 MB  

## Version 7: Multi-Skip Fusion Network
**Adds multiple skip connections and a learned fusion mechanism.**  
Building on Version 6, this model introduces skip connections from Block 1 -> 3, Block 2 -> 4, and Block 3 -> 4. A learned fusion layer (`Conv2d(336 -> 2)`) is used to dynamically weight and integrate features from these paths. Channels are also reduced for efficiency. This model served the starting point for a more efficient architecture, competing directly with WRN-B4-ECA  developed by Papoutsis et al.
- **Parameter Count**: 950,761  
- **Forward/Backward Pass Size**: 56.68 MB  
- **Params Size**: 3.80 MB 

## Version 8: Comprehensive Fusion Network
**Introduces a spectral skip connection and dual learned fusion layers.**  
This version adds a skip connection from the spectral mixing layer and introduces a second learned fusion stage (`Conv2d(928 -> 4)`) to integrate features from all skip paths and the final block. It retains `CBAM`, dropout, and wide bottlenecks.
- **Parameter Count**: 962,328  
- **Forward/Backward Pass Size**: 63.61 MB  
- **Params Size**: 3.85 MB  

## Version 9: Optimised Fusion Network
**Improves efficiency using grouped convolutions and normalised fusion weights.**  
The final version increases output channels from spectral mixing and replaces standard convolutions with grouped versions to reduce parameters. Fusion weights are normalised for stable training, and grouped convolutions are also used in the `MultiScaleBlock`.
- **Parameter Count**: 939,604  
- **Forward/Backward Pass Size**: 80.61 MB  
- **Params Size**: 3.76 MB 
