# Third-party imports
from graphviz import Digraph

# Initialize the main model diagram
dot = Digraph(
    comment='CustomModelV9 Main Architecture',
    format='png',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'false',
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'label': 'CustomModelV9 Main Architecture (~939k Parameters)',
        'labelloc': 't',
        'fontsize': '18',
        'fontcolor': '#333333',
    },
    node_attr={
        'shape': 'box',
        'style': 'filled,rounded',
        'fontcolor': '#333333',
        'fontname': 'Arial',
        'fontsize': '12',
    },
    edge_attr={
        'color': '#555555',
        'arrowsize': '0.8',
    }
)

# Main model structure
dot.node('Input', 'Input\n[B, 12, 120, 120]', fillcolor='#cce5ff')
dot.node('SpectralMixer', 'Spectral Mixer\nConv2d(12->64, 1x1)\nBatchNorm2d\nGELU\nConv2d(64->36, 3x3, groups=4)\nBatchNorm2d\nSpectralAttention\nMaxPool2d(2x2)\n[B, 36, 60, 60]', fillcolor='#cce5ff')
dot.node('Block1', 'Block 1\nDepthwiseSeparableConv(36->36)\nBatchNorm2d\nGELU\nWideBottleneck(36->52)\nSpectralAttention\nDropout\nCoordinateAttention\nDropout\n[B, 52, 60, 60]', fillcolor='#b3d9ff')
dot.node('Block2', 'Block 2\nDepthwiseSeparableConv(52->52, stride=2)\nBatchNorm2d\nGELU\nWideBottleneck(52->104)\nECA\nDropout\n[B, 104, 30, 30]', fillcolor='#b3d9ff')
dot.node('Block3', 'Block 3\nDepthwiseSeparableConv(104->104, stride=2)\nBatchNorm2d\nGELU\nWideBottleneck(104->172)\nSE\nDropout\n[B, 172, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapter', 'Skip Adapter\nConv2d(52->172, 1x1)\nBatchNorm2d\nAvgPool2d(4x4)\n[B, 172, 15, 15]', fillcolor='#b3d9ff')
dot.node('Fusion', 'Fusion 1\nConv2d(344->2, 1x1)\nBatchNorm2d\nSigmoid\nWeighted Sum\n[B, 172, 15, 15]', shape='box', fillcolor='#ffcccc')
dot.node('Block4', 'Block 4\nMultiScaleBlock(172->172)\nBatchNorm2d\nGELU\nWideBottleneck(172->236)\nCBAM\nDropout\n[B, 236, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapterSpectral', 'Skip Adapter Spectral\nConv2d(36->236, 1x1)\nBatchNorm2d\nAvgPool2d(4x4)\n[B, 236, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapterMid', 'Skip Adapter Mid\nConv2d(104->236, 1x1)\nBatchNorm2d\nAvgPool2d(2x2)\n[B, 236, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapterDeep', 'Skip Adapter Deep\nConv2d(172->236, 1x1)\nBatchNorm2d\n[B, 236, 15, 15]', fillcolor='#b3d9ff')
dot.node('Fusion2', 'Fusion 2\nConv2d(944->4, 1x1)\nBatchNorm2d\nSigmoid\nWeighted Sum\n[B, 236, 15, 15]', shape='box', fillcolor='#ffcccc')
dot.node('Classifier', 'Classifier\nECA\nConv2d(236->128, 1x1)\nBatchNorm2d\nGELU\nAdaptiveAvgPool2d(1)\nFlatten\nDropout\nLinear(128->num_classes)\n[B, num_classes]', fillcolor='#cce5ff')

# Dummy nodes for skip connection paths
dot.node('Dummy1', '', shape='point', width='0.01', style='invisible')
dot.node('Dummy2', '', shape='point', width='0.01', style='invisible')
dot.node('Dummy3', '', shape='point', width='0.01', style='invisible')
dot.node('Dummy4', '', shape='point', width='0.01', style='invisible')

# Edges for main structure
dot.edges([
    ('Input', 'SpectralMixer'),
    ('SpectralMixer', 'Block1'),
    ('Block1', 'Block2'),
    ('Block2', 'Block3'),
    ('Block3', 'Fusion'),
    ('Fusion', 'Block4'),
    ('Block4', 'Fusion2'),
    ('Fusion2', 'Classifier')
])

# Skip connection edges
dot.edge('Block1', 'Dummy1', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy1', 'SkipAdapter', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 1 to Fusion)')
dot.edge('SkipAdapter', 'Fusion', style='dashed', color='#ff3333', penwidth='2')

dot.edge('SpectralMixer', 'Dummy2', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy2', 'SkipAdapterSpectral', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Spectral to Fusion2)')
dot.edge('SkipAdapterSpectral', 'Fusion2', style='dashed', color='#ff3333', penwidth='2')

dot.edge('Block2', 'Dummy3', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy3', 'SkipAdapterMid', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 2 to Fusion2)')
dot.edge('SkipAdapterMid', 'Fusion2', style='dashed', color='#ff3333', penwidth='2')

dot.edge('Block3', 'Dummy4', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy4', 'SkipAdapterDeep', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 3 to Fusion2)')
dot.edge('SkipAdapterDeep', 'Fusion2', style='dashed', color='#ff3333', penwidth='2')

# Render and save the diagram
output_path = r'/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/CustomModelV9_Main_Diagram'
dot.render(output_path, view=False)