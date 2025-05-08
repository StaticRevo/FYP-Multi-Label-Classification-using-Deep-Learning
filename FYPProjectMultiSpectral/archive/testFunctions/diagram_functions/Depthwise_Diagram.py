# Third-party imports
from graphviz import Digraph
import os

# Initialize the DepthwiseSeparableConv diagram
dot = Digraph(
    comment='DepthwiseSeparableConv Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true',  
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'dpi': '300',  
        'label': 'DepthwiseSeparableConv Module',
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

# DepthwiseSeparableConv structure
dot.node('DSC_Input', 'Input\n[B, in_channels, H, W]', fillcolor='#b3d9ff')
dot.node('DSC_Depthwise', 'Conv2d\nin_channels -> in_channels\nkernel_size=kernel_size, stride=stride\npadding=padding, dilation=dilation\ngroups=in_channels, bias, padding_mode\n[B, in_channels, H\', W\']', fillcolor='#b3d9ff')
dot.node('DSC_Pointwise', 'Conv2d\nin_channels -> out_channels\nkernel_size=1, stride=1, padding=0\ndilation=dilation, groups=1\nbias, padding_mode\n[B, out_channels, H\', W\']', fillcolor='#b3d9ff')
dot.node('DSC_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  # Dummy for skip connection
dot.node('DSC_Downsample', 'Downsample (optional)\nConv2d\nin_channels -> out_channels\nkernel_size=1, stride=stride, bias=False\nBatchNorm2d\n[B, out_channels, H\', W\']', fillcolor='#b3d9ff', style='filled,dashed')
dot.node('DSC_Add', '+\nResidual\n[out_channels]', shape='circle', fillcolor='#ffcccc', width='0.7', fixedsize='false', fontsize='12')  

# Edges for main path
dot.edges([
    ('DSC_Input', 'DSC_Depthwise'),
    ('DSC_Depthwise', 'DSC_Pointwise'),
    ('DSC_Pointwise', 'DSC_Add')
])

# Skip connection edge
dot.edge('DSC_Input', 'DSC_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('DSC_Skip_Dummy', 'DSC_Downsample', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('DSC_Downsample', 'DSC_Add', style='dashed', color='#ff3333', penwidth='2', label='Residual Connection')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/DepthwiseSeparableConv_Diagram'
dot.render(output_path, view=False)