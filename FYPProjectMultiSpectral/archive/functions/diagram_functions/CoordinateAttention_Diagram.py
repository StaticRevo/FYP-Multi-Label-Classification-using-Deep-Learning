# Third-party imports
from graphviz import Digraph
import os

# Initialize the CoordinateAttention diagram
dot = Digraph(
    comment='CoordinateAttention Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true', 
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'label': 'CoordinateAttention Module',
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

# CoordinateAttention structure
dot.node('CA_Input', 'Input\n[B, C, H, W]', fillcolor='#cce6ff')
dot.node('CA_PoolH', 'Pool Height\nAdaptiveAvgPool2d\n[B, C, H, 1]', fillcolor='#cce6ff')
dot.node('CA_PoolW', 'Pool Width\nAdaptiveAvgPool2d\nPermute\n[B, C, W, 1]', fillcolor='#cce6ff')
dot.node('CA_Concat', 'Concat\n[B, C, H+W, 1]', fillcolor='#cce6ff')
dot.node('CA_Conv1', 'Conv2d\nC -> mid_channels\nkernel_size=1, bias=False\nBatchNorm2d\nReLU (inplace=True)', fillcolor='#cce6ff')
dot.node('CA_Split', 'Split\nH: [B, mid_channels, H, 1]\nW: [B, mid_channels, W, 1]', fillcolor='#cce6ff')
dot.node('CA_ConvH', 'Conv2d\nmid_channels -> C\nkernel_size=1, bias=False\nSigmoid\n[B, C, H, 1]', fillcolor='#cce6ff')
dot.node('CA_ConvW', 'Conv2d\nmid_channels -> C\nkernel_size=1, bias=False\nSigmoid\nPermute\n[B, C, 1, W]', fillcolor='#cce6ff')
dot.node('CA_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  # Dummy for skip connection
dot.node('CA_Mul', '×\nAttention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5')

# Edges for main paths
dot.edges([
    ('CA_Input', 'CA_PoolH'),
    ('CA_Input', 'CA_PoolW'),
    ('CA_PoolH', 'CA_Concat'),
    ('CA_PoolW', 'CA_Concat'),
    ('CA_Concat', 'CA_Conv1'),
    ('CA_Conv1', 'CA_Split'),
    ('CA_Split', 'CA_ConvH'),
    ('CA_Split', 'CA_ConvW'),
    ('CA_ConvH', 'CA_Mul'),
    ('CA_ConvW', 'CA_Mul')
])

# Skip connection edge
dot.edge('CA_Input', 'CA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('CA_Skip_Dummy', 'CA_Mul', style='dashed', color='#ff3333', penwidth='2', label='Spatial Attention')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/CoordinateAttention_Diagram'
dot.render(output_path, view=False)