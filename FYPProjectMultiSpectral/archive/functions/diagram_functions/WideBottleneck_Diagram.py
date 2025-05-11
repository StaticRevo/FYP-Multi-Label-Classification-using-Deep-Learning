# Third-party imports
from graphviz import Digraph
import os

# Initialize the WideBottleneck diagram
dot = Digraph(
    comment='WideBottleneck Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true', 
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'dpi': '300',  
        'label': 'WideBottleneck Module',
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

# WideBottleneck structure
dot.node('WB_Input', 'Input\n[B, in_channels, H, W]', fillcolor='#b3d9ff')
dot.node('WB_Conv1', 'Conv2d\nin_channels -> out_channels\nkernel_size=1, bias=False\n[B, out_channels, H, W]', fillcolor='#b3d9ff')
dot.node('WB_BN1', 'BatchNorm2d\n[B, out_channels, H, W]', fillcolor='#b3d9ff')
dot.node('WB_ReLU1', 'ReLU\ninplace=True\n[B, out_channels, H, W]', fillcolor='#b3d9ff')
dot.node('WB_Conv2', 'Conv2d\nout_channels -> out_channels * widen_factor\nkernel_size=3, stride=stride, padding=1, bias=False\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')
dot.node('WB_BN2', 'BatchNorm2d\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')
dot.node('WB_ReLU2', 'ReLU\ninplace=True\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')
dot.node('WB_Conv3', 'Conv2d\nout_channels * widen_factor -> out_channels * widen_factor\nkernel_size=1, bias=False\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')
dot.node('WB_BN3', 'BatchNorm2d\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')
dot.node('WB_DropPath', 'DropPath\ndropout_rt\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff', style='filled,dashed')
dot.node('WB_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  # Dummy for skip connection
dot.node('WB_Downsample', 'Downsample\n(optional)\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff', style='filled,dashed')
dot.node('WB_Add', '+\nResidual\n[out_channels * widen_factor]', shape='circle', fillcolor='#ffcccc', width='1.9', fixedsize='true', fontsize='10')  
dot.node('WB_ReLU3', 'ReLU\ninplace=True\n[B, out_channels * widen_factor, H\', W\']', fillcolor='#b3d9ff')

# Edges for main path
dot.edges([
    ('WB_Input', 'WB_Conv1'),
    ('WB_Conv1', 'WB_BN1'),
    ('WB_BN1', 'WB_ReLU1'),
    ('WB_ReLU1', 'WB_Conv2'),
    ('WB_Conv2', 'WB_BN2'),
    ('WB_BN2', 'WB_ReLU2'),
    ('WB_ReLU2', 'WB_Conv3'),
    ('WB_Conv3', 'WB_BN3'),
    ('WB_BN3', 'WB_DropPath'),
    ('WB_DropPath', 'WB_Add'),
    ('WB_Add', 'WB_ReLU3')
])

# Skip connection edge
dot.edge('WB_Input', 'WB_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('WB_Skip_Dummy', 'WB_Downsample', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('WB_Downsample', 'WB_Add', style='dashed', color='#ff3333', penwidth='2', label='Residual Connection')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/WideBottleneck_Diagram'
dot.render(output_path, view=False)