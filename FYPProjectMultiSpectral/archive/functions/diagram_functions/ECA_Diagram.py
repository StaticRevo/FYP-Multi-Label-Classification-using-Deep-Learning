# Third-party imports
from graphviz import Digraph
import os

# Initialize the ECA diagram
dot = Digraph(
    comment='ECA Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true', 
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'label': 'Efficient Channel Attention (ECA) Module',
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

# ECA structure
dot.node('ECA_Input', 'Input\n[B, C, H, W]', fillcolor='#ffe6cc')
dot.node('ECA_Pool', 'AdaptiveAvgPool2d\n[B, C, 1, 1]', fillcolor='#ffe6cc')
dot.node('ECA_Reshape1', 'Reshape\n[B, 1, C]', fillcolor='#ffe6cc')
dot.node('ECA_Conv', 'Conv1d\nin_channels=1, out_channels=1\nkernel_size=k_size, bias=False\npadding=(k_size-1)//2\n[B, 1, C]', fillcolor='#ffe6cc')
dot.node('ECA_Reshape2', 'Reshape\n[B, C, 1, 1]', fillcolor='#ffe6cc')
dot.node('ECA_Sigmoid', 'Sigmoid\n[B, C, 1, 1]', fillcolor='#ffe6cc') 
dot.node('ECA_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  
dot.node('ECA_Mul', '×\nAttention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5') 

# Edges for main path
dot.edges([
    ('ECA_Input', 'ECA_Pool'),
    ('ECA_Pool', 'ECA_Reshape1'),
    ('ECA_Reshape1', 'ECA_Conv'),
    ('ECA_Conv', 'ECA_Reshape2'),
    ('ECA_Reshape2', 'ECA_Sigmoid'),
    ('ECA_Sigmoid', 'ECA_Mul')
])

# Skip connection edge
dot.edge('ECA_Input', 'ECA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('ECA_Skip_Dummy', 'ECA_Mul', style='dashed', color='#ff3333', penwidth='2', label='Channel-wise Attention')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/ECA_Diagram'
dot.render(output_path, view=False)