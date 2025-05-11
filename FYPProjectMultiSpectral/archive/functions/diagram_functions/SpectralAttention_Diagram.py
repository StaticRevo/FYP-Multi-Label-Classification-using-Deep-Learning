# Third-party imports
from graphviz import Digraph
import os

# Initialize the SpectralAttention diagram
dot = Digraph(
    comment='SpectralAttention Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true',
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',
        'label': 'SpectralAttention Module',
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

# SpectralAttention structure
dot.node('SA_Input', 'Input\n[B, C, H, W]', fillcolor='#e6ccff')
dot.node('SA_Pool', 'GlobalAvgPool\n[B, C]', fillcolor='#e6ccff')
dot.node('SA_FC1', 'Linear\nC -> C//r\nbias=True', fillcolor='#e6ccff')
dot.node('SA_ReLU', 'ReLU\ninplace=True', fillcolor='#e6ccff')
dot.node('SA_FC2', 'Linear\nC//r -> C\nbias=True', fillcolor='#e6ccff')
dot.node('SA_Sigmoid', 'Sigmoid\nReshape [B, C, 1, 1]', fillcolor='#e6ccff')
dot.node('SA_Skip_Dummy', '', shape='point', width='0.01', style='invisible')
dot.node('SA_Add', '+\nAttention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5')  

# Edges for main path
dot.edges([
    ('SA_Input', 'SA_Pool'),
    ('SA_Pool', 'SA_FC1'),
    ('SA_FC1', 'SA_ReLU'),
    ('SA_ReLU', 'SA_FC2'),
    ('SA_FC2', 'SA_Sigmoid'),
    ('SA_Sigmoid', 'SA_Add')
])

# Skip connection edge
dot.edge('SA_Input', 'SA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('SA_Skip_Dummy', 'SA_Add', style='dashed', color='#ff3333', penwidth='2', label='Channel-wise Attention')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/SpectralAttention_Diagram'
dot.render(output_path, view=False)