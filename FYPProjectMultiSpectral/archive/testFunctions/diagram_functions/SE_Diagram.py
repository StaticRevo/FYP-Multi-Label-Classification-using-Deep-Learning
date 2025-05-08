# Third-party imports
from graphviz import Digraph
import os

# Initialize the SE diagram
dot = Digraph(
    comment='SE Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true',  
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'label': 'Squeeze and Excitation (SE) Module',
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

# SE structure
dot.node('SE_Input', 'Input\n[B, C, H, W]', fillcolor='#ffccdd')
dot.node('SE_Pool', 'AdaptiveAvgPool2d\n[B, C, 1, 1]', fillcolor='#ffccdd')
dot.node('SE_FC1', 'Conv2d\nC -> C//r\nkernel_size=1', fillcolor='#ffccdd')
dot.node('SE_Act', 'Activation\n(typically ReLU)', fillcolor='#ffccdd')
dot.node('SE_Dropout', 'Dropout\n(optional)', fillcolor='#ffccdd', style='filled,dashed')  # Dashed to indicate optional
dot.node('SE_FC2', 'Conv2d\nC//r -> C\nkernel_size=1', fillcolor='#ffccdd')
dot.node('SE_Sigmoid', 'Sigmoid', fillcolor='#ffccdd')
dot.node('SE_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  # Dummy for skip connection
dot.node('SE_Mul', '×\n[B, C, H, W]', shape='circle', fillcolor='#ffffff', width='0.5')

# Edges for main path
dot.edges([
    ('SE_Input', 'SE_Pool'),
    ('SE_Pool', 'SE_FC1'),
    ('SE_FC1', 'SE_Act'),
    ('SE_Act', 'SE_Dropout'),
    ('SE_Dropout', 'SE_FC2'),
    ('SE_FC2', 'SE_Sigmoid'),
    ('SE_Sigmoid', 'SE_Mul')
])

# Skip connection edge
dot.edge('SE_Input', 'SE_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('SE_Skip_Dummy', 'SE_Mul', style='dashed', color='#ff3333', penwidth='2', label='Input Scaling')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/SE_Diagram'
dot.render(output_path, view=False)