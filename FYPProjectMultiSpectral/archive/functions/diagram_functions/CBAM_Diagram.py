# Third-party imports
from graphviz import Digraph
import os

# Initialize the CBAM diagram
dot = Digraph(
    comment='CBAM Module',
    format='pdf',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'true', 
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'nodesep': '0.5',  
        'label': 'Convolutional Block Attention Module (CBAM)',
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

# CBAM structure
dot.node('CBAM_Input', 'Input\n[B, C, H, W]', fillcolor='#ccffdd')

# Channel Attention
dot.node('CBAM_CA_Pool', 'AdaptiveAvgPool2d\n[B, C, 1, 1]', fillcolor='#ccffdd')
dot.node('CBAM_CA_Conv1', 'Conv2d\nC -> max(16, C//8)\nkernel_size=1, bias=False', fillcolor='#ccffdd')
dot.node('CBAM_CA_ReLU', 'ReLU\ninplace=True', fillcolor='#ccffdd')
dot.node('CBAM_CA_Conv2', 'Conv2d\nmax(16, C//8) -> C\nkernel_size=1, bias=False', fillcolor='#ccffdd')
dot.node('CBAM_CA_Sigmoid', 'Sigmoid\n[B, C, 1, 1]', fillcolor='#ccffdd')
dot.node('CBAM_CA_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  
dot.node('CBAM_CA_Mul', '×\nChannel Attention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5') 

# Spatial Attention
dot.node('CBAM_SA_Avg', 'AvgPool\nmean over channels\n[B, 1, H, W]', fillcolor='#ccffdd')
dot.node('CBAM_SA_Max', 'MaxPool\nmax over channels\n[B, 1, H, W]', fillcolor='#ccffdd')
dot.node('CBAM_SA_Concat', 'Concat\n[B, 2, H, W]', fillcolor='#ccffdd')
dot.node('CBAM_SA_Conv', 'Conv2d\nin_channels=2, out_channels=1\nkernel_size=7, padding=3, bias=False\nBatchNorm2d\nSigmoid\n[B, 1, H, W]', fillcolor='#ccffdd')
dot.node('CBAM_SA_Skip_Dummy', '', shape='point', width='0.01', style='invisible') 
dot.node('CBAM_SA_Mul', '×\nSpatial Attention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5') 

# Residual Connection
dot.node('CBAM_Res_Skip_Dummy', '', shape='point', width='0.01', style='invisible')  
dot.node('CBAM_Add', '+\nAttention\n[B, C, H, W]', shape='circle', fillcolor='#ffcccc', width='0.5')  

# Edges
# Channel Attention path
dot.edges([
    ('CBAM_Input', 'CBAM_CA_Pool'),
    ('CBAM_CA_Pool', 'CBAM_CA_Conv1'),
    ('CBAM_CA_Conv1', 'CBAM_CA_ReLU'),
    ('CBAM_CA_ReLU', 'CBAM_CA_Conv2'),
    ('CBAM_CA_Conv2', 'CBAM_CA_Sigmoid'),
    ('CBAM_CA_Sigmoid', 'CBAM_CA_Mul')
])
# Channel Attention skip
dot.edge('CBAM_Input', 'CBAM_CA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('CBAM_CA_Skip_Dummy', 'CBAM_CA_Mul', style='dashed', color='#ff3333', penwidth='2', label='Channel Attention')
# Spatial Attention path
dot.edges([
    ('CBAM_CA_Mul', 'CBAM_SA_Avg'),
    ('CBAM_CA_Mul', 'CBAM_SA_Max'),
    ('CBAM_SA_Avg', 'CBAM_SA_Concat'),
    ('CBAM_SA_Max', 'CBAM_SA_Concat'),
    ('CBAM_SA_Concat', 'CBAM_SA_Conv'),
    ('CBAM_SA_Conv', 'CBAM_SA_Mul')
])
# Spatial Attention skip
dot.edge('CBAM_CA_Mul', 'CBAM_SA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('CBAM_SA_Skip_Dummy', 'CBAM_SA_Mul', style='dashed', color='#ff3333', penwidth='2', label='Spatial Attention')
# Residual Connection
dot.edge('CBAM_Input', 'CBAM_Res_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2', constraint='false')
dot.edge('CBAM_Res_Skip_Dummy', 'CBAM_Add', style='dashed', color='#ff3333', penwidth='2', label='Residual Connection')
dot.edge('CBAM_SA_Mul', 'CBAM_Add')

# Render and save the diagram
output_path = '/Users/isaacattard/Downloads/Github Repositories/FYP-Multi-Label-Classification-using-Deep-Learning/FYPProjectMultiSpectral/archive/Diagrams/CBAM_Diagram'
dot.render(output_path, view=False)