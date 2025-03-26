from graphviz import Digraph

# Initialize the diagram with global settings
dot = Digraph(
    comment='CustomModel Architecture',
    format='png',
    graph_attr={
        'rankdir': 'TB',
        'splines': 'false',  # Straight lines globally
        'bgcolor': '#f0f0f0',
        'pad': '0.5',
        'label': 'CustomModel Architecture (~949K Parameters)',
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
dot.node('SpectralMixer', 'Spectral Mixer\nConv2d(12->64, 1x1)\nBatchNorm2d\nGELU\nConv2d(64->32, 3x3)\nBatchNorm2d\nMaxPool2d(2x2)\n[B, 32, 60, 60]', fillcolor='#cce5ff')
dot.node('Block1', 'Block 1\nDepthwiseSeparableConv(32->32)\nBatchNorm2d\nGELU\nWideBottleneck(32->48)\nSpectralAttention\nDropout\nCoordinateAttention\nDropout\n[B, 48, 60, 60]', fillcolor='#b3d9ff')
dot.node('Block2', 'Block 2\nDepthwiseSeparableConv(48->48, stride=2)\nBatchNorm2d\nGELU\nWideBottleneck(48->96)\nECA\nDropout\n[B, 96, 30, 30]', fillcolor='#b3d9ff')
dot.node('Block3', 'Block 3\nDepthwiseSeparableConv(96->96, stride=2)\nBatchNorm2d\nGELU\nWideBottleneck(96->168)\nSE\nDropout\n[B, 168, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapter', 'Skip Adapter\nConv2d(48->168, 1x1)\nAvgPool2d(4x4)\n[B, 168, 15, 15]', fillcolor='#b3d9ff')
dot.node('Fusion', 'Lightweight Fusion\nConv2d(336->2, 1x1)\nBatchNorm2d\nSigmoid\nWeighted Sum\n[B, 168, 15, 15]', shape='box', fillcolor='#ffcccc')
dot.node('Block4', 'Block 4\nMultiScaleBlock(168->168)\nBatchNorm2d\nWideBottleneck(168->232)\nCBAM\nDropout\n[B, 232, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapterMid', 'Skip Adapter Mid\nConv2d(96->232, 1x1)\nAvgPool2d(2x2)\n[B, 232, 15, 15]', fillcolor='#b3d9ff')
dot.node('SkipAdapterDeep', 'Skip Adapter Deep\nConv2d(168->232, 1x1)\n[B, 232, 15, 15]', fillcolor='#b3d9ff')
dot.node('Fusion2', 'Parameter-Free Fusion\nSigmoid Masks\nWeighted Sum\n[B, 232, 15, 15]', shape='box', fillcolor='#ffcccc')
dot.node('Classifier', 'Classifier\nConv2d(232->128, 1x1)\nGELU\nAdaptiveAvgPool2d(1)\nFlatten\nDropout\nLinear(128->19)\n[B, 19]', fillcolor='#cce5ff')

# Dummy nodes for skip connection paths (to create dashed lines)
dot.node('Dummy1', '', shape='point', width='0.01', style='invisible')
dot.node('Dummy2', '', shape='point', width='0.01', style='invisible')
dot.node('Dummy3', '', shape='point', width='0.01', style='invisible')

# Edges for main structure (straight lines for main path)
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

# Skip connection edges (dashed lines)
dot.edge('Block1', 'Dummy1', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy1', 'SkipAdapter', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 1 to Fusion)')
dot.edge('SkipAdapter', 'Fusion', style='dashed', color='#ff3333', penwidth='2')

dot.edge('Block2', 'Dummy2', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy2', 'SkipAdapterMid', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 2 to Fusion2)')
dot.edge('SkipAdapterMid', 'Fusion2', style='dashed', color='#ff3333', penwidth='2')

dot.edge('Block3', 'Dummy3', style='dashed', color='#ff3333', penwidth='2')
dot.edge('Dummy3', 'SkipAdapterDeep', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection (Block 3 to Fusion2)')
dot.edge('SkipAdapterDeep', 'Fusion2', style='dashed', color='#ff3333', penwidth='2')

# WideBottleneck subgraph
with dot.subgraph(name='cluster_WideBottleneck') as wb:
    wb.attr(
        label='WideBottleneck',
        style='filled',
        fillcolor='#e6ffe6',
        fontname='Arial',
        fontsize='14',
        fontcolor='#006600',
    )
    wb.node('WB_Input', 'Input\n[B, in_channels, H, W]', fillcolor='#ccffcc')
    wb.node('WB_Conv1', 'Conv1 (1x1)\nBatchNorm2d\nReLU', fillcolor='#ccffcc')
    wb.node('WB_Conv2', 'Conv2 (3x3)\nBatchNorm2d\nReLU', fillcolor='#ccffcc')
    wb.node('WB_Conv3', 'Conv3 (1x1)\nBatchNorm2d', fillcolor='#ccffcc')
    wb.node('WB_Skip', 'Downsample (if needed)\nConv2d (1x1)\nBatchNorm2d', style='filled', fillcolor='#ccffcc')
    wb.node('WB_Skip_Dummy', '', shape='point', width='0.01', style='invisible')
    wb.node('WB_Add', '+', shape='circle', fillcolor='#ffcccc', width='0.7')
    wb.node('WB_ReLU', 'ReLU\n[B, out_channels, H, W]', fillcolor='#ccffcc')
    
    wb.edges([
        ('WB_Input', 'WB_Conv1'),
        ('WB_Conv1', 'WB_Conv2'),
        ('WB_Conv2', 'WB_Conv3'),
        ('WB_Conv3', 'WB_Add'),
        ('WB_Skip', 'WB_Add'),
        ('WB_Add', 'WB_ReLU')
    ])
    wb.edge('WB_Input', 'WB_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2')
    wb.edge('WB_Skip_Dummy', 'WB_Skip', style='dashed', color='#ff3333', penwidth='2', label='Skip Connection')

# ECA subgraph
with dot.subgraph(name='cluster_ECA') as eca:
    eca.attr(
        label='ECA Module',
        style='filled',
        fillcolor='#fff0e6',
        fontname='Arial',
        fontsize='14',
        fontcolor='#cc3300',
    )
    eca.node('ECA_Input', 'Input\n[B, C, H, W]', fillcolor='#ffe6cc')
    eca.node('ECA_Pool', 'AdaptiveAvgPool2d\n[B, C, 1, 1]', fillcolor='#ffe6cc')
    eca.node('ECA_Reshape1', 'Reshape\n[B, 1, C]', fillcolor='#ffe6cc')
    eca.node('ECA_Conv', 'Conv1d (k_size)\n[B, 1, C]', fillcolor='#ffe6cc')
    eca.node('ECA_Reshape2', 'Reshape\n[B, C, 1, 1]', fillcolor='#ffe6cc')
    eca.node('ECA_Sigmoid', 'Sigmoid', fillcolor='#ffccb3')
    eca.node('ECA_Skip_Dummy', '', shape='point', width='0.01', style='invisible')
    eca.node('ECA_Skip', 'Input Reuse', style='filled', fillcolor='#ffcccc')
    eca.node('ECA_Mul', '×\n[B, C, H, W]', shape='circle', fillcolor='#ffffff', width='0.5')
    
    eca.edges([
        ('ECA_Input', 'ECA_Pool'),
        ('ECA_Pool', 'ECA_Reshape1'),
        ('ECA_Reshape1', 'ECA_Conv'),
        ('ECA_Conv', 'ECA_Reshape2'),
        ('ECA_Reshape2', 'ECA_Sigmoid'),
        ('ECA_Sigmoid', 'ECA_Mul'),
        ('ECA_Skip', 'ECA_Mul')
    ])
    eca.edge('ECA_Input', 'ECA_Skip_Dummy', style='dashed', color='#ff3333', penwidth='2')
    eca.edge('ECA_Skip_Dummy', 'ECA_Skip', style='dashed', color='#ff3333', penwidth='2')

# SpectralAttention subgraph
with dot.subgraph(name='cluster_SpectralAttention') as sa:
    sa.attr(
        label='SpectralAttention',
        style='filled',
        fillcolor='#f0e6ff',
        fontname='Arial',
        fontsize='14',
        fontcolor='#6600cc',
    )
    sa.node('SA_Input', 'Input\n[B, C, H, W]', fillcolor='#e6ccff')
    sa.node('SA_Pool', 'GlobalAvgPool\n[B, C]', fillcolor='#e6ccff')
    sa.node('SA_FC1', 'Linear\nC -> C//r', fillcolor='#e6ccff')
    sa.node('SA_ReLU', 'ReLU', fillcolor='#e6ccff')
    sa.node('SA_FC2', 'Linear\nC//r -> C', fillcolor='#e6ccff')
    sa.node('SA_Sigmoid', 'Sigmoid\n[B, C, 1, 1]', fillcolor='#e6ccff')
    sa.node('SA_Add', '+', shape='circle', fillcolor='#ffcccc', width='0.7')
    
    sa.edges([
        ('SA_Input', 'SA_Pool'),
        ('SA_Pool', 'SA_FC1'),
        ('SA_FC1', 'SA_ReLU'),
        ('SA_ReLU', 'SA_FC2'),
        ('SA_FC2', 'SA_Sigmoid'),
        ('SA_Sigmoid', 'SA_Add'),
        ('SA_Input', 'SA_Add')
    ])

# CoordinateAttention subgraph
with dot.subgraph(name='cluster_CoordinateAttention') as ca:
    ca.attr(
        label='CoordinateAttention',
        style='filled',
        fillcolor='#e6f0ff',
        fontname='Arial',
        fontsize='14',
        fontcolor='#0033cc',
    )
    ca.node('CA_Input', 'Input\n[B, C, H, W]', fillcolor='#cce6ff')
    ca.node('CA_PoolH', 'Pool Height\n[B, C, H, 1]', fillcolor='#cce6ff')
    ca.node('CA_PoolW', 'Pool Width\n[B, C, 1, W]', fillcolor='#cce6ff')
    ca.node('CA_Concat', 'Concat\n[B, C, H+W, 1]', fillcolor='#cce6ff')
    ca.node('CA_Conv1', 'Conv2d\nBatchNorm2d\nReLU', fillcolor='#cce6ff')
    ca.node('CA_Split', 'Split\nH and W', fillcolor='#cce6ff')
    ca.node('CA_ConvH', 'Conv2d (H)\nSigmoid', fillcolor='#cce6ff')
    ca.node('CA_ConvW', 'Conv2d (W)\nSigmoid', fillcolor='#cce6ff')
    ca.node('CA_Mul', '×\n[B, C, H, W]', shape='circle', fillcolor='#ffffff', width='0.5')
    
    ca.edges([
        ('CA_Input', 'CA_PoolH'),
        ('CA_Input', 'CA_PoolW'),
        ('CA_PoolH', 'CA_Concat'),
        ('CA_PoolW', 'CA_Concat'),
        ('CA_Concat', 'CA_Conv1'),
        ('CA_Conv1', 'CA_Split'),
        ('CA_Split', 'CA_ConvH'),
        ('CA_Split', 'CA_ConvW'),
        ('CA_ConvH', 'CA_Mul'),
        ('CA_ConvW', 'CA_Mul'),
        ('CA_Input', 'CA_Mul')
    ])

# SE subgraph
with dot.subgraph(name='cluster_SE') as se:
    se.attr(
        label='SE Module (in_channels=168)',
        style='filled',
        fillcolor='#ffe6f0',
        fontname='Arial',
        fontsize='14',
        fontcolor='#cc0066',
    )
    se.node('SE_Input', 'Input\n[B, 168, H, W]', fillcolor='#ffccdd')
    se.node('SE_Pool', 'AdaptiveAvgPool2d\n[B, 168, 1, 1]', fillcolor='#ffccdd')
    se.node('SE_FC1', 'Conv2d\n168 -> 168//r', fillcolor='#ffccdd')
    se.node('SE_ReLU', 'ReLU', fillcolor='#ffccdd')
    se.node('SE_FC2', 'Conv2d\n168//r -> 168', fillcolor='#ffccdd')
    se.node('SE_Sigmoid', 'Sigmoid', fillcolor='#ffccdd')
    se.node('SE_Mul', '×\n[B, 168, H, W]', shape='circle', fillcolor='#ffffff', width='0.5')
    
    se.edges([
        ('SE_Input', 'SE_Pool'),
        ('SE_Pool', 'SE_FC1'),
        ('SE_FC1', 'SE_ReLU'),
        ('SE_ReLU', 'SE_FC2'),
        ('SE_FC2', 'SE_Sigmoid'),
        ('SE_Sigmoid', 'SE_Mul'),
        ('SE_Input', 'SE_Mul')
    ])

# CBAM subgraph
with dot.subgraph(name='cluster_CBAM') as cbam:
    cbam.attr(
        label='CBAM Module (in_channels=232)',
        style='filled',
        fillcolor='#f0ffe6',
        fontname='Arial',
        fontsize='14',
        fontcolor='#006633',
    )
    cbam.node('CBAM_Input', 'Input\n[B, 232, H, W]', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_Pool', 'AdaptiveAvgPool2d\n[B, 232, 1, 1]', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_Conv1', 'Conv2d\n232 -> 232//r', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_ReLU', 'ReLU', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_Conv2', 'Conv2d\n232//r -> 232', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_Sigmoid', 'Sigmoid', fillcolor='#ccffdd')
    cbam.node('CBAM_CA_Mul', '×', shape='circle', fillcolor='#ffffff', width='0.5')
    cbam.node('CBAM_SA_Avg', 'AvgPool', fillcolor='#ccffdd')
    cbam.node('CBAM_SA_Max', 'MaxPool', fillcolor='#ccffdd')
    cbam.node('CBAM_SA_Concat', 'Concat', fillcolor='#ccffdd')
    cbam.node('CBAM_SA_Conv', 'Conv2d\nBatchNorm2d\nSigmoid', fillcolor='#ccffdd')
    cbam.node('CBAM_SA_Mul', '×', shape='circle', fillcolor='#ffffff', width='0.5')
    cbam.node('CBAM_Add', '+', shape='circle', fillcolor='#ffcccc', width='0.7')
    
    cbam.edges([
        ('CBAM_Input', 'CBAM_CA_Pool'),
        ('CBAM_CA_Pool', 'CBAM_CA_Conv1'),
        ('CBAM_CA_Conv1', 'CBAM_CA_ReLU'),
        ('CBAM_CA_ReLU', 'CBAM_CA_Conv2'),
        ('CBAM_CA_Conv2', 'CBAM_CA_Sigmoid'),
        ('CBAM_CA_Sigmoid', 'CBAM_CA_Mul'),
        ('CBAM_Input', 'CBAM_CA_Mul'),
        ('CBAM_CA_Mul', 'CBAM_SA_Avg'),
        ('CBAM_CA_Mul', 'CBAM_SA_Max'),
        ('CBAM_SA_Avg', 'CBAM_SA_Concat'),
        ('CBAM_SA_Max', 'CBAM_SA_Concat'),
        ('CBAM_SA_Concat', 'CBAM_SA_Conv'),
        ('CBAM_SA_Conv', 'CBAM_SA_Mul'),
        ('CBAM_CA_Mul', 'CBAM_SA_Mul'),
        ('CBAM_SA_Mul', 'CBAM_Add'),
        ('CBAM_Input', 'CBAM_Add')
    ])

# Render and save the diagram
dot.render('CustomModel_Diagram', view=True)

