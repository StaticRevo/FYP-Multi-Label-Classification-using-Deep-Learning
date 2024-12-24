import numpy as np
import matplotlib.pyplot as plt
from config.config import DatasetConfig, clean_and_parse_labels
from utils.test_functions import *
from utils.helper_functions import *
from models.models import *
from dataloader import BigEarthNetTIFDataModule
import pandas as pd
import torch
import os
import random
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from utils.helper_functions import decode_target
from config.config import DatasetConfig, clean_and_parse_labels, calculate_class_weights
from transformations.normalisation import BandNormalisation

metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['5'])
class_labels = DatasetConfig.class_labels

# Load the saved file
data = np.load('test_predictions_ResNet18_5%_BigEarthNet.npz')

# Retrieve predictions and labels from the file
all_preds = data['all_preds']
all_labels = data['all_labels']

print((all_preds[0]))
print((all_labels[0]))

plot_roc_auc(all_labels, all_preds, class_labels)






