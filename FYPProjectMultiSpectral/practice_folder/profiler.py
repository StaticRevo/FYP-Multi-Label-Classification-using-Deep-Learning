import cProfile, pstats, io
import pandas as pd
from dataloader import BigEarthNetDataLoader
from config.config import DatasetConfig, ModelConfig
import os
from utils.data_utils import extract_number

selected = '0.5%_BigEarthNet'
num = extract_number(selected)

main_path = r'C:\Users\isaac\Desktop\BigEarthTests\0.5%_BigEarthNet'
print("Main Path:", main_path)

# Create the cache file name using an f-string.
cache_file = f"{num}%_sample_weights.npy"
cache_path = os.path.join(main_path, cache_file)
print("Cache Path:", cache_path)