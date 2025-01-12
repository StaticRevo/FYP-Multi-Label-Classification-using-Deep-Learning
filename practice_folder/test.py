import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import logging
import rasterio


def check_crs(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            crs = src.crs
            logging.info(f"{tif_path}: CRS is {crs}")
            return crs
    except Exception as e:
        logging.error(f"Error reading {tif_path}: {e}")
        return None
    

print(check_crs(r"C:\Users\isaac\Desktop\BigEarthTests\1%_BigEarthNet\CombinedImages\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_68.tif"))
    
