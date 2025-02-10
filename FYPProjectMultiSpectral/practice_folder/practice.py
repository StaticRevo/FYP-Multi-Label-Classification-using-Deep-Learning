import os
from ..config.config import DatasetConfig

checkpoint_path = r'C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_0.5%_BigEarthNet_2epochs\checkpoints\last.ckpt'
print(checkpoint_path)

main_path = os.path.dirname(os.path.dirname(checkpoint_path))
print(f"Main path: {main_path}")

print(DatasetConfig.reversed_class_labels_dict)