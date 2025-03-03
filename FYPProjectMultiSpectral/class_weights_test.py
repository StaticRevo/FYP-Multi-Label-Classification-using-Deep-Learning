from config.config import calculate_class_weights, DatasetConfig
import pandas as pd

metadata_csv = pd.read_csv(DatasetConfig.metadata_paths["100"])
class_weights = calculate_class_weights(metadata_csv)
print(class_weights)
print(DatasetConfig.reversed_class_labels_dict)