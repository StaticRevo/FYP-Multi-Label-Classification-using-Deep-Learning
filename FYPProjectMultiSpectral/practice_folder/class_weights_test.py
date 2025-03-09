from config.config import *

metadata_path = DatasetConfig.metadata_paths["10"]
metadata = pd.read_csv(metadata_path)

class_weights = calculate_class_weights(metadata)
print(class_weights)

class_labels = calculate_class_labels(metadata)
print(class_labels)