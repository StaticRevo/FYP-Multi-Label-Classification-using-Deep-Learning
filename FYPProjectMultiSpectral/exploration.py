from config.config import *

metadata_csv = pd.read_csv(DatasetConfig.metadata_paths['100'])
class_weights = calculate_class_weights(metadata_csv)

print(class_weights)
print(DatasetConfig.class_labels)