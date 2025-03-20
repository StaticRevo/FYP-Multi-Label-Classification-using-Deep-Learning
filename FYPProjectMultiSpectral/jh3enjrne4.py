from config.config import *

class_labels = DatasetConfig.class_labels
for i, label_name in enumerate(class_labels):
    print(i, label_name)