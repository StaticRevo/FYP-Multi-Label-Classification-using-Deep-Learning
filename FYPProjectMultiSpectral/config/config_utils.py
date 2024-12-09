import ast
import numpy as np
import os
import pandas as pd

def clean_and_parse_labels(label_string):
    cleaned_labels = label_string.replace(" '", ", '").replace("[", "[").replace("]", "]")
    return ast.literal_eval(cleaned_labels)

def calculate_class_weights(metadata_path):
    metadata_csv = pd.read_csv(metadata_path)
    metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

    class_labels = set()
    for labels in metadata_csv['labels']:
        class_labels.update(labels)

    label_counts = metadata_csv['labels'].explode().value_counts()
    total_counts = label_counts.sum()
    class_weights = {label: total_counts / count for label, count in label_counts.items()}
    class_weights_array = np.array([class_weights[label] for label in class_labels])

    return class_labels, class_weights, class_weights_array, metadata_csv

