# Standard library imports
import ast

# Third-party imports
import numpy as np

# Clean and parse labels from the metadata CSV
def clean_and_parse_labels(label_string):
    if isinstance(label_string, str):
        cleaned_labels = label_string.replace(" '", ", '").replace("[", "[").replace("]", "]")
        return ast.literal_eval(cleaned_labels)
    return label_string  

# Normalize class weights
def normalize_class_weights(class_weights):
    total_weight = sum(class_weights)
    return [weight / total_weight for weight in class_weights]

# Calculate class weights based on the label counts of each category
def calculate_class_weights(metadata_csv):
    metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels)

    class_labels = calculate_class_labels(metadata_csv)

    label_counts = metadata_csv['labels'].explode().value_counts()
    total_counts = label_counts.sum()
    class_weights = {label: total_counts / count for label, count in label_counts.items()}
    class_weights_array = np.array([class_weights[label] for label in class_labels])

    # normalized_class_weights = normalize_class_weights(class_weights_array) # Normalize class weights 
    log_weights = np.log1p(class_weights_array)

    boost_factor = 3.0
    log_weights[2] *= boost_factor # Boost the weight of class 3 - Beaches, dunes, sands
    log_weights[4] *= boost_factor # Boost the weight of class 5 - Coastal Wetlands

    return log_weights

# Calculate the class labels within the metadata
def calculate_class_labels(metadata_csv):
    metadata_csv['labels'] = metadata_csv['labels'].apply(clean_and_parse_labels) # Apply the cleaning and parsing function to the 'labels' column

    class_labels = set() 
    for labels in metadata_csv['labels']:
        class_labels.update(labels)

    class_labels = sorted(class_labels) # Convert the set to a sorted list

    return class_labels
