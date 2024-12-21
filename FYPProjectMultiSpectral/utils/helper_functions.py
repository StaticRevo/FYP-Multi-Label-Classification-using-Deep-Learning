import torch
import ast
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import random
from config.config import DatasetConfig

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# Helper functions
def denormalize(tensors, *, mean, std):
    for c in range(DatasetConfig.band_channels):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)

def encode_label(label: list, num_classes=DatasetConfig.num_classes):
    target = torch.zeros(num_classes)
    for l in label:
        if l in DatasetConfig.class_labels_dict:
            target[DatasetConfig.class_labels_dict[l]] = 1.0
    return target

def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.4,
    cls_labels: dict = None,
):
    result = []
    for i, x in enumerate(target):
        if x >= threshold:
            if text_labels:
                result.append(cls_labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)


def get_band_indices(band_names, all_band_names):
    return [all_band_names.index(band) for band in band_names]



def get_labels_for_image(image_path, model, transform, patch_to_labels):
    # Load and preprocess the image
    with rasterio.open(image_path) as src:
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        image = transform(image).unsqueeze(0).to(model.device)  # Add batch dimension and move to device

    # Get the predicted labels
    model.eval()
    with torch.no_grad():
        preds = model(image).sigmoid() > 0.5  # Apply sigmoid and threshold at 0.5
        preds = preds.cpu().numpy().astype(int).flatten()

    # Get the true labels
    patch_id = os.path.basename(image_path).split('.')[0]
    true_labels = patch_to_labels[patch_id]

    return preds, true_labels, image

def display_image(image_path):
    with rasterio.open(image_path) as src:
        bands = [2, 3, 4]  # Bands to combine for display
        image = np.stack([src.read(band) for band in bands], axis=-1)
        plt.imshow(image)
        plt.title("Image with Bands 2, 3, and 4")
        plt.show()

def display_image_and_labels(image_path, model, transform, patch_to_labels):
    # Display the image
    display_image(image_path)

    # Get predicted and true labels
    preds, true_labels, _ = get_labels_for_image(image_path, model, transform, patch_to_labels)
    print(f"Predicted Labels: {preds}")
    print(f"True Labels: {true_labels}")

# Function to save TensorBoard graphs as images
def save_tensorboard_graphs(log_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get the list of tags (metrics) in the logs
    tags = event_acc.Tags()['scalars']

    # Filter tags
    filtered_tags = [tag for tag in tags if '_epoch' in tag or tag.startswith('val_')]

    # Iterate over each tag and plot the graph
    for tag in filtered_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Plot the graph
        plt.figure()
        plt.plot(steps, values)
        plt.xlabel('Steps')
        plt.ylabel(tag)
        plt.title(tag)
        plt.grid(True)

        # Save the graph as an image
        output_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Graphs saved to {output_dir}")

def extract_number(string):
    number_str = string.split('%')[0]
    try:
        number = float(number_str)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        raise ValueError(f"Cannot extract a number from the string: {string}")

    

