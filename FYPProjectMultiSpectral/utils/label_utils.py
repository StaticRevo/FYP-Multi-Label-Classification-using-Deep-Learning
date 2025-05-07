# Third-party imports
import torch

# Local application imports
from config.config import DatasetConfig
from models.models import *

# Encode the labels
def encode_label(label: list, num_classes=DatasetConfig.num_classes):
    target = torch.zeros(num_classes)
    for l in label:
        if l in DatasetConfig.class_labels_dict:
            target[DatasetConfig.class_labels_dict[l]] = 1.0
    return target

# Decode the labels
def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.5,
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