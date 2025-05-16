# Third-party imports
import numpy as np
from torch.utils.data import WeightedRandomSampler

# Compute the frequency of each label in the dataset
def compute_label_frequencies(dataset, num_classes):
    label_counts = np.zeros(num_classes)
    for _, label in dataset:
        label_counts += label.numpy()  # multi-hot vector
    return label_counts

# Compute a weight for each sample based on inverse frequency of its positive labels
def compute_sample_weights(dataset, label_counts):
    sample_weights = []
    for _, label in dataset:
        label_np = label.numpy()
        pos_indices = np.where(label_np == 1)[0]
        if len(pos_indices) == 0:
            weight = 1.0
        else:
            inv_freqs = [1.0 / label_counts[i] for i in pos_indices]
            weight = np.mean(inv_freqs)
        sample_weights.append(weight)
    return sample_weights

# Create a weighted sampler for the dataset
def create_weighted_sampler(dataset, num_classes):
    label_counts = compute_label_frequencies(dataset, num_classes)
    sample_weights = compute_sample_weights(dataset, label_counts)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler # return the sampler for use in DataLoader
