# -- Classes for Band Normalisation and Unnormalisation --

# Normalisation: Applies per-band normalization to an image tensor using provided mean and std
class BandNormalisation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Normalize each band: (value - mean) / std
        for i in range(image.shape[0]):
            image[i] = (image[i] - self.mean[i]) / self.std[i]
        return image

# Unnormalisation: Reverts per-band normalization to restore original image values
class BandUnnormalisation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Unnormalize each band: (value * std) + mean
        for i in range(image.shape[0]):
            image[i] = (image[i] * self.std[i]) + self.mean[i]
        return image