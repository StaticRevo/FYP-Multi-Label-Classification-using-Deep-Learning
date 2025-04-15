# -- Classes for Band Normalisation and Unnormalisation --

# Normalisation 
class BandNormalisation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        for i in range(image.shape[0]):
            image[i] = (image[i] - self.mean[i]) / self.std[i]
        return image

# Unnormalisation
class BandUnnormalisation:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        for i in range(image.shape[0]):
            image[i] = (image[i] * self.std[i]) + self.mean[i]
        return image