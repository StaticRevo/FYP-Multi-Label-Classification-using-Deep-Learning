from models.models import *

def get_model_class(model_name):
    model_mapping = {
        'custom_model': (CustomModel, 'custom_model'),
        'ResNet18': (ResNet18, 'resnet18'),
        'ResNet50': (ResNet50, 'resnet50'),
        'VGG16': (VGG16, 'vgg16'),
        'VGG19': (VGG19, 'vgg19'),
        'DenseNet121': (DenseNet121, 'densenet121'),
        'EfficientNetB0': (EfficientNetB0, 'efficientnetb0'),
        'EfficientNet_v2': (EfficientNetV2, 'efficientnet_v2'),
        'Vit-Transformer': (VitTransformer, 'vit_transformer'),
        'Swin-Transformer': (SwinTransformer, 'swin_transformer')
    }
    return model_mapping.get(model_name, (None, None))
