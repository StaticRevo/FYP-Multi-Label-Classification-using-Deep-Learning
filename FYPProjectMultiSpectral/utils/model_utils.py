from models.models import *
import os
import sys
import inspect

# Get the model class
def get_model_class(model_name):
    model_mapping = {
        'CustomResNet50': (CustomResNet50, 'custom_model_resnet50'),
        'ResNet18': (ResNet18, 'resnet18'),
        'ResNet50': (ResNet50, 'resnet50'),
        'ResNet101': (ResNet101, 'resnet101'),
        'ResNet152': (ResNet152, 'resnet152'),
        'VGG16': (VGG16, 'vgg16'),
        'VGG19': (VGG19, 'vgg19'),
        'DenseNet121': (DenseNet121, 'densenet121'),
        'EfficientNetB0': (EfficientNetB0, 'efficientnetb0'),
        'EfficientNet_v2': (EfficientNetV2, 'efficientnet_v2'),
        'Vit-Transformer': (VitTransformer, 'vit_transformer'),
        'Swin-Transformer': (SwinTransformer, 'swin_transformer'),
        'CustomWRNB0': (CustomWRNB0, 'custom_wrn_b0'),
        'CustomWRNB4ECA': (CustomWRNB4ECA, 'custom_wrn_b4_eca'),
        'CustomModel': (CustomModel, 'custom_model'),
        'CustomModelV1': (CustomModelV1, 'custom_model_v1'),
        'CustomModelV2': (CustomModelV2, 'custom_model_v2'),
        'CustomModelV3': (CustomModelV3, 'custom_model_v3'),
        'CustomModelV4': (CustomModelV4, 'custom_model_v4'),
        'CustomModelV5': (CustomModelV5, 'custom_model_v5'),
        'CustomModelV6': (CustomModelV6, 'custom_model_v6'),
        'CustomModelV7': (CustomModelV7, 'custom_model_v7'),
        'CustomModelV8': (CustomModelV8, 'custom_model_v8'),
        'CustomModelV9': (CustomModelV9, 'custom_model_v9'),
    }
    return model_mapping.get(model_name, (None, None))

def get_class_names(module):
    # Get all classes defined in the module
    classes = inspect.getmembers(module, inspect.isclass)
    class_names = [cls[0] for cls in classes if cls[1].__module__ == module.__name__]
    
    return class_names