import timm
from models.BaseModel import BaseModel

class BigEarthNetVitTransformerModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Vision Transformer model with the specified input size
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(BigEarthNetVitTransformerModelTIF, self).__init__(vit_model, num_classes, class_weights)