import timm
from models.BaseModel import BaseModel

class BigEarthNetSwinTransformerModelTIF(BaseModel):
    def __init__(self, class_weights, num_classes, in_channels, model_weights):
        if model_weights is None:
            model_weights = False
        else :
            model_weights = True

        # Load the Swin Transformer model with the specified input size
        swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=model_weights, num_classes=num_classes, in_chans=in_channels, img_size=120)

        # Call the parent class constructor with the modified model
        super(BigEarthNetSwinTransformerModelTIF, self).__init__(swin_model, num_classes, class_weights)