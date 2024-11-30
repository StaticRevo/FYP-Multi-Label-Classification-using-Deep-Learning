from models.CustomModel import CustomModel
from models.ResNet18 import BigEarthNetResNet18ModelTIF
from models.VGG16 import BigEarthNetVGG16ModelTIF
from config.config import ModelConfig, DatasetConfig

model = CustomModel(class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3)
model.print_summary((3, 120, 120))
model.visualize_model((3, 120, 120), 'custom_model.png')

model = BigEarthNetResNet18ModelTIF(class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3)
model.print_summary((3, 120, 120))

model = BigEarthNetVGG16ModelTIF(class_weights=DatasetConfig.class_weights, num_classes=19, in_channels=3)
model.print_summary((3, 120, 120))



