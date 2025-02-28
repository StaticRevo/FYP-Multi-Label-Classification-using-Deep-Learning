from models.models import *
from config.config import *

metadata_path = DatasetConfig.metadata_paths["1"]
metadata_csv = pd.read_csv(metadata_path)
class_weights = calculate_class_weights(metadata_csv)
num_classes = DatasetConfig.num_classes 
in_channels = len(DatasetConfig.all_bands)
model_weights = None
main_path = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_15epochs_1"
model = ResNet50.load_from_checkpoint(r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_15epochs_1\checkpoints\last.ckpt", 
                                      class_weights=class_weights,
                                      num_classes=num_classes,
                                      in_channels=in_channels,
                                      model_weights=model_weights,
                                      main_path=main_path)

main_path2 = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_15epochs_2"
model2 = ResNet50.load_from_checkpoint(r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_15epochs_2\checkpoints\last.ckpt", 
                                      class_weights=class_weights,
                                      num_classes=num_classes,
                                      in_channels=in_channels,
                                      model_weights=model_weights,
                                      main_path=main_path2)

main_path3 = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_30epochs"
model3 = ResNet50.load_from_checkpoint(r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_1%_BigEarthNet_30epochs\checkpoints\last.ckpt", 
                                      class_weights=class_weights,
                                      num_classes=num_classes,
                                      in_channels=in_channels,
                                      model_weights=model_weights,
                                      main_path=main_path3)

metadata_path = DatasetConfig.metadata_paths["10"]
metadata_csv = pd.read_csv(metadata_path)
class_weights = calculate_class_weights(metadata_csv)
main_path4 = r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_25epochs"
model4 = ResNet50.load_from_checkpoint(r"C:\Users\isaac\Desktop\experiments\ResNet50_None_all_bands_10%_BigEarthNet_25epochs\checkpoints\last.ckpt", 
                                      class_weights=class_weights,
                                      num_classes=num_classes,
                                      in_channels=in_channels,
                                      model_weights=model_weights,
                                      main_path=main_path4)

print(model.criterion)
print(model2.criterion)
print(model3.criterion)
print(model4.criterion)