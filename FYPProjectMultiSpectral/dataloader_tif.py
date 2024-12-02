from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from config.config import DatasetConfig, ModelConfig
from dataset_tif import BigEarthNetDatasetTIF
from transformations.transforms import TransformsConfig

class BigEarthNetTIFDataModule(pl.LightningDataModule):
    def __init__(self, bands=None):
        super().__init__()
        self.bands = bands

    def setup(self, stage=None):
        train_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'train']
        val_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'validation']
        test_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'test']

        self.train_dataset = BigEarthNetDatasetTIF(df=train_df, root_dir=DatasetConfig.dataset_path, transforms=None, selected_bands=self.bands)
        self.val_dataset = BigEarthNetDatasetTIF(df=val_df, root_dir=DatasetConfig.dataset_path, transforms=None, selected_bands=self.bands)
        self.test_dataset = BigEarthNetDatasetTIF(df=test_df, root_dir=DatasetConfig.dataset_path, transforms=None, selected_bands=self.bands)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=ModelConfig.batch_size, num_workers=ModelConfig.num_workers, pin_memory=True, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=ModelConfig.batch_size,  num_workers=ModelConfig.num_workers, pin_memory=True,  persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=ModelConfig.batch_size,  num_workers=ModelConfig.num_workers, pin_memory=True,  persistent_workers=True)