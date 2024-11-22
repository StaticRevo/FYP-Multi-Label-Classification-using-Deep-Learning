from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from FYPProjectRGB.preprocessing.config import DatasetConfig, ModelConfig
from dataset_rgbjpg import BigEarthNetSubset

class BigEarthNetSubsetDataModule(pl.LightningDataModule):    
    def setup(self, stage=None):
        self.train_transform = ModelConfig.train_transforms
        self.val_transform = ModelConfig.val_transforms
        self.test_transform = ModelConfig.test_transforms

        train_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'train']
        val_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'validation']
        test_df = DatasetConfig.metadata_csv[DatasetConfig.metadata_csv['split'] == 'test']

        self.train_dataset = BigEarthNetSubset(df=train_df, root_dir=DatasetConfig.combined_path, transforms=self.train_transform)
        self.val_dataset = BigEarthNetSubset(df=val_df, root_dir=DatasetConfig.combined_path, transforms=self.val_transform)
        self.test_dataset = BigEarthNetSubset(df=test_df, root_dir=DatasetConfig.combined_path, transforms=self.test_transform)
        print(f"Number of samples in train set: {len(self.train_dataset)}, val set: {len(self.val_dataset)}, test set: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=ModelConfig.batch_size, num_workers=ModelConfig.num_workers, pin_memory=True, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=ModelConfig.batch_size,  num_workers=ModelConfig.num_workers, pin_memory=True,  persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=ModelConfig.batch_size,  num_workers=ModelConfig.num_workers, pin_memory=True,  persistent_workers=True)