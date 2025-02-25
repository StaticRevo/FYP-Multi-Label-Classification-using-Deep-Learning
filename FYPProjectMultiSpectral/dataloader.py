# Third-party imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Local application imports
from config.config import ModelConfig
from dataset import BigEarthNetDataset
from transformations.transforms import TransformsConfig

# Data module class for BigEarthNet dataset
class BigEarthNetDataLoader(pl.LightningDataModule):
    def __init__(self, bands=None, dataset_dir=None, metadata_csv=None):
        super().__init__()
        self.bands = bands
        self.dataset_dir = dataset_dir
        self.metadata_csv = metadata_csv
    
    def setup(self, stage=None):
        # Split the dataset into train, validation and test
        train_df = self.metadata_csv[self.metadata_csv['split'] == 'train']
        val_df = self.metadata_csv[self.metadata_csv['split'] == 'validation']
        test_df = self.metadata_csv[self.metadata_csv['split'] == 'test']

        # Initialize the dataset for training, validation and testing
        self.train_dataset = BigEarthNetDataset(df=train_df, 
                                                root_dir=self.dataset_dir, 
                                                transforms=TransformsConfig.train_transforms, 
                                                normalisation=TransformsConfig.normalisations, 
                                                selected_bands=self.bands, 
                                                metadata_csv=self.metadata_csv)
        
        self.val_dataset = BigEarthNetDataset(df=val_df, 
                                              root_dir=self.dataset_dir, 
                                              transforms=TransformsConfig.val_transforms, 
                                              normalisation=TransformsConfig.normalisations, 
                                              selected_bands=self.bands, 
                                              metadata_csv=self.metadata_csv)
        
        self.test_dataset = BigEarthNetDataset(df=test_df, 
                                               root_dir=self.dataset_dir, 
                                               transforms=TransformsConfig.test_transforms, 
                                               normalisation=TransformsConfig.normalisations, 
                                               selected_bands=self.bands, 
                                               metadata_csv=self.metadata_csv)
        
    def train_dataloader(self):
        # Create Dataloader for training dataset
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=ModelConfig.batch_size, 
                                num_workers=ModelConfig.num_workers, 
                                prefetch_factor=2, 
                                pin_memory=True, 
                                shuffle=True,  # sampler=sampler,
                                persistent_workers=True)
        return dataloader

    def val_dataloader(self):
        # Create Dataloader for validation dataset
        dataloader = DataLoader(self.val_dataset, 
                                batch_size=ModelConfig.batch_size,  
                                num_workers=ModelConfig.num_workers, 
                                pin_memory=True,  
                                persistent_workers=True)
        return dataloader

    def test_dataloader(self):
        # Create Dataloader for testing dataset
        dataloader = DataLoader(self.test_dataset,
                                batch_size=ModelConfig.batch_size,  
                                num_workers=ModelConfig.num_workers, 
                                pin_memory=True,  
                                persistent_workers=True)
        return dataloader
    