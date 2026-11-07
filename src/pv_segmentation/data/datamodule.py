"""SolarDataModule: LightningDataModule for PV01 chips."""
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from pv_segmentation.data.dataset import SolarChipDataset
from pv_segmentation.transforms import train_transform, val_transform



class SolarDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data/processed",
        split_file: str = "data/splits/fixed_seed42.json",
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.root = root
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        self.train_ds = SolarChipDataset(self.root, self.split_file, "train", train_transform)
        self.val_ds = SolarChipDataset(self.root, self.split_file, "val", val_transform)
        self.test_ds = SolarChipDataset(self.root, self.split_file, "test", val_transform)

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )