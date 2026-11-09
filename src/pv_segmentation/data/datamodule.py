"""SolarDataModule: LightningDataModule for PV01 chips."""
import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from pv_segmentation.data.dataset import SolarChipDataset
from pv_segmentation.transforms import train_transform, val_transform

FIXED_SPLIT_SEED = 42  # seed used to generate data/splits/fixed_seed42.json


def _seed_worker(worker_id: int):
    """Seed numpy/random inside each DataLoader worker (torch is seeded by PyTorch)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class SolarDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "data/processed",
        split_dir: str = "data/splits",
        split_strategy: str = "fixed",      # fixed | per_seed
        split_file: str | None = None,      # explicit path overrides split_strategy
        chip_size: int = 256,
        stride: int = 128,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        if split_strategy not in ("fixed", "per_seed"):
            raise ValueError(f"unknown split_strategy: {split_strategy!r}")
        self.root = root
        self.chip_size = chip_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()

        if split_file is None:
            name = (f"fixed_seed{FIXED_SPLIT_SEED}.json" if split_strategy == "fixed"
                    else f"per_seed_{seed}.json")
            split_file = str(Path(split_dir) / name)
        if not Path(split_file).exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        self.split_file = split_file

    def _make(self, split_name: str, transform):
        return SolarChipDataset(
            self.root, self.split_file, split_name, transform,
            chip_size=self.chip_size, stride=self.stride,
        )

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_ds = self._make("train", train_transform)
            self.val_ds = self._make("val", val_transform)
        if stage in (None, "test"):
            self.test_ds = self._make("test", val_transform)

    def _loader(self, ds, shuffle: bool = False):
        kwargs = {}
        if shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            kwargs["generator"] = g
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker,
            **kwargs,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)