"""SolarDataModule: LightningDataModule for PV01 chips with corrugated oversampling."""
import random
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from pv_segmentation.data.dataset import SolarChipDataset
from pv_segmentation.transforms import train_transform, val_transform

FIXED_SPLIT_SEED = 42  # seed used to generate data/splits/fixed_seed42.json


def _seed_worker(worker_id: int):
    """Seed numpy/random inside each DataLoader worker (torch is seeded by PyTorch)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

def _corrugated_score(img_chip: torch.Tensor) -> float:
    """Heuristic: high horizontal-gradient energy + high brightness variance.

    Operates on a (3,H,W) float [0,1] chip. Returns a scalar score; tiles above
    a configurable threshold are flagged as corrugated-metal candidates.
    """
    gray = img_chip.mean(dim=0, keepdim=True)  # (1,H,W)
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    energy = gx.abs().mean().item()
    brightness_std = gray.std().item()
    return energy * (1.0 + brightness_std)


def _flag_indices(ds, threshold: float = 0.02) -> list[int]:
    """Return indices of flagged tiles by scanning the dataset (slow once)."""
    idx = []
    for i in range(len(ds)):
        img, _ = ds[i]
        if _corrugated_score(img) > threshold:
            idx.append(i)
    return idx


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
        oversample: bool = False,
        oversample_factor: float = 3.0,
        corrugated_threshold: float = 0.02,
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
        self.oversample = oversample
        self.oversample_factor = oversample_factor
        self.corrugated_threshold = corrugated_threshold
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

    def _loader(self, ds, shuffle: bool = False, sampler=None):
        kwargs = {}
        if shuffle and sampler is None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            kwargs["generator"] = g
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker,
            **kwargs,
        )

    def train_dataloader(self):
        if self.oversample:
            flagged = _flag_indices(self.train_ds, self.corrugated_threshold)
            w = torch.ones(len(self.train_ds))
            if flagged:
                w[flagged] = self.oversample_factor
            sampler = WeightedRandomSampler(w, num_samples=len(self.train_ds), replacement=True)
            return self._loader(self.train_ds, shuffle=False, sampler=sampler)
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)
