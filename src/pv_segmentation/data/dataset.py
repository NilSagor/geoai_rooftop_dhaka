"""SolarChipDataset: reads PV01 GeoTIFF chips and applies synchronized transforms."""
import json
from pathlib import Path

import rasterio
import torch
from torch.utils.data import Dataset

from pv_segmentation.transforms import apply_sync_transform


class SolarChipDataset(Dataset):
    def __init__(self, root: str, split_file: str, split_name: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        with open(split_file) as f:
            self.ids = json.load(f)[split_name]
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        with rasterio.open(self.img_dir / f"{stem}.tif") as src:
            image = src.read([1, 2, 3]).astype("float32") / 255.0
        with rasterio.open(self.mask_dir / f"{stem}.tif") as src:
            mask = src.read(1).astype("float32")

        image = torch.from_numpy(image)          # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            image, mask = apply_sync_transform(image, mask, self.transform)

        return image, mask