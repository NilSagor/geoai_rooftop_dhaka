"""SolarChipDataset: reads PV01 GeoTIFF chips and applies synchronized transforms."""
import json
from pathlib import Path

import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from pv_segmentation.transforms import apply_sync_transform

def _origins(size: int, chip: int, stride: int) -> list[int]:
    """Window start positions along one axis; the last window is flush with the edge."""
    if size <= chip:
        return [0]
    starts = list(range(0, size - chip + 1, stride))
    if starts[-1] != size - chip:
        starts.append(size - chip)
    return starts

class SolarChipDataset(Dataset):
    def __init__(self, root: str, split_file: str, split_name: str, transform=None, chip_size:int=256, stride:int=128):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.transform = transform
        self.chip_size = chip_size
        self.stride = stride

        with open(split_file) as f:
            ids = json.load(f)[split_name]

        # One entry per chip: (stem, row, col, needs_padding)
        self.chips = []
        for stem in ids:
            with rasterio.open(self.img_dir / f"{stem}.tif") as src:
                h, w = src.height, src.width
            for r in _origins(h, chip_size, stride):
                for c in _origins(w, chip_size, stride):
                    pad = (r + chip_size > h) or (c + chip_size > w)
                    self.chips.append((stem, r, c, pad))

    def __len__(self):
        return len(self.chips)

    def _read(self, path, bands, window, boundless):
        with rasterio.open(path) as src:
            return src.read(bands, window=window, boundless=boundless, fill_value=0).astype("float32")

    def __getitem__(self, idx):
        stem, r, c, pad = self.chips[idx]
        win = Window(c, r, self.chip_size, self.chip_size)

        image = self._read(self.img_dir / f"{stem}.tif", [1, 2, 3], win, pad)
        mask = self._read(self.mask_dir / f"{stem}.tif", 1, win, pad)

        image = torch.from_numpy(image.astype("float32") / 255.0)        # (3, H, W)
        mask = torch.from_numpy(mask.astype("float32")).unsqueeze(0)     # (1, H, W)

        if self.transform is not None:
            image, mask = apply_sync_transform(image, mask, self.transform)

        return image, mask