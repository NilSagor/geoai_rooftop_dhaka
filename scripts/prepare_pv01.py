"""Phase 1: Convert PV01 BMP files to paired GeoTIFF images and masks."""
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import numpy as np

ROOF_MAP = {
    "PV01_Rooftop_Brick": "brick",
    "PV01_Rooftop_FlatConcrete": "concrete",
    "PV01_Rooftop_SteelTile": "steel",
}

def convert(src, dst, is_mask=False):
    with rasterio.open(src) as s:
        data = s.read()
        h, w = s.height, s.width
        count, dtype = s.count, s.dtypes[0]
    if is_mask:
        data = (data > 0).astype(np.uint8)
        dtype = "uint8"
    transform = from_bounds(0, 0, w, h, w, h)
    profile = dict(driver="GTiff", height=h, width=w, count=count,
                   dtype=dtype, crs="EPSG:4326", transform=transform)
    with rasterio.open(dst, "w", **profile) as d:
        d.write(data)

def main(raw_dir="data/raw/PV01", out_img="data/processed/images",
         out_mask="data/processed/masks"):
    Path(out_img).mkdir(parents=True, exist_ok=True)
    Path(out_mask).mkdir(parents=True, exist_ok=True)
    for folder, prefix in ROOF_MAP.items():
        files = sorted(Path(raw_dir, folder).glob("*"))
        imgs = [f for f in files if "_label" not in f.stem]
        masks = {f.stem.replace("_label", ""): f
                 for f in files if "_label" in f.stem}
        for idx, img in enumerate(imgs, 1):
            if img.stem not in masks:
                continue
            name = f"{prefix}_{idx:04d}.tif"
            convert(img, Path(out_img, name))
            convert(masks[img.stem], Path(out_mask, name), is_mask=True)
        print(f"{prefix}: {len(imgs)} pairs processed")

if __name__ == "__main__":
    main()