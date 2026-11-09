"""Phase 1: Convert PV01 BMP files to paired GeoTIFF images and masks."""
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import numpy as np
import re

ROOF_MAP = {
    "PV01_Rooftop_Brick": "brick",
    "PV01_Rooftop_FlatConcrete": "concrete",
    "PV01_Rooftop_SteelTile": "steel",
}

EXTS = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
LABEL_RE = re.compile(r"_label$", re.IGNORECASE)

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
        files = sorted(f for f in Path(raw_dir, folder).rglob("*")
                       if f.is_file() and f.suffix.lower() in EXTS)
        imgs = {f.stem: f for f in files if not LABEL_RE.search(f.stem)}
        masks = {LABEL_RE.sub("", f.stem): f
                 for f in files if LABEL_RE.search(f.stem)}

        paired = sorted(set(imgs) & set(masks))
        no_mask = sorted(set(imgs) - set(masks))
        no_img = sorted(set(masks) - set(imgs))

        for idx, stem in enumerate(paired, 1):   # sequential, no gaps
            name = f"{prefix}_{idx:04d}.tif"
            convert(imgs[stem], Path(out_img, name))
            convert(masks[stem], Path(out_mask, name), is_mask=True)

        print(f"{prefix}: {len(imgs)} images, {len(masks)} masks, "
              f"{len(paired)} pairs written")
        if no_mask:
            print(f"  images without mask: {no_mask[:5]}")
        if no_img:
            print(f"  masks without image: {no_img[:5]}")

    n_i = len(list(Path(out_img).glob("*.tif")))
    n_m = len(list(Path(out_mask).glob("*.tif")))
    print(f"Output totals -> images: {n_i}, masks: {n_m}")
    assert n_i == n_m, "Image/mask counts differ (stale files in output dirs?)"

if __name__ == "__main__":
    main()