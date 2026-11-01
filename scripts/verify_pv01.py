import rasterio
from pathlib import Path

imgs = {p.stem for p in Path("data/processed/images").glob("*.tif")}
masks = {p.stem for p in Path("data/processed/masks").glob("*.tif")}
assert imgs == masks, f"Mismatch: {imgs ^ masks}"
for stem in sorted(imgs)[:5]:
    with rasterio.open(f"data/processed/images/{stem}.tif") as i, \
         rasterio.open(f"data/processed/masks/{stem}.tif") as m:
        assert i.shape == m.shape, f"Shape mismatch: {stem}"
        assert i.transform == m.transform, f"Transform mismatch: {stem}"
print(f"OK: {len(imgs)} pairs verified")