"""Reassemble per-tile test probabilities into a full raster using the split manifest."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


def assemble_probability_mosaic(
    probs_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Reassemble per-tile probability arrays into a single GeoTIFF mosaic."""
    probs_p = Path(probs_path)
    manifest_p = Path(manifest_path)
    out_p = Path(output_path)

    if not probs_p.exists():
        raise FileNotFoundError(f"Probabilities file not found: {probs_p}")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_p}")

    probs = np.load(probs_p)  # Expected shape: (N, 1, H, W)
    if probs.ndim != 4 or probs.shape[1] != 1:
        raise ValueError(f"Expected probability array of shape (N, 1, H, W), got {probs.shape}")

    manifest = pd.read_csv(manifest_p)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    if len(test_df) != len(probs):
        raise ValueError(f"Mismatch between test tiles in manifest ({len(test_df)}) and loaded probabilities ({len(probs)})")

    required_cols = {"row", "col", "chip_size", "origin_x", "origin_y", "pixel_size"}
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required spatial columns: {missing_cols}")

    H, W = probs.shape[-2:]
    pixel_size = float(test_df["pixel_size"].iloc[0])

    xs = (test_df["origin_x"] + test_df["col"] * pixel_size).astype(float)
    ys = (test_df["origin_y"] + test_df["row"] * pixel_size).astype(float)

    xmin, ymax = xs.min(), ys.max()
    xmax = xs.max() + W * pixel_size
    ymin = ys.min() - H * pixel_size

    width = int(round((xmax - xmin) / pixel_size))
    height = int(round((ymax - ymin) / pixel_size))

    canvas = np.zeros((height, width), dtype=np.float32)

    for i, r in test_df.iterrows():
        col_idx = int(round((xs[i] - xmin) / pixel_size))
        row_idx = int(round((ymax - ys[i]) / pixel_size))
        
        # Clip indices to canvas boundaries if necessary
        canvas[row_idx:row_idx + H, col_idx:col_idx + W] = np.maximum(
            canvas[row_idx:row_idx + H, col_idx:col_idx + W],
            probs[i, 0],
        )

    crs = test_df["crs"].iloc[0] if "crs" in test_df else None

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_p,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(xmin, ymax, pixel_size, pixel_size),
    ) as dst:
        dst.write(canvas, 1)

    return out_p


def main():
    parser = argparse.ArgumentParser(description="Reassemble per-tile test probabilities into a full GeoTIFF raster.")
    parser.add_argument("--probs", type=Path, required=True, help="Path to per-tile probabilities (.npy).")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to split manifest CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/inventory/mosaic.tif"), help="Output GeoTIFF mosaic path.")
    args = parser.parse_args()

    print(f"=== Reassembling mosaic from {args.probs} ===")
    out_path = assemble_probability_mosaic(args.probs, args.manifest, args.out)
    print(f"Successfully wrote mosaic -> {out_path}")


if __name__ == "__main__":
    main()