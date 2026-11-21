"""Rasterize building footprints to 10 m and 30 m chip masks.

Defaults to `all_touched=False` (pixel-center rule).
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

from .stac_common import (
    DHAKA_BBOX_4326,
    Grid,
    TARGET_CRS,
    chip_origins_from_shape,
    dhaka_grid,
)


def _write_mask_chip(
    path: Path,
    chip: np.ndarray,
    crs: str,
    transform: Affine,
) -> None:
    """Write a 2D `uint8` building mask array to a single-band GeoTIFF."""
    height, width = chip.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(chip, 1)


def rasterize_to_chips(
    gdf: gpd.GeoDataFrame,
    grid: Grid,
    chip_px: int,
    stride_px: int,
    out_dir: Path,
    all_touched: bool = False,
    overwrite: bool = False,
) -> int:
    """Rasterize valid footprints onto `grid` and slice into `(chip_px, chip_px)` chips."""
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_bounds_geom = box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    shapes = (
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty and geom.intersects(grid_bounds_geom)
    )

    full_mask = rasterize(
        shapes,
        out_shape=(grid.height_px, grid.width_px),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )

    n_written = 0
    for row, col in chip_origins_from_shape(
        height_px=grid.height_px,
        width_px=grid.width_px,
        chip_px=chip_px,
        stride_px=stride_px,
    ):
        stem = f"m{grid.res_m}_r{row:05d}_c{col:05d}"
        out_path = out_dir / f"{stem}.tif"
        if out_path.exists() and not overwrite:
            continue

        chip = full_mask[row : row + chip_px, col : col + chip_px]
        chip_tf = grid.chip_transform(row_px=row, col_px=col)
        _write_mask_chip(out_path, chip, crs=grid.crs, transform=chip_tf)
        n_written += 1

    print(f"[mask {grid.res_m}m] wrote {n_written} chips to {out_dir}")
    return n_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasterize building footprints into 10 m and 30 m chip masks."
    )
    parser.add_argument(
        "--footprints",
        type=Path,
        required=True,
        help="Path to building footprint vector file (.gpkg, .shp, .geojson).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/chips/masks"),
        help="Root output directory for rasterized mask chips.",
    )
    parser.add_argument(
        "--chip-px",
        type=int,
        default=256,
        help="Chip width and height in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Step size in pixels between adjacent chips.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize all pixels touched by footprint geometries rather than center rule.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask chip GeoTIFF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gdf = gpd.read_file(args.footprints)
    if gdf.crs is None:
        raise ValueError(f"Footprint dataset has no CRS: {args.footprints}")
    gdf = gdf.to_crs(TARGET_CRS)

    for res_m in (10, 30):
        grid = dhaka_grid(
            res_m=res_m,
            bbox_4326=DHAKA_BBOX_4326,
            target_crs=TARGET_CRS,
        )
        rasterize_to_chips(
            gdf=gdf,
            grid=grid,
            chip_px=args.chip_px,
            stride_px=args.stride,
            out_dir=args.out / f"m{res_m}",
            all_touched=args.all_touched,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()