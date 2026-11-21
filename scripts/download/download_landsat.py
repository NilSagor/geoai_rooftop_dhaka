"""Download 30 m Landsat Collection-2 L2 median winter composites as chips.

One chip = 256x256 px in TARGET_CRS at 30 m. Writes:
- images/<stem>.tif  (5 bands: blue, green, red, nir08, swir16, uint16 scaled by 10000)
- valid/<stem>.tif   (1 band: uint8, 1 = clear & valid pixel present)

Resumable: skips chips whose outputs already exist unless `--overwrite` is passed.
"""
import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import odc.stac
import planetary_computer
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import rioxarray  # noqa: F401 - registers `.rio` accessor on xarray objects
import xarray as xr

from .stac_common import (
    DHAKA_BBOX_4326,
    Grid,
    TARGET_CRS,
    chip_origins_from_shape,
    dhaka_grid,
    open_catalog,
)

LS_BANDS: list[str] = ["blue", "green", "red", "nir08", "swir16"]
LS_PLATFORMS: list[str] = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
REFLECTANCE_SCALE_OUT: float = 10000.0

# Landsat C2 L2 QA_PIXEL bit masks (bits 0..4):
# 0: Fill, 1: Dilated Cloud, 2: Cirrus, 3: Cloud, 4: Cloud Shadow
QA_BAD_BITS_MASK: int = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

GDAL_NET_ENV: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2.0",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}


def _band_scale_offset(item: pystac.Item, band: str) -> tuple[float, float]:
    """Read Collection-2 scale/offset from STAC asset metadata, falling back to C2 L2 SR."""
    asset = item.assets.get(band)
    if asset is not None:
        rb_list: list[dict[str, Any]] = asset.extra_fields.get("raster:bands", [])
        if rb_list:
            rb = rb_list[0]
            if "scale" in rb and "offset" in rb:
                return float(rb["scale"]), float(rb["offset"])
    return 2.75e-5, -0.2


def fetch_winter_items(
    year: int,
    bbox: tuple[float, float, float, float] = DHAKA_BBOX_4326,
    max_cloud_cover: float = 15.0,
) -> list[pystac.Item]:
    """Query Landsat C2 L2 scenes for Nov (year-1) through Mar (year)."""
    catalog = open_catalog()
    date_range = f"{year - 1}-11-01/{year}-03-31"
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=list(bbox),
        datetime=date_range,
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover},
            "platform": {"in": LS_PLATFORMS},
        },
    )
    items = list(search.item_collection())
    print(f"[LS {year}] {len(items)} scenes (< {max_cloud_cover}% cloud cover)")
    return items


def _attach_spatial_ref(
    da: xr.DataArray,
    crs: str,
    transform: Affine,
    nodata: int | float = 0,
) -> xr.DataArray:
    """Attach spatial dimensions, CRS, affine transform, and nodata metadata."""
    return (
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        .rio.write_crs(crs, inplace=False)
        .rio.write_transform(transform, inplace=False)
        .rio.write_nodata(nodata, inplace=False)
    )


def _write_gtiff(
    path: Path,
    data: np.ndarray,
    crs: str,
    transform: Affine,
    dtype: str,
) -> None:
    """Write a 3D `(bands, height, width)` array to a tiled, deflate-compressed GeoTIFF."""
    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(data)


def _compute_composites_with_retry(
    year: int,
    items: list[pystac.Item],
    grid: Grid,
    max_attempts: int = 3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load, mask, composite, and align Landsat bands and validity mask with retries."""
    target_shape = (grid.height_px, grid.width_px)
    band_transforms = {
        band: _band_scale_offset(items[0], band) for band in LS_BANDS
    }

    for attempt in range(1, max_attempts + 1):
        signed_items = [
            planetary_computer.sign_inplace(pystac.Item.from_dict(item.to_dict()))
            for item in items
        ]

        try:
            with rasterio.Env(**GDAL_NET_ENV):
                ds = odc.stac.load(
                    signed_items,
                    bands=LS_BANDS + ["qa_pixel"],
                    bbox=DHAKA_BBOX_4326,
                    crs=grid.crs,
                    resolution=grid.res_m,
                    groupby="solar_day",
                    chunks={"x": 1024, "y": 1024},
                    resampling="nearest",
                )

                scaled_bands = []
                for band in LS_BANDS:
                    scale, offset = band_transforms[band]
                    raw = ds[band].astype("float32")
                    refl = raw.where(raw > 0) * scale + offset
                    scaled_bands.append(refl)

                refl_stack = xr.concat(scaled_bands, dim="band").assign_coords(
                    band=LS_BANDS
                )

                if "qa_pixel" in ds:
                    qa_uint = ds["qa_pixel"].fillna(QA_BAD_BITS_MASK).astype("uint32")
                    clear_mask = (qa_uint & QA_BAD_BITS_MASK) == 0
                else:
                    clear_mask = xr.ones_like(refl_stack.isel(band=0), dtype=bool)

                clean_refl = refl_stack.where(clear_mask)
                median_refl = clean_refl.median(dim="time", skipna=True)

                quantized_da = _attach_spatial_ref(
                    (median_refl * REFLECTANCE_SCALE_OUT)
                    .clip(min=0, max=65535)
                    .fillna(0)
                    .astype(np.uint16),
                    grid.crs,
                    grid.transform,
                    nodata=0,
                )

                valid_da = _attach_spatial_ref(
                    (clear_mask & refl_stack.notnull().all(dim="band"))
                    .any(dim="time")
                    .astype(np.uint8),
                    grid.crs,
                    grid.transform,
                    nodata=0,
                )

                aligned_bands = (
                    quantized_da.rio.reproject(
                        grid.crs,
                        shape=target_shape,
                        transform=grid.transform,
                        resampling=Resampling.nearest,
                        nodata=0,
                    )
                    .to_numpy()
                    .astype(np.uint16)
                )
                aligned_valid = (
                    valid_da.rio.reproject(
                        grid.crs,
                        shape=target_shape,
                        transform=grid.transform,
                        resampling=Resampling.nearest,
                        nodata=0,
                    )
                    .to_numpy()
                    .astype(np.uint8)
                )

            return aligned_bands, aligned_valid

        except Exception as exc:
            wait_s = 3 * attempt
            print(
                f"[LS {year}] attempt {attempt}/{max_attempts} failed ({exc}); "
                f"retrying in {wait_s}s..."
            )
            time.sleep(wait_s)

    print(f"[LS {year}] all {max_attempts} attempts failed; skipping year")
    return None


def download_year(
    year: int,
    out_root: Path,
    chip_px: int = 256,
    stride: int = 128,
    max_cloud_cover: float = 15.0,
    overwrite: bool = False,
) -> None:
    """Build 30 m winter median composite for `year` and write chips to `out_root`."""
    grid = dhaka_grid(res_m=30, bbox_4326=DHAKA_BBOX_4326, target_crs=TARGET_CRS)
    items = fetch_winter_items(
        year=year,
        bbox=DHAKA_BBOX_4326,
        max_cloud_cover=max_cloud_cover,
    )
    if not items:
        print(f"[LS {year}] no scenes found; skipping")
        return

    composites = _compute_composites_with_retry(year=year, items=items, grid=grid)
    if composites is None:
        return
    aligned_bands, aligned_valid = composites

    img_dir = out_root / "ls" / str(year) / "images"
    val_dir = out_root / "ls" / str(year) / "valid"
    img_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for row, col in chip_origins_from_shape(
        height_px=grid.height_px,
        width_px=grid.width_px,
        chip_px=chip_px,
        stride_px=stride,
    ):
        stem = f"ls_{year}_r{row:05d}_c{col:05d}"
        img_path = img_dir / f"{stem}.tif"
        val_path = val_dir / f"{stem}.tif"

        if img_path.exists() and val_path.exists() and not overwrite:
            continue

        img_chip = aligned_bands[:, row : row + chip_px, col : col + chip_px]
        val_chip = aligned_valid[np.newaxis, row : row + chip_px, col : col + chip_px]
        chip_tf = grid.chip_transform(row_px=row, col_px=col)

        _write_gtiff(img_path, img_chip, grid.crs, chip_tf, dtype="uint16")
        _write_gtiff(val_path, val_chip, grid.crs, chip_tf, dtype="uint8")
        n_written += 1

    print(
        f"[LS {year}] finished: {n_written} new chips written to {out_root / 'ls' / str(year)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 30 m Landsat Collection-2 L2 winter composite chips."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2010, 2016)),
        help="Target winter composite years (e.g., 2015 covers Nov 2014 - Mar 2015).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/chips"),
        help="Root output directory for generated chips.",
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
        "--max-cloud",
        type=float,
        default=15.0,
        help="Maximum scene-level cloud cover percentage for STAC search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chip GeoTIFF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = list(dict.fromkeys(args.years))
    for year in years:
        download_year(
            year=year,
            out_root=args.out,
            chip_px=args.chip_px,
            stride=args.stride,
            max_cloud_cover=args.max_cloud,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()