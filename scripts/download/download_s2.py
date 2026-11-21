"""Download cloud-free 10 m Sentinel-2 median winter composites as chips.

One chip = 256x256 px in TARGET_CRS at 10 m. Writes:
- images/<stem>.tif  (5 bands: B02, B03, B04, B08, B11, uint16)
- valid/<stem>.tif   (1 band: uint8, 1 = cloud-free & valid data present)

Resumable: skips years where all expected chips already exist, and skips
individual chips that already exist unless `--overwrite` is passed.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import odc.stac
import planetary_computer
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import rioxarray  
import xarray as xr

from .stac_common import (
    DHAKA_BBOX_4326,
    TARGET_CRS,
    Grid,
    chip_origins_from_shape,
    dhaka_grid,
    open_catalog,
    search_stac,
)

S2_BANDS: list[str] = ["B02", "B03", "B04", "B08", "B11"]
VALID_SCL: list[int] = [4, 5, 7]  # 4=vegetation, 5=not vegetated/built, 7=unclassified

GDAL_NET_ENV: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "6",
    "GDAL_HTTP_RETRY_DELAY": "2.0",
    "GDAL_HTTP_TIMEOUT": "90",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}


def fetch_winter_items(
    year: int,
    bbox: tuple[float, float, float, float] = DHAKA_BBOX_4326,
    max_cloud_cover: float = 15.0,
) -> list[pystac.Item]:
    """Fetch Sentinel-2 L2A items for the Nov (year-1) through Mar (year) dry season."""
    catalog = open_catalog()
    date_range = f"{year - 1}-11-01/{year}-03-31"
    items = search_stac(
        catalog=catalog,
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        max_cloud_cover=max_cloud_cover,
    )
    print(f"[S2 {year}] {len(items)} scenes (< {max_cloud_cover}% cloud cover)")
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


def _is_year_complete(
    year: int,
    out_root: Path,
    grid: Grid,
    chip_px: int,
    stride: int,
) -> bool:
    """Return True if every expected image and valid chip for `year` exists."""
    img_dir = out_root / "s2" / str(year) / "images"
    val_dir = out_root / "s2" / str(year) / "valid"
    if not img_dir.exists() or not val_dir.exists():
        return False

    origins = list(
        chip_origins_from_shape(
            height_px=grid.height_px,
            width_px=grid.width_px,
            chip_px=chip_px,
            stride_px=stride,
        )
    )
    if not origins:
        return False

    for row, col in origins:
        stem = f"s2_{year}_r{row:05d}_c{col:05d}"
        img_path = img_dir / f"{stem}.tif"
        val_path = val_dir / f"{stem}.tif"
        if not (img_path.exists() and val_path.exists()):
            return False
        if img_path.stat().st_size == 0 or val_path.stat().st_size == 0:
            return False

    return True


def _load_single_item_to_grid(
    item: pystac.Item,
    grid: Grid,
    max_attempts: int = 3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and align one STAC item onto `grid` with retries and fresh SAS signing."""
    target_shape = (grid.height_px, grid.width_px)
    item_id = item.id

    for attempt in range(1, max_attempts + 1):
        signed_item = planetary_computer.sign_inplace(
            pystac.Item.from_dict(item.to_dict())
        )
        try:
            with rasterio.Env(**GDAL_NET_ENV):
                ds = odc.stac.load(
                    [signed_item],
                    bands=S2_BANDS + ["SCL"],
                    bbox=DHAKA_BBOX_4326,
                    crs=grid.crs,
                    resolution=grid.res_m,
                    groupby="solar_day",
                    chunks={"x": 1024, "y": 1024},
                    resampling="nearest",
                ).compute()

                if "time" in ds.dims and ds.sizes["time"] > 0:
                    ds_step = ds.isel(time=0)
                else:
                    ds_step = ds

                scl_valid = ds_step["SCL"].isin(VALID_SCL)
                band_da = ds_step[S2_BANDS].to_array(dim="band").astype(np.float32)
                valid_2d = (scl_valid & band_da.notnull().all(dim="band")).astype(np.uint8)

                masked_bands = band_da.where(scl_valid)
                masked_bands = _attach_spatial_ref(
                    masked_bands, grid.crs, grid.transform, nodata=np.nan
                )
                valid_da = _attach_spatial_ref(
                    valid_2d, grid.crs, grid.transform, nodata=0
                )

                reproj_bands = (
                    masked_bands.rio.reproject(
                        grid.crs,
                        shape=target_shape,
                        transform=grid.transform,
                        resampling=Resampling.nearest,
                        nodata=np.nan,
                    )
                    .to_numpy()
                    .astype(np.float32)
                )
                reproj_valid = (
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
                reproj_valid = (reproj_valid > 0).astype(np.uint8)
                reproj_bands[:, reproj_valid == 0] = np.nan
                return reproj_bands, reproj_valid

        except Exception as exc:
            if attempt < max_attempts:
                time.sleep(2 * attempt)
            else:
                print(f"[S2] skipping scene {item_id} after {max_attempts} failures: {exc}")

    return None


def _compute_composites_from_items(
    year: int,
    items: list[pystac.Item],
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build winter median composite per scene so single-scene failures do not abort."""
    scene_bands: list[np.ndarray] = []
    scene_valids: list[np.ndarray] = []

    for item in items:
        res = _load_single_item_to_grid(item=item, grid=grid)
        if res is not None:
            b_arr, v_arr = res
            scene_bands.append(b_arr)
            scene_valids.append(v_arr)

    if not scene_bands:
        print(f"[S2 {year}] no usable scenes loaded; skipping year")
        return None

    print(f"[S2 {year}] compositing {len(scene_bands)}/{len(items)} usable scenes")
    stack_bands = np.stack(scene_bands, axis=0)   # (S, B, H, W)
    stack_valids = np.stack(scene_valids, axis=0)  # (S, H, W)

    median_float = np.nanmedian(stack_bands, axis=0)  # (B, H, W)
    median_uint16 = np.clip(np.nan_to_num(median_float, nan=0.0), 0, 65535).astype(
        np.uint16
    )
    any_valid = (stack_valids.max(axis=0) > 0).astype(np.uint8)
    median_uint16[:, any_valid == 0] = 0

    return median_uint16, any_valid


def download_year(
    year: int,
    out_root: Path,
    chip_px: int = 256,
    stride: int = 128,
    max_cloud_cover: float = 15.0,
    overwrite: bool = False,
) -> None:
    """Build winter median composite for `year` and write 10 m chips to `out_root`."""
    grid = dhaka_grid(res_m=10, bbox_4326=DHAKA_BBOX_4326, target_crs=TARGET_CRS)

    if not overwrite and _is_year_complete(
        year=year,
        out_root=out_root,
        grid=grid,
        chip_px=chip_px,
        stride=stride,
    ):
        print(f"[S2 {year}] already complete; skipping")
        return

    items = fetch_winter_items(
        year=year,
        bbox=DHAKA_BBOX_4326,
        max_cloud_cover=max_cloud_cover,
    )
    if not items:
        print(f"[S2 {year}] no scenes found; skipping")
        return

    composites = _compute_composites_from_items(year=year, items=items, grid=grid)
    if composites is None:
        return
    aligned_bands, aligned_valid = composites

    img_dir = out_root / "s2" / str(year) / "images"
    val_dir = out_root / "s2" / str(year) / "valid"
    img_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for row, col in chip_origins_from_shape(
        height_px=grid.height_px,
        width_px=grid.width_px,
        chip_px=chip_px,
        stride_px=stride,
    ):
        stem = f"s2_{year}_r{row:05d}_c{col:05d}"
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
        if n_written % 100 == 0:
            print(f"[S2 {year}] {n_written} chips written")

    print(
        f"[S2 {year}] finished: {n_written} new chips written to {out_root / 's2' / str(year)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download cloud-free 10 m Sentinel-2 winter composite chips."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2016, 2027)),
        help="Target winter composite years (e.g., 2025 covers Nov 2024 - Mar 2025).",
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