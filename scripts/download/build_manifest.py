"""Build data/manifest.csv from data/chips/.

Columns:
  stem, scale, year, res_m, row, col,
  origin_x, origin_y, pixel_size, crs,
  unit, footprint_m2, usable_area_m2, roof_ratio, valid_fraction,
  image_path, valid_path, mask_path, block_key
"""
import argparse
import os
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHIP_ROOT = ROOT / "data" / "chips"
DEFAULT_OUT = ROOT / "data" / "manifest.csv"
DEFAULT_UNITS = ROOT / "data" / "processed" / "admin_units.gpkg"

SCALE_RES: dict[str, int] = {"s2": 10, "ls": 30}
DEFAULT_USABLE_FRACTION: float = 0.70
STEM_RC_PATTERN = re.compile(r"r(\d+)_c(\d+)$")


def _rel_path(target: Path, base: Path = ROOT) -> str:
    """Return a path string relative to `base` when possible."""
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return os.path.relpath(target.resolve(), base.resolve())


def _parse_row_col(stem: str) -> tuple[int, int]:
    """Extract `(row, col)` integer pixel indices from a chip stem."""
    match = STEM_RC_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"Chip stem does not match expected 'r<row>_c<col>' suffix: {stem}")
    return int(match.group(1)), int(match.group(2))


def _list_stems(scale: str, year: int, chip_root: Path) -> list[str]:
    """List sorted chip stems for a given scale and year."""
    img_dir = chip_root / scale / str(year) / "images"
    if not img_dir.exists():
        return []
    return sorted(p.stem for p in img_dir.glob("*.tif"))


def _mask_path(scale: str, row: int, col: int, chip_root: Path) -> Path:
    """Resolve the corresponding building mask path for a chip scale and origin."""
    res_m = SCALE_RES[scale]
    return chip_root / "masks" / f"m{res_m}" / f"m{res_m}_r{row:05d}_c{col:05d}.tif"


def _read_raster_stats(path: Path, mode: str) -> tuple[float, float, int]:
    """Read `(primary_metric, secondary_metric, pixel_count)` for mask or valid rasters."""
    if not path.exists():
        return float("nan"), float("nan"), 0
    with rasterio.open(path) as src:
        arr = src.read(1)
    n_pixels = arr.size
    if n_pixels == 0:
        return float("nan"), float("nan"), 0
    if mode == "mask":
        roof_ratio = float((arr > 0).mean())
        return roof_ratio, 0.0, n_pixels
    valid_fraction = float((arr > 0).mean())
    return valid_fraction, 0.0, n_pixels


def _lookup_unit(
    cx: float,
    cy: float,
    crs: object,
    units: gpd.GeoDataFrame | None,
) -> str | None:
    """Resolve the administrative `unit` containing `(cx, cy)`."""
    if units is None or units.empty or crs is None:
        return None
    if "unit" not in units.columns:
        return None

    pt_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([cx], [cy]),
        crs=crs,
    ).to_crs(units.crs)

    joined = gpd.sjoin(
        pt_gdf,
        units[["unit", "geometry"]],
        predicate="within",
        how="left",
    )
    if not joined.empty and pd.notna(joined["unit"].iloc[0]):
        return str(joined["unit"].iloc[0])
    return None


def _read_meta(
    scale: str,
    year: int,
    stem: str,
    chip_root: Path,
    units: gpd.GeoDataFrame | None,
    usable_fraction: float = DEFAULT_USABLE_FRACTION,
) -> dict[str, object]:
    """Collect metadata and area metrics for a single chip."""
    res_m = SCALE_RES[scale]
    row, col = _parse_row_col(stem)

    img_path = chip_root / scale / str(year) / "images" / f"{stem}.tif"
    valid_path = chip_root / scale / str(year) / "valid" / f"{stem}.tif"
    mask_path = _mask_path(scale, row, col, chip_root)

    with rasterio.open(img_path) as src:
        bounds = src.bounds
        crs = src.crs
        transform = src.transform

    cx = (bounds.left + bounds.right) / 2.0
    cy = (bounds.bottom + bounds.top) / 2.0
    pixel_size = float(abs(transform.a))
    pixel_area_m2 = pixel_size * pixel_size

    roof_ratio, _, mask_pixels = _read_raster_stats(mask_path, mode="mask")
    valid_fraction, _, _ = _read_raster_stats(valid_path, mode="valid")

    if pd.notna(roof_ratio) and mask_pixels > 0:
        footprint_m2: float | None = float(roof_ratio * mask_pixels * pixel_area_m2)
        usable_area_m2: float | None = float(footprint_m2 * usable_fraction)
    else:
        footprint_m2 = None
        usable_area_m2 = None

    unit = _lookup_unit(cx=cx, cy=cy, crs=crs, units=units)
    block_key = f"{scale}_{row // 64:05d}_{col // 64:05d}"

    return {
        "stem": stem,
        "scale": scale,
        "year": year,
        "res_m": res_m,
        "row": row,
        "col": col,
        "origin_x": float(bounds.left),
        "origin_y": float(bounds.top),
        "pixel_size": pixel_size,
        "crs": str(crs),
        "unit": unit,
        "footprint_m2": footprint_m2,
        "usable_area_m2": usable_area_m2,
        "roof_ratio": roof_ratio,
        "valid_fraction": valid_fraction,
        "image_path": _rel_path(img_path),
        "valid_path": _rel_path(valid_path) if valid_path.exists() else "",
        "mask_path": _rel_path(mask_path) if mask_path.exists() else "",
        "block_key": block_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or incrementally update data/manifest.csv from data/chips/."
    )
    parser.add_argument(
        "--chip-root",
        type=Path,
        default=DEFAULT_CHIP_ROOT,
        help="Root directory containing s2/, ls/, and masks/ chip folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path for manifest CSV.",
    )
    parser.add_argument(
        "--units",
        type=Path,
        default=DEFAULT_UNITS,
        help="Optional administrative units vector file for spatial unit lookup.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=list(SCALE_RES.keys()),
        default=list(SCALE_RES.keys()),
        help="Sensor scales to index.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to specific years; defaults to all year folders present.",
    )
    parser.add_argument(
        "--usable-fraction",
        type=float,
        default=DEFAULT_USABLE_FRACTION,
        help="Fraction of roof footprint area usable for solar array placement.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild manifest from scratch instead of merging with existing CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chip_root: Path = args.chip_root
    out_path: Path = args.out
    units = gpd.read_file(args.units) if args.units.exists() else None

    existing_df: pd.DataFrame | None = None
    seen_keys: set[tuple[str, int, str]] = set()
    if out_path.exists() and not args.rebuild:
        existing_df = pd.read_csv(out_path)
        seen_keys = {
            (str(r.scale), int(r.year), str(r.stem))
            for r in existing_df[["scale", "year", "stem"]].itertuples(index=False)
        }

    new_rows: list[dict[str, object]] = []
    for scale in args.scales:
        scale_dir = chip_root / scale
        if not scale_dir.exists():
            print(f"[manifest] no {scale_dir}; skipping")
            continue

        years = (
            args.years
            if args.years is not None
            else sorted(int(p.name) for p in scale_dir.iterdir() if p.is_dir() and p.name.isdigit())
        )
        for year in years:
            stems = _list_stems(scale=scale, year=year, chip_root=chip_root)
            print(f"[manifest] {scale} {year}: {len(stems)} chips")
            for stem in stems:
                key = (scale, year, stem)
                if key in seen_keys:
                    continue
                new_rows.append(
                    _read_meta(
                        scale=scale,
                        year=year,
                        stem=stem,
                        chip_root=chip_root,
                        units=units,
                        usable_fraction=args.usable_fraction,
                    )
                )

    if not new_rows:
        print(f"[manifest] nothing new; {len(seen_keys)} chips already indexed")
        return

    new_df = pd.DataFrame(new_rows)
    combined_df = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if existing_df is not None
        else new_df
    )
    combined_df = combined_df.sort_values(["scale", "year", "stem"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path, index=False)
    print(f"[manifest] wrote {len(combined_df)} rows -> {out_path}")
    print(combined_df.groupby(["scale", "year"]).size().to_string())


if __name__ == "__main__":
    main()