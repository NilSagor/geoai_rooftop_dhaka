"""Raster -> polygon -> footprint intersection -> setback (paper §Vectorization)."""
import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from analysis.io_utils import write_csv


def raster_to_polygons(mask_path: str | Path, min_area_m2: float = 20.0, simplify_m: float = 0.5) -> gpd.GeoDataFrame:
    """Convert a binary raster mask into simplified polygon geometries."""
    mask_p = Path(mask_path)
    if not mask_p.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_p}")

    with rasterio.open(mask_p) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms = []
    for geom, val in shapes(arr.astype("uint8"), mask=arr > 0, transform=transform):
        if val == 0:
            continue
        g = shape(geom)
        if g.area < min_area_m2:
            continue
        geoms.append(g.simplify(simplify_m, preserve_topology=True))
        
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)


def apply_setback(gdf: gpd.GeoDataFrame, setback_m: float) -> gpd.GeoDataFrame:
    """Apply negative buffer for spatial setback and calculate usable area."""
    if setback_m <= 0:
        gdf["usable_area_m2"] = gdf.geometry.area
        return gdf

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(-setback_m)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf["usable_area_m2"] = gdf.geometry.area
    return gdf


def intersect_footprints(roofs: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame, unit_field: str = "unit") -> gpd.GeoDataFrame:
    """Keep only roof polygons that intersect a building footprint; tag each with its unit."""
    if unit_field not in footprints.columns:
        raise KeyError(f"Footprints GeoDataFrame missing required unit field: '{unit_field}'")

    if roofs.crs != footprints.crs:
        footprints = footprints.to_crs(roofs.crs)

    joined = gpd.sjoin(roofs, footprints[[unit_field, "geometry"]], how="inner", predicate="intersects")
    
    # Drop index_right if present and remove duplicate entries based on original geometry index
    if "index_right" in joined.columns:
        joined = joined.drop(columns="index_right")
        
    subset_col = joined.index.name if joined.index.name else None
    joined = joined.drop_duplicates(subset=subset_col)
    
    return joined


def main():
    parser = argparse.ArgumentParser(description="Vectorize rooftop rasters, intersect with footprints, and apply setbacks.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary rooftop raster (GeoTIFF).")
    parser.add_argument("--footprints", type=Path, required=True, help="Building footprints GeoJSON.")
    parser.add_argument("--setback-m", type=float, default=0.5, help="Setback distance in meters.")
    parser.add_argument("--unit-field", type=str, default="unit", help="Attribute field name for unit identifier.")
    parser.add_argument("--out", type=Path, default=Path("results/inventory/rooftops.geojson"), help="Output GeoJSON path.")
    args = parser.parse_args()

    if not args.mask.exists():
        raise FileNotFoundError(f"Mask file not found: {args.mask}")
    if not args.footprints.exists():
        raise FileNotFoundError(f"Footprints file not found: {args.footprints}")

    print(f"=== Vectorizing mask from {args.mask} ===")
    roofs = raster_to_polygons(args.mask)
    
    print(f"=== Loading footprints from {args.footprints} ===")
    fp = gpd.read_file(args.footprints)
    
    if roofs.crs != fp.crs:
        print(f"Reprojecting footprints from CRS {fp.crs} to match raster CRS {roofs.crs}")
        fp = fp.to_crs(roofs.crs)

    print("=== Performing spatial intersection with footprints ===")
    roofs = intersect_footprints(roofs, fp, unit_field=args.unit_field)

    print(f"=== Applying {args.setback_m}m structural setback ===")
    roofs = apply_setback(roofs, args.setback_m)
    roofs["footprint_m2"] = roofs.geometry.area

    args.out.parent.mkdir(parents=True, exist_ok=True)
    roofs.to_file(args.out, driver="GeoJSON")
    print(f"Successfully wrote {len(roofs)} processed rooftops -> {args.out}")


if __name__ == "__main__":
    main()