"""Shared I/O helpers: load rooftops, units, ancillary climate, configs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
INVENTORY = RESULTS / "inventory"

UNITS = ["DNCC", "DSCC", "Savar", "Keraniganj", "Dhamrai", "Dohar", "Nawabganj"]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_rooftops(path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Per-building rooftop polygons with at least `unit`, `footprint_m2`, `usable_area_m2`."""
    path = Path(path) if path else INVENTORY / "rooftops.geojson"
    gdf = gpd.read_file(path)
    required = {"unit", "footprint_m2", "usable_area_m2"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"rooftops.geojson missing columns: {missing}")
    if gdf.crs is None:
        raise ValueError("rooftops.geojson has no CRS; set it to the projected CRS used for areas.")
    return gdf


def load_irradiance(csv_path: str | Path) -> pd.DataFrame:
    """Hourly POA irradiance (W/m2), ambient temp (°C), wind (m/s).

    Expects columns: timestamp, G_poa, T_amb, wind.
    Assign to buildings by unit (district-mean fallback) unless a raster is provided.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    for c in ("G_poa", "T_amb", "wind"):
        if c not in df.columns:
            raise ValueError(f"irradiance csv missing column {c!r}")
    return df.sort_values("timestamp").reset_index(drop=True)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)