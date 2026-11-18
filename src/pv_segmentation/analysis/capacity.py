"""Cell-based module counting and installed capacity (paper §RSUF, Eq. for N_b and P_b)."""
import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.io_utils import load_config, load_rooftops, write_csv


def capacity_per_building(
    gdf: gpd.GeoDataFrame,
    rsuf: float,
    module_area_m2: float,
    module_wp: float,
) -> pd.DataFrame:
    """Calculate module count and capacity per building:
    N_b = floor(RSUF_b * A_u,b / a_m); P_b = N_b * P_m.
    """
    required_cols = {"unit", "usable_area_m2"}
    if not required_cols.issubset(gdf.columns):
        raise ValueError(f"GeoDataFrame missing required columns. Expected: {required_cols}")

    df = pd.DataFrame({
        "building_id": np.arange(len(gdf)),
        "unit": gdf["unit"].values,
        "usable_area_m2": gdf["usable_area_m2"].values,
    })
    df["rsuf"] = rsuf
    df["n_modules"] = np.floor(df["rsuf"] * df["usable_area_m2"] / module_area_m2).astype(int)
    df["capacity_wp"] = df["n_modules"] * module_wp
    return df


def aggregate_by_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate building-level capacity and module counts by administrative or spatial unit."""
    required_cols = {"unit", "building_id", "usable_area_m2", "capacity_wp"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns for aggregation. Expected: {required_cols}")

    return (
        df.groupby("unit", as_index=False)
        .agg(
            roofs=("building_id", "count"),
            usable_km2=("usable_area_m2", lambda x: x.sum() / 1e6),
            capacity_mwp=("capacity_wp", lambda x: x.sum() / 1e6),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate cell-based module counting and installed capacity.")
    parser.add_argument("--config", type=Path, default=Path("configs/economics.yaml"), help="Path to config YAML.")
    parser.add_argument("--rooftops", type=Path, default=None, help="Path to rooftops vector file.")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/capacity_by_building.csv"), help="Output path for building-level capacity.")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load configuration
    cfg_data = load_config(args.config)
    if "module" not in cfg_data:
        raise KeyError("Missing 'module' section in configuration file.")
    
    module_cfg = cfg_data["module"]
    required_module_keys = {"rsuf_conservative", "area_m2", "rated_wp"}
    missing_keys = required_module_keys - module_cfg.keys()
    if missing_keys:
        raise KeyError(f"Missing module configuration keys: {missing_keys}")

    # Load rooftops
    print("=== Loading rooftop inventory ===")
    gdf = load_rooftops(args.rooftops)

    # Compute building-level capacity
    print("=== Computing per-building module counts and capacity ===")
    per_b = capacity_per_building(
        gdf,
        rsuf=module_cfg["rsuf_conservative"],
        module_area_m2=module_cfg["area_m2"],
        module_wp=module_cfg["rated_wp"],
    )

    # Save outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(per_b, args.out)

    summary_path = args.out.with_name("capacity_by_unit.csv")
    summary = aggregate_by_unit(per_b)
    write_csv(summary, summary_path)

    print("\n=== Installed Capacity Summary by Unit ===")
    print(summary.to_string(index=False))
    print(f"\nBuilding-level results written to {args.out}")
    print(f"Unit summary written to {summary_path}")


if __name__ == "__main__":
    main()