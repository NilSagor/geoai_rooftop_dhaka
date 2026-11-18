"""Hourly PV yield with cell-temperature derating (paper §Yield)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.io_utils import load_config, load_irradiance, write_csv

G_STC = 1000.0
T_STC = 25.0


def cell_temperature(T_amb: np.ndarray, G_poa: np.ndarray, wind: np.ndarray, U0: float, U1: float) -> np.ndarray:
    """Calculate cell temperature using the Faiman model: T_c = T_a + G / (U0 + U1 * v)."""
    return T_amb + G_poa / (U0 + U1 * wind)


def hourly_efficiency(eta_stc: float, beta: float, T_cell: np.ndarray) -> np.ndarray:
    """Calculate temperature-adjusted module efficiency: eta_t = eta_STC * [1 - beta (T_c - 25)]."""
    return eta_stc * (1.0 - beta * (T_cell - T_STC))


def annual_generation(capacity_wp: float, climate: pd.DataFrame, cfg: dict, temp_correction: bool = True) -> float:
    """Return annual energy (kWh) for a capacity, applying temperature + system loss."""
    required_cols = {"G_poa", "T_amb", "wind"}
    if not required_cols.issubset(climate.columns):
        raise ValueError(f"Climate dataframe missing required columns: {required_cols - set(climate.columns)}")

    G = climate["G_poa"].to_numpy()
    Ta = climate["T_amb"].to_numpy()
    v = climate["wind"].to_numpy()
    
    eta_stc = cfg["eta_stc"]
    beta = cfg["beta_per_C"]
    loss = cfg["system_loss"]

    if temp_correction:
        Tc = cell_temperature(Ta, G, v, cfg["U0"], cfg["U1"])
        eta_t = hourly_efficiency(eta_stc, beta, Tc)
        factor = eta_t / eta_stc
    else:
        factor = np.ones_like(G)

    hourly_kwh = (capacity_wp / 1000.0) * (G / G_STC) * factor * (1.0 - loss)
    return float(hourly_kwh.sum())


def main():
    parser = argparse.ArgumentParser(description="Calculate hourly PV yield with cell-temperature derating.")
    parser.add_argument("--config", type=Path, default=Path("configs/economics.yaml"), help="Path to economics/yield config YAML.")
    parser.add_argument("--capacity", type=Path, default=Path("results/analysis/capacity_by_unit.csv"), help="Path to capacity CSV.")
    parser.add_argument("--climate", type=Path, required=True, help="Path to hourly climate CSV with G_poa, T_amb, wind.")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/generation_by_unit.csv"), help="Output generation CSV path.")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.capacity.exists():
        raise FileNotFoundError(f"Capacity file not found: {args.capacity}")
    if not args.climate.exists():
        raise FileNotFoundError(f"Climate file not found: {args.climate}")

    # Load configuration and data
    full_cfg = load_config(args.config)
    if "yield" not in full_cfg:
        raise KeyError("Missing 'yield' section in configuration file.")
    cfg = full_cfg["yield"]

    cap = pd.read_csv(args.capacity)
    required_cap_cols = {"unit", "capacity_mwp"}
    if not required_cap_cols.issubset(cap.columns):
        raise ValueError(f"Capacity CSV missing required columns. Expected: {required_cap_cols}")

    climate = load_irradiance(args.climate)

    rows = []
    for _, r in cap.iterrows():
        cap_wp = r["capacity_mwp"] * 1e6
        E_temp = annual_generation(cap_wp, climate, cfg, temp_correction=True)
        E_notemp = annual_generation(cap_wp, climate, cfg, temp_correction=False)
        
        reduction_pct = 0.0 if E_notemp == 0 else 100.0 * (1.0 - E_temp / E_notemp)
        
        rows.append({
            "unit": r["unit"],
            "capacity_mwp": r["capacity_mwp"],
            "generation_gwh": E_temp / 1e6,
            "generation_gwh_no_temp": E_notemp / 1e6,
            "yield_kwh_per_kwp": E_temp / (r["capacity_mwp"] * 1000.0),
            "temp_reduction_pct": reduction_pct,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, args.out)

    print("=== Annual PV Generation & Temperature Derating ===")
    print(df.to_string(index=False))
    print(f"\nResults successfully written to {args.out}")


if __name__ == "__main__":
    main()