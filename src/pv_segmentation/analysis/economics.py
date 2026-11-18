"""LCOE, NPV and one-at-a-time sensitivity (paper §Economics, Eq. for LCOE_u and NPV_u)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.io_utils import load_config, write_csv, UNITS


def lcoe(
    capex_per_kwp: float,
    opex_per_kwp_yr: float,
    generation_kwh_first_year: float,
    lifetime: int,
    discount: float,
    degradation: float,
    capacity_kwp: float,
) -> float:
    """Calculate Levelized Cost of Energy (LCOE).

    Formula: LCOE = [C0 + Σ O_y/(1+r)^y] / [Σ E1 (1-d)^(y-1)/(1+r)^y].
    """
    C0 = capex_per_kwp * capacity_kwp
    O = opex_per_kwp_yr * capacity_kwp
    E1 = generation_kwh_first_year
    
    num = C0 + sum(O / (1 + discount) ** y for y in range(1, lifetime + 1))
    den = sum(
        E1 * (1 - degradation) ** (y - 1) / (1 + discount) ** y
        for y in range(1, lifetime + 1)
    )
    return num / den if den > 0 else float("inf")


def npv(
    capex_per_kwp: float,
    opex_per_kwp_yr: float,
    generation_kwh_first_year: float,
    tariff: float,
    lifetime: int,
    discount: float,
    degradation: float,
    capacity_kwp: float,
) -> float:
    """Calculate Net Present Value (NPV)."""
    C0 = capex_per_kwp * capacity_kwp
    O = opex_per_kwp_yr * capacity_kwp
    E1 = generation_kwh_first_year
    
    cash = sum(
        (tariff * E1 * (1 - degradation) ** (y - 1) - O) / (1 + discount) ** y
        for y in range(1, lifetime + 1)
    )
    return cash - C0


def per_unit(df_gen: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute LCOE and NPV for every unit at its own capacity and yield."""
    eco = cfg.get("economics", {})
    required_eco_keys = {"capex_per_kwp", "opex_per_kwp_yr", "lifetime", "discount", "degradation", "tariff"}
    missing = required_eco_keys - eco.keys()
    if missing:
        raise KeyError(f"Missing economic configuration keys: {missing}")

    rows = []
    for _, r in df_gen.iterrows():
        cap_kwp = r["capacity_mwp"] * 1000.0
        e1 = r["yield_kwh_per_kwp"] * cap_kwp
        
        unit_lcoe = lcoe(
            eco["capex_per_kwp"], eco["opex_per_kwp_yr"], e1,
            eco["lifetime"], eco["discount"], eco["degradation"], cap_kwp,
        )
        unit_npv_per_kwp = npv(
            1.0, eco["opex_per_kwp_yr"], r["yield_kwh_per_kwp"],
            eco["tariff"], eco["lifetime"], eco["discount"],
            eco["degradation"], 1.0,
        )
        
        rows.append({
            "unit": r["unit"],
            "capacity_mwp": r["capacity_mwp"],
            "lcoe": unit_lcoe,
            "npv_per_kwp": unit_npv_per_kwp,
            "tariff": eco["tariff"],
            "viable": float(unit_lcoe < eco["tariff"]),
        })
    return pd.DataFrame(rows)


def tornado(df_gen: pd.DataFrame, cfg: dict, unit: str | None = None, delta: float = 0.2) -> pd.DataFrame:
    """Perform one-at-a-time sensitivity analysis for district or single unit LCOE."""
    eco = cfg.get("economics", {})
    if unit is None:
        cap = df_gen["capacity_mwp"].sum() * 1000.0
        e1 = (df_gen["yield_kwh_per_kwp"] * df_gen["capacity_mwp"] * 1000.0).sum()
    else:
        row = df_gen.set_index("unit").loc[unit]
        cap = row["capacity_mwp"] * 1000.0
        e1 = row["yield_kwh_per_kwp"] * cap

    base = lcoe(
        eco["capex_per_kwp"], eco["opex_per_kwp_yr"], e1,
        eco["lifetime"], eco["discount"], eco["degradation"], cap,
    )
    
    rows = []
    for key in ("capex_per_kwp", "opex_per_kwp_yr", "discount", "degradation"):
        lo = eco[key] * (1 - delta)
        hi = eco[key] * (1 + delta)
        
        args_lo = dict(eco)
        args_lo[key] = lo
        args_hi = dict(eco)
        args_hi[key] = hi
        
        l_lo = lcoe(
            args_lo["capex_per_kwp"], args_lo["opex_per_kwp_yr"], e1,
            args_lo["lifetime"], args_lo["discount"], args_lo["degradation"], cap,
        )
        l_hi = lcoe(
            args_hi["capex_per_kwp"], args_hi["opex_per_kwp_yr"], e1,
            args_hi["lifetime"], args_hi["discount"], args_hi["degradation"], cap,
        )
        
        rows.append({
            "parameter": key,
            "base": base,
            "lcoe_low": l_lo,
            "lcoe_high": l_hi,
            "delta_pct": 100.0 * (l_hi - l_lo) / base,
        })
        
    return pd.DataFrame(rows).sort_values("delta_pct", key=lambda s: s.abs(), ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Calculate LCOE, NPV, and one-at-a-time sensitivity.")
    parser.add_argument("--config", type=Path, default=Path("configs/economics.yaml"), help="Path to economics config YAML.")
    parser.add_argument("--generation", type=Path, default=Path("results/analysis/generation_by_unit.csv"), help="Path to generation CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/economics_by_unit.csv"), help="Output economics CSV path.")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.generation.exists():
        raise FileNotFoundError(f"Generation file not found: {args.generation}")

    cfg = load_config(args.config)
    df_gen = pd.read_csv(args.generation)
    
    required_cols = {"unit", "capacity_mwp", "yield_kwh_per_kwp"}
    if not required_cols.issubset(df_gen.columns):
        raise ValueError(f"Generation CSV missing required columns. Expected: {required_cols}")

    eco_df = per_unit(df_gen, cfg)
    tornado_df = tornado(df_gen, cfg)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(eco_df, args.out)
    write_csv(tornado_df, args.out.with_name("tornado.csv"))

    print("=== Per-Unit Economics Results ===")
    print(eco_df.to_string(index=False))
    currency = cfg.get("economics", {}).get("currency", "USD")
    print(f"\nDistrict LCOE (mean) = {eco_df['lcoe'].mean():.4f} {currency}/kWh")
    print(f"Results successfully saved to {args.out} and {args.out.with_name('tornado.csv')}")


if __name__ == "__main__":
    main()