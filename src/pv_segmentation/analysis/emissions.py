"""Avoided operational CO2 emissions calculation (paper §Emissions, Eq. C_u = E_u · EF)."""
import argparse
from pathlib import Path
import pandas as pd
import yaml


def compute_avoided_emissions(generation_csv: str | Path, emission_factor: float) -> pd.DataFrame:
    """Load generation data by unit and compute avoided CO2 emissions in kilotons.
    
    Formula: C_u = E_u (GWh) * 10^6 (kWh/GWh) * EF (kg/kWh) / 10^9 (kg/kt)
    Result is in kt CO2.
    """
    gen_path = Path(generation_csv)
    if not gen_path.exists():
        raise FileNotFoundError(f"Generation file not found: {gen_path}")

    df = pd.read_csv(gen_path)
    required_cols = {"unit", "generation_gwh"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in generation CSV. Expected: {required_cols}, found: {list(df.columns)}")

    # Compute avoided kt CO2: (GWh * 1e6 * EF) / 1e9 = kt CO2
    df["avoided_kt_co2"] = df["generation_gwh"] * 1e6 * emission_factor / 1e9
    return df[["unit", "generation_gwh", "avoided_kt_co2"]]


def main():
    parser = argparse.ArgumentParser(description="Calculate avoided operational CO2 emissions by unit.")
    parser.add_argument("--config", type=Path, default=Path("configs/economics.yaml"), help="Path to economics config YAML.")
    parser.add_argument("--generation", type=Path, default=Path("results/analysis/generation_by_unit.csv"), help="Path to generation CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/emissions_by_unit.csv"), help="Output CSV path.")
    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    emission_factor = config_data.get("economics", {}).get("emission_factor_kg_per_kwh")
    if emission_factor is None:
        raise KeyError("Missing 'emission_factor_kg_per_kwh' under 'economics' in config file.")

    # Compute emissions
    df_results = compute_avoided_emissions(args.generation, float(emission_factor))

    # Save output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(args.out, index=False)

    print(f"=== Avoided Operational CO2 Emissions (EF = {emission_factor} kg/kWh) ===")
    print(df_results.to_string(index=False))
    print(f"\nResults successfully written to {args.out}")


if __name__ == "__main__":
    main()