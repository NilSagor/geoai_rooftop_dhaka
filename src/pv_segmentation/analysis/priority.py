"""Capacity-only vs capacity+suitability rankings (paper §Priority, H4)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from analysis.io_utils import load_config, write_csv, UNITS


def normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series safely."""
    denom = s.max() - s.min()
    if denom == 0 or pd.isna(denom):
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / denom


def compute_ranking(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Compute weighted multi-criteria suitability ranking."""
    out = df.copy()
    for c, w in weights.items():
        if c not in out.columns:
            raise KeyError(f"Weight feature '{c}' not found in input dataframe columns.")
        out[f"{c}_norm"] = normalize(out[c])
    
    out["score"] = sum(out[f"{c}_norm"] * w for c, w in weights.items())
    out["rank"] = out["score"].rank(ascending=False).astype(int)
    return out.sort_values("rank")


def main():
    parser = argparse.ArgumentParser(description="Evaluate capacity-only vs capacity+suitability rankings (H4).")
    parser.add_argument("--config", type=Path, default=Path("configs/economics.yaml"), help="Path to config YAML.")
    parser.add_argument("--inputs", type=Path, default=Path("results/analysis/suitability_inputs.csv"), help="Path to inputs CSV.")
    parser.add_argument("--out", type=Path, default=Path("results/analysis/priority.csv"), help="Output CSV path.")
    args = parser.parse_args()

    # Validate paths
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.inputs.exists():
        raise FileNotFoundError(f"Inputs file not found: {args.inputs}")

    # Load configuration
    cfg_data = load_config(args.config)
    priority_cfg = cfg_data.get("priority")
    if not priority_cfg:
        raise KeyError("Missing 'priority' section in configuration file.")

    weights = priority_cfg.get("weights")
    top_k = priority_cfg.get("top_k", 10)
    rho_threshold = priority_cfg.get("rho_threshold", 0.7)

    # Load data
    df = pd.read_csv(args.inputs)
    if "unit" not in df.columns:
        raise ValueError("Inputs CSV must contain a 'unit' column.")
    if "capacity_mwp" not in df.columns:
        raise ValueError("Inputs CSV must contain 'capacity_mwp' for capacity-only baseline ranking.")

    df = df.set_index("unit")

    # Compute rankings
    cap_only = compute_ranking(df.assign(score=df["capacity_mwp"]), {"capacity_mwp": 1.0})
    cap_suit = compute_ranking(df, weights)

    # Statistical evaluation
    rho, p_rho = spearmanr(cap_only["rank"], cap_suit["rank"])
    tau, _ = kendalltau(cap_only["rank"], cap_suit["rank"])
    
    overlap = len(set(cap_only.index[:top_k]) & set(cap_suit.index[:top_k]))

    merged = cap_only[["rank"]].rename(columns={"rank": "rank_capacity"}).join(
        cap_suit[["rank", "score"]].rename(columns={"rank": "rank_suitability"})
    )

    # Save and output results
    write_csv(merged.reset_index(), args.out)

    print("=== Priority Ranking & H4 Validation ===")
    print(merged)
    print(f"\nSpearman rho={rho:.4f} (p={p_rho:.3g}); Kendall tau={tau:.4f}; "
          f"top-{top_k} overlap={overlap}/{top_k}; threshold={rho_threshold}")
    print("H4 supported" if rho < rho_threshold else "H4 not supported")
    print(f"\nResults successfully written to {args.out}")


if __name__ == "__main__":
    main()