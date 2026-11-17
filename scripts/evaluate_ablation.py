"""Aggregate and compare test metrics across all variants of the 2x2 ablation study."""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description="Aggregate 2x2 ablation study results.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"), help="Path to config file.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory where evaluation metrics are saved.")
    parser.add_argument("--out", type=Path, default=Path("results/ablation_summary.csv"), help="Output summary CSV path.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    variants = cfg.get("ablation", {}).get("variants", [])
    seeds = cfg.get("seeds", [42])
    
    rows = []

    print("=== Aggregating Ablation Results ===")
    for variant in variants:
        v_name = variant["name"]
        has_boundary = variant["boundary"]
        has_oversample = variant["oversample"]

        for seed in seeds:
            run_name = f"abl_{v_name}_seed{seed}"
            # Expecting individual result JSONs or CSVs saved per run
            # e.g., results/abl_full_seed42/metrics.json
            metrics_path = args.results_dir / run_name / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path, "r") as mf:
                    data = json.load(mf)
                    rows.append({
                        "variant": v_name,
                        "boundary": has_boundary,
                        "oversample": has_oversample,
                        "seed": seed,
                        "f1": data.get("f1", 0.0),
                        "iou": data.get("iou", 0.0),
                    })
            else:
                print(f"[WARNING] Metrics not found for {run_name} at {metrics_path}")

    if not rows:
        print("\n[ERROR] No ablation metrics found. Ensure evaluation scripts have populated results/.")
        return

    df = pd.DataFrame(rows)
    
    # Group by ablation configuration and compute mean ± std (ddof=1)
    summary = (
        df.groupby(["variant", "boundary", "oversample"])[["f1", "iou"]]
        .agg(["mean", "std", "count"])
    )
    
    # Flatten multi-index columns for readability
    summary.columns = ["_".join(col).strip() for col in summary.columns.values()]
    summary = summary.reset_index()

    # Sort by IoU mean descending
    summary = summary.sort_values(by="iou_mean", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print(" 2x2 ABLATION STUDY SUMMARY (Sorted by Mean IoU)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)
    print(f"\nSummary successfully written to {args.out}")


if __name__ == "__main__":
    main()