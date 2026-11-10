"""Single-model and ensemble test metrics per seed (weights from validation F1)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torchmetrics.functional.classification import (
    binary_f1_score, binary_jaccard_index,
)

from pv_segmentation.ensemble import majority_vote, weighted_vote
from pv_segmentation.inference import load_targets


def score(pred_binary: torch.Tensor, target: torch.Tensor) -> dict:
    pred, tgt = pred_binary.long(), target.long()
    return {
        "f1": float(binary_f1_score(pred, tgt)),
        "iou": float(binary_jaccard_index(pred, tgt)),
    }


def load_run(probs_dir: Path, model: str, seed: int):
    """Return (probabilities, val_f1) for a run, or None if not predicted yet."""
    npy = probs_dir / f"{model}_seed{seed}.npy"
    meta = npy.with_suffix(".json")
    if not (npy.exists() and meta.exists()):
        return None
    probs = torch.from_numpy(np.load(npy))
    return probs, float(json.loads(meta.read_text())["val_f1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="data/splits/fixed_seed42.json")
    parser.add_argument("--probs", default="results/probabilities")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    models, seeds = cfg["models"], cfg["seeds"]
    probs_dir, out_dir = Path(args.probs), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = load_targets(args.split)
    rows = []

    for seed in seeds:
        runs = {}
        for model in models:
            run = load_run(probs_dir, model, seed)
            if run is None:
                print(f"MISSING {model} seed={seed}")
                continue
            if run[0].shape != target.shape:
                raise ValueError(
                    f"{model} seed={seed}: probs {tuple(run[0].shape)} "
                    f"!= targets {tuple(target.shape)}")
            runs[model] = run

        for model, (probs, _) in runs.items():
            rows.append({"method": "single", "model": model, "n_models": 1,
                         "seed": seed, **score((probs > 0.5).float(), target)})

        if len(runs) < 2:
            continue

        names = list(runs)
        probs_list = [runs[n][0] for n in names]
        val_f1s = [runs[n][1] for n in names]
        label = "+".join(names)

        results = {
            "majority": score(majority_vote(probs_list), target),
            "weighted": score(weighted_vote(probs_list, val_f1s), target),
        }
        for method, m in results.items():
            rows.append({"method": method, "model": label, "n_models": len(names),
                         "seed": seed, **m})
        print(f"seed={seed}: majority F1={results['majority']['f1']:.4f}  "
              f"weighted F1={results['weighted']['f1']:.4f}  (n={len(names)})")

    if not rows:
        sys.exit("No probabilities found. Run scripts/predict_test.py first.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ensemble_metrics.csv", index=False)

    # mean ± sample std (ddof=1) across seeds, same convention for every row
    summary = (df.groupby(["method", "model", "n_models"])[["f1", "iou"]]
                 .agg(["mean", "std", "count"]))
    summary.to_csv(out_dir / "ensemble_summary.csv")
    print("\n" + summary.round(4).to_string())
    print(f"\nSaved {len(df)} rows to {out_dir / 'ensemble_metrics.csv'}")


if __name__ == "__main__":
    main()