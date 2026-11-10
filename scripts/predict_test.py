"""Generate test-set probabilities for every trained checkpoint.

For each (model, seed) writes:
  <out>/<model>_seed<seed>.npy   probabilities, (N, 1, H, W) float32
  <out>/<model>_seed<seed>.json  checkpoint path + its validation F1
The JSON is what evaluate_ensemble.py uses for the weighted vote.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from pv_segmentation.inference import (
    find_best_checkpoint, is_complete, load_model, predict_probabilities,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", nargs="+", help="override the model list")
    parser.add_argument("--seeds", nargs="+", type=int, help="override the seed list")
    parser.add_argument("--split", default="data/splits/fixed_seed42.json")
    parser.add_argument("--out", default="results/probabilities")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even if the .npy already exists")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="also use runs that have no DONE marker")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    models = args.models or cfg["models"]
    seeds = args.seeds or cfg["seeds"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for seed in seeds:
            tag = f"{model} seed={seed}"
            out_npy = out_dir / f"{model}_seed{seed}.npy"

            if out_npy.exists() and not args.overwrite:
                print(f"SKIP {tag} (already predicted)")
                continue
            if not args.allow_incomplete and not is_complete(model, seed):
                print(f"SKIP {tag} (no DONE marker; use --allow-incomplete to override)")
                continue
            try:
                ckpt, val_f1 = find_best_checkpoint(model, seed)
            except FileNotFoundError as e:
                print(f"SKIP {tag}: {e}")
                continue

            print(f"Loading {tag} from {ckpt} (val_f1={val_f1:.4f})")
            module = load_model(model, ckpt, device=args.device)
            probs = predict_probabilities(module, args.split, device=args.device)

            np.save(out_npy, probs)
            out_npy.with_suffix(".json").write_text(
                json.dumps({"checkpoint": str(ckpt), "val_f1": val_f1}))
            print(f"  saved {probs.shape} -> {out_npy}")

            del module
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()