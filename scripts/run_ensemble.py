"""Run ensemble evaluation across models and seeds, computing final test F1 and IoU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchmetrics
import yaml

from pv_segmentation.inference import (
    find_best_checkpoint,
    load_model,
    predict_probabilities,
    load_targets,
)
from pv_segmentation.ensemble import majority_vote, weighted_vote


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble predictions on the test split.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"), help="Path to config file.")
    parser.add_argument("--split-file", type=Path, default=None, help="Explicit path to split JSON file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    parser.add_argument("--out", type=Path, default=Path("results/ensemble_scores.json"), help="Output path for scores.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    models_to_ensemble = cfg.get("models", [])
    seeds = cfg.get("seeds", [42])
    data_cfg = cfg.get("data", {})
    batch_size = data_cfg.get("batch_size", 4)
    data_root = "data/processed"

    # Determine split file path
    if args.split_file is None:
        split_dir = Path("data/splits")
        split_strategy = cfg.get("split_strategy", "fixed")
        fixed_seed = 42
        split_name = f"fixed_seed{fixed_seed}.json" if split_strategy == "fixed" else f"per_seed_{cfg.get('seed', 42)}.json"
        split_file = split_dir / split_name
    else:
        split_file = args.split_file

    print(f"=== Loading ground truth targets from {split_file} ===")
    targets = load_targets(str(split_file), data_root=data_root, batch_size=batch_size)
    targets_gpu = targets.to(args.device)

    probs_list = []
    val_scores = []

    print("=== Collecting test probabilities and validation scores ===")
    for model_name in models_to_ensemble:
        for seed in seeds:
            try:
                ckpt_path, val_score = find_best_checkpoint(model_name, seed)
                print(f"Loading {model_name} (seed {seed}) from {ckpt_path} (Val Score: {val_score:.4f})")
                
                module = load_model(model_name, ckpt_path, device=args.device)
                probs = predict_probabilities(module, str(split_file), data_root=data_root, batch_size=batch_size, device=args.device)
                
                probs_list.append(torch.tensor(probs, dtype=torch.float32))
                val_scores.append(val_score)
            except FileNotFoundError as e:
                print(f"[WARNING] Skipping {model_name} seed {seed}: {e}")

    if not probs_list:
        raise RuntimeError("No valid model checkpoints found for ensemble evaluation!")

    print(f"=== Running ensemble over {len(probs_list)} models/seeds ===")

    # 1. Majority Voting Ensemble
    preds_maj = majority_vote(probs_list, threshold=0.5).to(args.device)
    
    # 2. Weighted Voting Ensemble
    preds_wt = weighted_vote(probs_list, val_scores, threshold=0.5).to(args.device)

    # Compute Metrics using torchmetrics
    f1_metric = torchmetrics.classification.BinaryF1Score().to(args.device)
    iou_metric = torchmetrics.classification.BinaryJaccardIndex().to(args.device)

    results = {}
    for name, preds in [("majority_vote", preds_maj), ("weighted_vote", preds_wt)]:
        # Flatten for metric calculation
        p_flat = preds.long().squeeze(1)
        t_flat = targets_gpu.long().squeeze(1)
        
        f1 = f1_metric(p_flat, t_flat).item()
        iou = iou_metric(p_flat, t_flat).item()
        
        results[name] = {"f1": f1, "iou": iou}
        print(f"[{name.upper()}] Test F1: {f1:.4f} | Test IoU: {iou:.4f}")

    # Save scores to JSON
    args.out.write_text(json.dumps(results, indent=4))
    print(f"=== Ensemble scores successfully written to {args.out} ===")


if __name__ == "__main__":
    main()