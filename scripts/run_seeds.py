"""Run all five models across five seeds."""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
# sam_lora memory problem


# MODELS = ["unet", "deeplabv3", "pspnet", "mask2former", "sam_lora"]
MODELS = ["unet", "deeplabv3", "pspnet", "mask2former"]
SEEDS = [42, 123, 456, 789, 1024]

with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)

EPOCHS = cfg["training"]["epochs"]


def find_metrics_csv(model: str, seed: int) -> Path | None:
    """Return the newest metrics.csv for a run, handling flat and versioned layouts."""
    run_dir = Path(f"logs/{model}_seed{seed}")
    candidates = list(run_dir.glob("metrics.csv")) + list(run_dir.glob("version_*/metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_done(model: str, seed: int) -> bool:
    ckpt_dir = Path(f"checkpoints/{model}_seed{seed}")

    if not any(ckpt_dir.glob("*.ckpt")):
        return False
    
    csv = find_metrics_csv(model, seed)
    if csv is None:
        return False
    
    try:
        df = pd.read_csv(csv)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if "epoch" not in df.columns:
        return False
    
    
    # csv = Path(f"logs/{model}_seed{seed}/version_0/metrics.csv")
    # csv = Path(f"logs/{model}_seed{seed}/metrics.csv")
    # if not (ckpt_dir.exists() and csv.exists()):
    #     return False
    # if not any(ckpt_dir.glob("*.ckpt")):
    #     return False
    # df = pd.read_csv(csv)
    # if "epoch" not in df.columns:
    #     return False
    # return df["epoch"].dropna().nunique() >= MIN_EPOCHS
    return df["epoch"].dropna().max() >= EPOCHS - 1


def main():
    failed = []
    for model in MODELS:
        for seed in SEEDS:
            if is_done(model, seed):
                print(f"SKIP {model} seed={seed} (complete)")
                continue
            print(f"=== TRAIN {model} seed={seed} ===")
            result = subprocess.run(
                [sys.executable, "scripts/train.py", "--model", model, "--seed", str(seed)],
                check=False,
            )
            if result.returncode != 0:
                print(f"FAILED: {model} seed={seed}")
                failed.append((model, seed))

    if failed:
        print(f"\n{len(failed)} run(s) failed: {failed}")
        sys.exit(1)

if __name__ == "__main__":
    main()