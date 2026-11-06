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
MIN_EPOCHS = cfg["training"]["epochs"]

def is_done(model: str, seed: int) -> bool:
    ckpt_dir = Path(f"checkpoints/{model}_seed{seed}")
    # csv = Path(f"logs/{model}_seed{seed}/version_0/metrics.csv")
    csv = Path(f"logs/{model}_seed{seed}/metrics.csv")
    if not (ckpt_dir.exists() and csv.exists()):
        return False
    if not any(ckpt_dir.glob("*.ckpt")):
        return False
    df = pd.read_csv(csv)
    if "epoch" not in df.columns:
        return False
    return df["epoch"].dropna().nunique() >= MIN_EPOCHS


def main():
    for model in MODELS:
        for seed in SEEDS:
            if is_done(model, seed):
                print(f"SKIP {model} seed={seed} (complete)")
                continue
            print(f"=== TRAIN {model} seed={seed} ===")
            result = subprocess.run(
                [sys.executable, "scripts/train.py",
                 "--model", model, "--seed", str(seed)],
                check=False,
            )
            if result.returncode != 0:
                print(f"FAILED: {model} seed={seed}")

if __name__ == "__main__":
    main()