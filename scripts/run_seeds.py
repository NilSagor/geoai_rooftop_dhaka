"""Run every configured model across every configured seed."""
import subprocess
import sys
import argparse
from pathlib import Path

import yaml

def is_done(model: str, seed: int, epochs: int) -> bool:
    """A run is complete only if train.py wrote DONE with the same epoch budget."""
    marker = Path("checkpoints") / f"{model}_seed{seed}" / "DONE"
    return marker.exists() and marker.read_text().strip() == str(epochs)

# def find_metrics_csv(model: str, seed: int) -> Path | None:
#     """Return the newest metrics.csv for a run, handling flat and versioned layouts."""
#     run_dir = Path(f"logs/{model}_seed{seed}")
#     candidates = list(run_dir.glob("metrics.csv")) + list(run_dir.glob("version_*/metrics.csv"))
#     if not candidates:
#         return None
#     return max(candidates, key=lambda p: p.stat().st_mtime)


# def is_done(model: str, seed: int) -> bool:
#     ckpt_dir = Path(f"checkpoints/{model}_seed{seed}")

#     if not any(ckpt_dir.glob("*.ckpt")):
#         return False
    
#     csv = find_metrics_csv(model, seed)
#     if csv is None:
#         return False
    
#     try:
#         df = pd.read_csv(csv)
#     except (pd.errors.EmptyDataError, pd.errors.ParserError):
#         return False
#     if "epoch" not in df.columns:
#         return False
    
    
#     # csv = Path(f"logs/{model}_seed{seed}/version_0/metrics.csv")
#     # csv = Path(f"logs/{model}_seed{seed}/metrics.csv")
#     # if not (ckpt_dir.exists() and csv.exists()):
#     #     return False
#     # if not any(ckpt_dir.glob("*.ckpt")):
#     #     return False
#     # df = pd.read_csv(csv)
#     # if "epoch" not in df.columns:
#     #     return False
#     # return df["epoch"].dropna().nunique() >= MIN_EPOCHS
#     return df["epoch"].dropna().max() >= EPOCHS - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models", nargs="+", help="override the model list")
    parser.add_argument("--seeds", nargs="+", type=int, help="override the seed list")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = args.models or cfg["models"]
    seeds = args.seeds or cfg["seeds"]
    epochs = cfg["training"]["epochs"]

    failed = []
    for model in models:
        for seed in seeds:
            if not args.smoke and is_done(model, seed, epochs):
                print(f"SKIP {model} seed={seed} (complete)")
                continue
            print(f"=== TRAIN {model} seed={seed} ===", flush=True)
            cmd = [sys.executable, "scripts/train.py", "--model", model,
                   "--seed", str(seed), "--config", args.config]
            if args.smoke:
                cmd.append("--smoke")
            if subprocess.run(cmd, check=False).returncode != 0:
                print(f"FAILED: {model} seed={seed}")
                failed.append((model, seed))

    if failed:
        print(f"\n{len(failed)} run(s) failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()