from pathlib import Path
import pandas as pd

MODELS = ["unet", "deeplabv3", "pspnet", "mask2former"]
SEEDS = [42, 123, 456, 789, 1024]

print(f"{'model':12s} {'seed':>5s}  {'epochs':>6s}  {'ckpt':>5s}  {'status':>10s}")
for m in MODELS:
    for s in SEEDS:
        ckpt_dir = Path(f"checkpoints/{m}_seed{s}")
        csv = Path(f"logs/{m}_seed{s}/version_0/metrics.csv")
        n_epochs = 0
        ckpt_ok = False
        if csv.exists():
            df = pd.read_csv(csv)
            if "epoch" in df.columns:
                n_epochs = df["epoch"].dropna().nunique()
        if ckpt_dir.exists():
            ckpt_ok = any(ckpt_dir.glob("*.ckpt"))
        status = "DONE" if n_epochs >= 30 and ckpt_ok else ("PARTIAL" if n_epochs > 0 else "MISSING")
        print(f"{m:12s} {s:>5d}  {n_epochs:>6d}  {str(ckpt_ok):>5s}  {status:>10s}")