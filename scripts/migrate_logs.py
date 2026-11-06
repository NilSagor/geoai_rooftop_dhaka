from pathlib import Path
import pandas as pd

# Order you trained in (must match reality)
trained_order = {
    "unet": [42, 123, 456, 789, 1024],
    "deeplabv3": [42, 123, 456, 789],
}

for model, seeds in trained_order.items():
    src_root = Path(f"logs/{model}")
    if not src_root.exists():
        continue
    versions = sorted(src_root.glob("version_*"),
                      key=lambda p: int(p.name.split("_")[1]))
    if len(versions) != len(seeds):
        print(f"WARN {model}: {len(versions)} versions vs {len(seeds)} seeds — skipping")
        continue
    for seed, version_dir in zip(seeds, versions):
        dest = Path(f"logs/{model}_seed{seed}")
        if dest.exists():
            print(f"SKIP {dest} already exists")
            continue
        # Move contents of version_dir up one level to dest
        dest.mkdir(parents=True, exist_ok=True)
        for child in version_dir.iterdir():
            child.rename(dest / child.name)
        version_dir.rmdir()
        print(f"moved {version_dir} -> {dest}/")

print("\nAfter migration, is_done should skip these. Verify:")
for model, seeds in trained_order.items():
    for seed in seeds:
        csv = Path(f"logs/{model}_seed{seed}/metrics.csv")
        ckpt = Path(f"checkpoints/{model}_seed{seed}")
        n_epochs = 0
        if csv.exists():
            df = pd.read_csv(csv)
            if "epoch" in df.columns:
                n_epochs = df["epoch"].dropna().nunique()
        print(f"  {model:12s} seed={seed:<5d} epochs={n_epochs:>2d} "
              f"ckpt={'yes' if ckpt.exists() else 'no'}")