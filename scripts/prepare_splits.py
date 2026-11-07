import json, random
from pathlib import Path

ids = sorted([p.stem for p in Path("data/processed/images").glob("*.tif")])
rng = random.Random(42)
rng.shuffle(ids)
n = len(ids)
train_end = int(0.7 * n)
val_end = int(0.8 * n)
split = {
    "train": ids[:train_end],
    "val": ids[train_end:val_end],
    "test": ids[val_end:],
}
Path("data/splits").mkdir(parents=True, exist_ok=True)
with open("data/splits/fixed_seed42.json", "w") as f:
    json.dump(split, f, indent=2)
print(f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

