import json
import random
from pathlib import Path
from collections import defaultdict

img_dir, mask_dir = Path("data/processed/images"), Path("data/processed/masks")
ids = sorted(p.stem for p in img_dir.glob("*.tif"))

missing = [i for i in ids if not (mask_dir / f"{i}.tif").exists()]
assert not missing, f"images without masks: {missing[:5]}"
assert len(ids) == len(list(mask_dir.glob("*.tif"))), "image/mask counts differ"

TRAIN, VAL = 0.70, 0.15   # test = remaining 0.15  (use VAL = 0.10 for 70/10/20)

# group by roof type 
# prefix before the first underscore: (brick / concrete / steel)
groups = defaultdict(list)
for i in ids:
    groups[i.split("_")[0]].append(i)

rng = random.Random(42)
split = {"train": [], "val": [], "test": []}

for roof in sorted(groups):
    g = groups[roof]            
    n = len(g)
    a = int(TRAIN * n)
    b = int((TRAIN + VAL) * n)
    split["train"] += g[:a]
    split["val"] += g[a:b]
    split["test"] += g[b:]

for k in split:
    split[k].sort()

# sanity checks: no overlap, nothing lost
all_ids = split["train"] + split["val"] + split["test"]
assert len(all_ids) == len(set(all_ids)) == len(ids), "overlap or missing ids"

Path("data/splits").mkdir(parents=True, exist_ok=True)
with open("data/splits/fixed_seed42.json", "w") as f:
    json.dump(split, f, indent=2)

for k, v in split.items():
    by_roof = {r: sum(i.startswith(r + "_") for i in v) for r in sorted(groups)}
    print(f"{k}: {len(v)} {by_roof}")