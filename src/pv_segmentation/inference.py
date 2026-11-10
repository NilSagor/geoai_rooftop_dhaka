"""Load checkpoints and generate test-set probability maps."""
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pv_segmentation.data.dataset import SolarChipDataset
from pv_segmentation.lightning_module import SolarSegModule
from pv_segmentation.models import MODEL_BUILDERS
from pv_segmentation.transforms import val_transform

CKPT_ROOT = Path("checkpoints")

# Lightning renders '{val_f1:.4f}' as 'val_f1=0.8123' in the filename, e.g.
# 'unet-epoch=12-val_f1=0.8123.ckpt' (or '...-v1.ckpt' on a name clash).
_VAL_F1_RE = re.compile(r"val_f1=(\d+\.\d+)")


def run_dir(model_name: str, seed: int) -> Path:
    return CKPT_ROOT / f"{model_name}_seed{seed}"


def is_complete(model_name: str, seed: int) -> bool:
    """True if train.py finished this run (it writes DONE at the end)."""
    return (run_dir(model_name, seed) / "DONE").exists()


def find_best_checkpoint(model_name: str, seed: int) -> tuple[Path, float]:
    """Return (path, val_f1) of the best checkpoint for a (model, seed) pair.

    last.ckpt has no val_f1 in its name, so it is skipped automatically.
    """
    ckpt_dir = run_dir(model_name, seed)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory at {ckpt_dir}")

    candidates = []
    for path in ckpt_dir.glob("*.ckpt"):
        match = _VAL_F1_RE.search(path.stem)
        if match:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No best-model .ckpt files in {ckpt_dir}")

    val_f1, path = max(candidates, key=lambda t: t[0])
    return path, val_f1


def load_model(
    model_name: str,
    ckpt_path: str | Path,
    device: str = "cpu",
    strict: bool = True,
) -> SolarSegModule:
    """Rebuild the architecture, load the checkpoint, move to device, eval mode.

    strict=False silently ignores missing keys, which can leave randomly
    initialised layers in the model. Only relax it after inspecting the
    missing/unexpected keys.
    """
    backbone = MODEL_BUILDERS[model_name]()
    module = SolarSegModule.load_from_checkpoint(
        str(ckpt_path),
        model=backbone,
        map_location=device,
        strict=strict,
    )
    return module.to(device).eval()


def _test_loader(split_file: str, data_root: str, batch_size: int) -> DataLoader:
    ds = SolarChipDataset(data_root, split_file, "test", val_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.inference_mode()
def predict_probabilities(
    module: SolarSegModule,
    split_file: str,
    data_root: str = "data/processed",
    batch_size: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    """Sigmoid probabilities for the test split, shape (N, 1, H, W) float32."""
    all_probs = []
    for image, _ in _test_loader(split_file, data_root, batch_size):
        logits = module(image.to(device))
        if logits.shape[1] > 1:
            logits = logits[:, :1]
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def load_targets(
    split_file: str,
    data_root: str = "data/processed",
    batch_size: int = 4,
) -> torch.Tensor:
    """Test-set masks as (N, 1, H, W) float {0,1}.

    Read through the same dataset and transform as predict_probabilities, so
    targets and predictions are aligned by construction (same order, same
    cropping/resizing).
    """
    masks = []
    for _, mask in _test_loader(split_file, data_root, batch_size):
        mask = torch.as_tensor(mask)
        if mask.ndim == 3:  # (B, H, W) -> (B, 1, H, W)
            mask = mask.unsqueeze(1)
        masks.append(mask.float())
    targets = torch.cat(masks, dim=0)

    if not torch.isin(targets, torch.tensor([0.0, 1.0])).all():
        raise ValueError("Test masks are not binary {0,1}; check dataset/transforms.")
    return targets