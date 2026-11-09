"""Train a single PV segmentation model."""
import argparse
import os
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from pv_segmentation.utils.determinism import setup_determinism
from pv_segmentation.data.datamodule import SolarDataModule
from pv_segmentation.lightning_module import SolarSegModule
from pv_segmentation.models import MODEL_BUILDERS


def load_config(path: str, model: str) -> dict:
    """Load the YAML and apply per-model overrides to data/training."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ov = cfg.get("overrides", {}).get(model, {})
    if "batch_size" in ov:
        cfg["data"]["batch_size"] = ov["batch_size"]
    if "accumulate_grad_batches" in ov:
        cfg["training"]["accumulate_grad_batches"] = ov["accumulate_grad_batches"]
    return cfg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore last.ckpt and start from scratch")
    args = parser.parse_args()

    cfg = load_config(args.config, args.model)
    data = cfg["data"]
    tr = cfg["training"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    
    # Needed for deterministic cuBLAS; must be set before the first CUDA call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    setup_determinism(seed)

    run_name = f"{args.model}_seed{seed}" + ("_smoke" if args.smoke else "")
    ckpt_dir = Path("checkpoints") / run_name

    dm = SolarDataModule(
        chip_size=data["chip_size"],
        stride=data["stride"],
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        split_strategy=cfg["split_strategy"],
        seed=seed,
    )
    model = SolarSegModule(
        MODEL_BUILDERS[args.model](),
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
    )

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{args.model}-{{epoch:02d}}-{{val_f1:.4f}}",
        monitor="val_f1", mode="max", save_top_k=1,
        save_last=True,          
    )

    max_epochs = 1 if args.smoke else tr["epochs"]
    trainer = L.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        accumulate_grad_batches=tr.get("accumulate_grad_batches", 1),
        precision=tr.get("precision", "32-true"),
        accelerator=tr["accelerator"],
        deterministic=tr.get("deterministic", True),
        logger=CSVLogger("logs", name=run_name),
        callbacks=[checkpoint],
    )

    last = ckpt_dir / "last.ckpt"
    resume = last if (last.exists() and not args.no_resume and not args.smoke) else None
    if resume:
        print(f"Resuming from {resume}")
    trainer.fit(model, datamodule=dm, ckpt_path=resume)

    if not args.smoke:
        (ckpt_dir / "DONE").write_text(str(max_epochs))


if __name__ == "__main__":
    main()