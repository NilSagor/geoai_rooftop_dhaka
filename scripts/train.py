"""Train a single PV segmentation model."""
import argparse
import os
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pv_segmentation.utils.determinism import setup_determinism
from pv_segmentation.data.datamodule import SolarDataModule
from pv_segmentation.lightning_module import SolarSegModule
from pv_segmentation.models import MODEL_BUILDERS
from pv_segmentation.losses import BoundaryAwareLoss


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
    parser = argparse.ArgumentParser(description="Train a PV segmentation model.")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-suffix", type=str, default=None, help="Optional custom name suffix for logs/checkpoints")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore last.ckpt and start from scratch")
    parser.add_argument("--no-oversample", action="store_true", help="Disable oversampling for this run")
    args = parser.parse_args()

    cfg = load_config(args.config, args.model)
    if args.no_oversample:
        cfg["oversample"]["enabled"] = False
    data = cfg["data"]
    tr = cfg["training"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    
    # Needed for deterministic cuBLAS; must be set before the first CUDA call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    setup_determinism(seed)

    # Setup run naming and directory paths
    base_run_name = f"{args.model}_seed{seed}"
    if args.run_suffix:
        run_name = args.run_suffix
    else:
        run_name = base_run_name + ("_smoke" if args.smoke else "")
        
    ckpt_dir = Path("checkpoints") / run_name

    # dm = SolarDataModule(
    #     chip_size=data["chip_size"],
    #     stride=data["stride"],
    #     batch_size=data["batch_size"],
    #     num_workers=data["num_workers"],
    #     split_strategy=cfg["split_strategy"],
    #     seed=seed,
    # )

    dm = SolarDataModule(
        chip_size=data["chip_size"],
        stride=data["stride"],
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        split_strategy=cfg["split_strategy"],
        seed=seed,
        oversample=cfg["oversample"]["enabled"],
        oversample_factor=cfg["oversample"]["factor"],
        corrugated_threshold=cfg["oversample"]["corrugated_threshold"],
    )

    # Instantiate model and attach boundary loss if it's a boundary-aware network
    model_instance = MODEL_BUILDERS[args.model]()
    is_boundary_model = hasattr(model_instance, "edge_head")

    loss_fn = BoundaryAwareLoss(lam=0.3, edge_pos_weight=5.0) if is_boundary_model else None

    max_epochs = 1 if args.smoke else tr["epochs"]
    
    # Initialize LightningModule with correct epoch horizon for CosineAnnealingLR
    model = SolarSegModule(
        model=model_instance,
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
        epochs=max_epochs,
        loss_fn=loss_fn
    )

    # checkpoint = ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename=f"{args.model}-{{epoch:02d}}-{{val_f1:.4f}}",
    #     monitor="val_f1", mode="max", save_top_k=1,
    #     save_last=True,          
    # )

    # Checkpoint on validation IoU (matching paper standards), tracking both IoU and F1 in filename
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{args.model}-{{epoch:02d}}-{{val_iou:.4f}}-{{val_f1:.4f}}",
        monitor="val_iou", 
        mode="max", 
        save_top_k=1,
        save_last=True,          
    )

    
    # Early stopping on validation IoU to prevent unnecessary epochs
    early_stopping = EarlyStopping(
        monitor="val_iou", 
        mode="max", 
        patience=15, 
        verbose=True
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        accumulate_grad_batches=tr.get("accumulate_grad_batches", 1),
        precision=tr.get("precision", "32-true"),
        accelerator=tr["accelerator"],
        deterministic=tr.get("deterministic", True),
        logger=CSVLogger("logs", name=run_name),
        callbacks=[checkpoint, early_stopping],
    )

    # Handle checkpoint resuming
    last = ckpt_dir / "last.ckpt"
    resume = last if (last.exists() and not args.no_resume and not args.smoke) else None
    if resume:
        print(f"Resuming from {resume}")
        
    trainer.fit(model, datamodule=dm, ckpt_path=resume)

    if not args.smoke:
        (ckpt_dir / "DONE").write_text(str(max_epochs))
        print(f"=== Training complete for {run_name} ===")


if __name__ == "__main__":
    main()