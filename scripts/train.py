"""Train a single PV segmentation model."""
import argparse
import yaml
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from pv_segmentation.utils.determinism import setup_determinism
from pv_segmentation.data.datamodule import SolarDataModule
from pv_segmentation.lightning_module import SolarSegModule
from pv_segmentation.models import MODEL_BUILDERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 2 batches")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["seed"]
    
    setup_determinism(seed)

    dm = SolarDataModule(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )
    model = SolarSegModule(
        MODEL_BUILDERS[args.model](),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/{args.model}_seed{seed}",
        filename=f"{args.model}-{{epoch:02d}}-{{val_f1:.4f}}",
        monitor="val_f1", mode="max", save_top_k=1,
    )

    logger=CSVLogger("logs", name=f"{args.model}_seed{seed}")

    accelerator = "cpu"
    
    trainer = L.Trainer(
        max_epochs=1 if args.smoke else cfg["training"]["epochs"],
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        logger=logger,
        callbacks=[ckpt],
        deterministic=True,
        accelerator=accelerator,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()