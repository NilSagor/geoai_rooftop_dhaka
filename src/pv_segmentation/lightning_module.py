"""Shared LightningModule for all PV segmentation models."""
import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall,
    BinaryF1Score, BinaryJaccardIndex,
)


class SolarSegModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        self.val_acc = BinaryAccuracy()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.test_f1 = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()

    def forward(self, x):
        out = self.model(x)
        # Mask2Former wrapper returns dict; smp models return tensor
        if isinstance(out, dict):
            out = out["out"] if "out" in out else out["logits"]
        return out

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        if logits.shape[1] != 1:
            logits = logits[:, :1]  # safety for models with aux heads
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).long()

        if stage == "val":
            self.val_acc.update(preds, y.long())
            self.val_prec.update(preds, y.long())
            self.val_rec.update(preds, y.long())
            self.val_f1.update(preds, y.long())
            self.val_iou.update(preds, y.long())
        elif stage == "test":
            self.test_f1.update(preds, y.long())
            self.test_iou.update(preds, y.long())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        for name, metric in [("acc", self.val_acc), ("prec", self.val_prec),
                              ("rec", self.val_rec), ("f1", self.val_f1),
                              ("iou", self.val_iou)]:
            self.log(f"val_{name}", metric.compute(), prog_bar=True)
            metric.reset()

    def on_test_epoch_end(self):
        self.log("test_f1", self.test_f1.compute())
        self.log("test_iou", self.test_iou.compute())
        self.test_f1.reset()
        self.test_iou.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        return {"optimizer": opt, "lr_scheduler": sched}