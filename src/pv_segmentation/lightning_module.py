"""Shared LightningModule for all PV segmentation models."""
import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall,
    BinaryF1Score, BinaryJaccardIndex,
)




class SolarSegModule(L.LightningModule):
    def __init__(
            self, 
            model: torch.nn.Module, 
            lr: float = 1e-4, 
            weight_decay: float = 1e-4,
            lam: float = 0.3,
            epochs: int = 200,
            loss_fn = None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lam = lam
        self.epochs = epochs
        self.loss_fn = loss_fn  #  custom loss (e.g., BoundaryAwareLoss)
        
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Metrics
        self.val_acc = BinaryAccuracy()
        self.val_prec = BinaryPrecision()
        self.val_rec = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.test_f1 = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()

    def forward(self, x, return_edge: bool = False):
        # Check if the model supports boundary output and user requested it
        is_boundary_model = hasattr(self.model, "edge_head")
        
        if is_boundary_model and return_edge:
            out = self.model(x, return_edge=True)
        else:
            out = self.model(x)

        # Handle various output formats (dicts, tuples, tensors)
        if isinstance(out, dict):
            out = out.get("out", out.get("logits", out))
            
        if isinstance(out, tuple):
            mask, edge = out
            if mask.shape[1] != 1: 
                mask = mask[:, :1]
            if edge.shape[1] != 1: 
                edge = edge[:, :1]
            return mask, edge
            
        if out.shape[1] != 1:
            out = out[:, :1]
        return out

    def _shared_step(self, batch, stage: str):
        x, y = batch
        is_boundary_model = hasattr(self.model, "edge_head")

        if is_boundary_model:
            mask_logits, edge_logits = self(x, return_edge=True)
            
            if self.loss_fn is not None:
                loss, l_m, l_e, _ = self.loss_fn(mask_logits, edge_logits, y)
            else:
                # Fallback to standard mask_edge_loss if imported
                from pv_segmentation.losses import mask_edge_loss
                loss, l_m, l_e, _ = mask_edge_loss(mask_logits, edge_logits, y, lam=self.lam)
                
            logits = mask_logits
            
            # Log separate loss components for monitoring
            self.log(f"{stage}_mask_loss", l_m, on_step=False, on_epoch=True)
            self.log(f"{stage}_edge_loss", l_e, on_step=False, on_epoch=True)
        else:
            logits = self(x, return_edge=False)
            loss = F.binary_cross_entropy_with_logits(logits, y)

        preds = (torch.sigmoid(logits) > 0.5).long()
        y_long = y.long()

        if stage == "val":
            self.val_acc.update(preds, y_long)
            self.val_prec.update(preds, y_long)
            self.val_rec.update(preds, y_long)
            self.val_f1.update(preds, y_long)
            self.val_iou.update(preds, y_long)
        elif stage == "test":
            self.test_f1.update(preds, y_long)
            self.test_iou.update(preds, y_long)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        metrics = [
            ("acc", self.val_acc), 
            ("prec", self.val_prec),
            ("rec", self.val_rec), 
            ("f1", self.val_f1),
            ("iou", self.val_iou)
        ]
        for name, metric in metrics:
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}