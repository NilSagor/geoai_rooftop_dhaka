"""Model registry for PV segmentation."""
import segmentation_models_pytorch as smp
import torch.nn as nn

def build_segformer():
    """Segformer using MiT backbone (mit_b2) natively supported by smp."""
    return smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        classes=1,
    )
