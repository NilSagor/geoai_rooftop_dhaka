import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class BoundaryAwareUNet(nn.Module):
    """Clean dual-head U-Net using SMP's modular encoder and decoder.

    Avoids orphaned layers by instantiating the encoder and decoder directly
    without initializing a default segmentation head.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        decoder_channels: tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        
        # 1. Instantiate encoder directly
        self.encoder = get_encoder(
            encoder_name,
            weights=encoder_weights,
            depth=len(decoder_channels),
            in_channels=3,
        )
        
        # 2. Instantiate decoder directly using encoder skip connection channels
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        # 3. Direct 1x1 heads from final decoder features (16 channels)
        final_channels = decoder_channels[-1]
        self.mask_head = nn.Conv2d(final_channels, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(final_channels, 1, kernel_size=1)

    def forward(self, x, return_edge: bool = False):
        features = self.encoder(x)
        dec = self.decoder(*features)       # (B, 16, H, W)
        
        mask = self.mask_head(dec)          # (B,  1, H, W)
        if return_edge:
            edge = self.edge_head(dec)      # (B,  1, H, W)
            return mask, edge
        return mask


def build_boundary_unet():
    return BoundaryAwareUNet(encoder_name="resnet34", encoder_weights="imagenet")