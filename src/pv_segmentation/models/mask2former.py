import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation


class Mask2FormerBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-ade-semantic",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        out = self.model(pixel_values=x)
        masks = out.masks_queries_logits.sigmoid()  # (B, Q, H/4, W/4)
        binary = masks.max(dim=1, keepdim=True).values  # (B, 1, H/4, W/4)
        return F.interpolate(
            binary, size=x.shape[-2:], mode="bilinear", align_corners=False
        )  # (B, 1, H, W)


def build_mask2former():
    return Mask2FormerBinary()