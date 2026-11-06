from .unet import build_unet
from .deeplabv3 import build_deeplabv3
from .pspnet import build_pspnet
from .mask2former import build_mask2former
from .sam_lora import build_sam_lora

MODEL_BUILDERS = {
    "unet": build_unet,
    "deeplabv3": build_deeplabv3,
    "pspnet": build_pspnet,
    "mask2former": build_mask2former,
    "sam_lora": build_sam_lora,
}