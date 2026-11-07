import segmentation_models_pytorch as smp


def build_unet():
    return smp.Unet("resnet50", encoder_weights="imagenet", classes=1)