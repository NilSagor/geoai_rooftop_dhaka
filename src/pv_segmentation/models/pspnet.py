import segmentation_models_pytorch as smp

def build_pspnet():
    return smp.PSPNet("resnet50", encoder_weights="imagenet", classes=1)