import segmentation_models_pytorch as smp

def build_deeplabv3():
    return smp.DeepLabV3("resnet50", encoder_weights="imagenet", classes=1)

