"""Synchronized transforms for image-mask pairs."""
import torch
import torchvision.transforms.v2 as T


# Training: geometric augmentation only, applied to concatenated tensor
train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10, fill=0.0),
])

# Validation/test: no random transforms; only ensure float dtype
# val_transform = T.Compose([])
val_transform = None


def apply_sync_transform(image: torch.Tensor, mask: torch.Tensor, transform):
    """Apply transform to image and mask simultaneously.

    Concatenates along channel dim so both share identical random parameters.
    image: (3, H, W) float; mask: (1, H, W) float.
    """
    if transform is None:
        return image, mask
    
    combined = torch.cat([image, mask], dim=0)  # (4, H, W)
    combined = transform(combined)
    return combined[:3], combined[3:4]