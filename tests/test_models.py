"""Phase 3 smoke test: all five models build and forward-pass."""
import torch
import pytest
from pv_segmentation.models import MODEL_BUILDERS


@pytest.mark.parametrize("name", ["unet", "deeplabv3", "pspnet", "mask2former"])
def test_model_forward_shape(name):
    model = MODEL_BUILDERS[name]()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    assert out.shape[0] == 1
    assert out.shape[-2:] == (256, 256)



@pytest.mark.skipif(
    not __import__("pathlib").Path("checkpoints/sam_vit_b_01ec64.pth").exists(),
    reason="SAM checkpoint not downloaded",
)
def test_sam_lora_forward_shape():
    model = MODEL_BUILDERS["sam_lora"]()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256)