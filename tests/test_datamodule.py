"""Phase 2 smoke test: DataModule yields correctly shaped batches."""
import pytest
from pv_segmentation.data.datamodule import SolarDataModule


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/processed/images").exists(),
    reason="Phase 1 processed data not available",
)
def test_datamodule_shapes():
    dm = SolarDataModule(batch_size=2, num_workers=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    image, mask = batch
    assert image.shape == (2, 3, 256, 256), f"image shape: {image.shape}"
    assert mask.shape == (2, 1, 256, 256), f"mask shape: {mask.shape}"
    assert image.dtype == __import__("torch").float32
    assert mask.dtype == __import__("torch").float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0