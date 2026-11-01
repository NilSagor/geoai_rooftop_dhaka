"""Phase 0 smoke test: package imports and determinism runs."""
from pv_segmentation.utils.determinism import setup_determinism


def test_determinism_setup():
    setup_determinism(42)
    assert True
