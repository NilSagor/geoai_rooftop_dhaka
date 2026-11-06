import lightning as L
import torch



def setup_determinism(seed: int = 42) -> None:
    """Seed all RNGs and configure PyTorch for deterministic execution.

    Note: torch.use_deterministic_algorithms(warn_only=True) permits
    F.interpolate(mode='bilinear') on CUDA to run, but it will not be
    bitwise deterministic. This limitation is documented in the README.
    """
    L.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True