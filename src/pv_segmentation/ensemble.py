"""Majority and weighted voting for ensemble predictions."""
import torch


def majority_vote(probs_list, threshold: float = 0.5) -> torch.Tensor:
    """Eq. 1: strict majority, i.e. at least floor(N/2) + 1 models predict positive.

    For odd N this equals ceil(N/2). For even N it requires more than half
    (N=4 -> 3), so ties are negative.

    probs_list: list of tensors, each (B, 1, H, W), values in [0, 1].
    Returns: (B, 1, H, W) binary float tensor.
    """
    if not probs_list:
        raise ValueError("probs_list is empty")
    n_needed = len(probs_list) // 2 + 1
    votes = sum((p > threshold).float() for p in probs_list)  # no (N, ...) stack
    return (votes >= n_needed).float()


def weighted_vote(probs_list, val_f1_scores, threshold: float = 0.5) -> torch.Tensor:
    """Eq. 2: weighted sum of probabilities with w_i = F1_i / sum(F1_j).

    val_f1_scores: one float per model, computed on the VALIDATION set.
    threshold: Tw in the paper (0.5).
    """
    if len(probs_list) != len(val_f1_scores):
        raise ValueError(
            f"{len(probs_list)} prob maps but {len(val_f1_scores)} weights")
    total = float(sum(val_f1_scores))
    if total <= 0:
        raise ValueError("val F1 weights must sum to a positive number")
    fused = sum((f1 / total) * p for f1, p in zip(val_f1_scores, probs_list))
    return (fused > threshold).float()