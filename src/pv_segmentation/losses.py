import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    """Combined loss containing:
      1. Mask Loss = BCE + Dice Loss (for overlap and crisp boundaries)
      2. Edge Loss = BCE with pos_weight (to handle sparse boundaries)
    """
    def __init__(self, lam: float = 0.3, edge_pos_weight: float = 5.0, dice_smooth: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.edge_pos_weight = edge_pos_weight
        self.dice_smooth = dice_smooth

        # Register Sobel filters as buffers for automatic device placement
        self.register_buffer(
            "_SOBEL_X", 
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "_SOBEL_Y", 
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def sobel_edges(self, mask: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(mask, self._SOBEL_X, padding=1)
        gy = F.conv2d(mask, self._SOBEL_Y, padding=1)
        mag = gx.abs() + gy.abs()
        return (mag > 0).float()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes Soft-Dice loss on probabilities."""
        probs = torch.sigmoid(logits)
        
        # Flatten tensors to shape (B, -1) to compute per-batch overlap
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        return (1.0 - dice).mean()

    def forward(self, mask_logits, edge_logits, y):
        y = y.float()
        e = self.sobel_edges(y)
        
        # 1. Mask Loss (BCE + Dice)
        bce_mask = F.binary_cross_entropy_with_logits(mask_logits, y)
        d_mask = self.dice_loss(mask_logits, y)
        l_mask = bce_mask + d_mask  # You can weight them like 0.5 * bce + 0.5 * dice if desired
        
        # 2. Edge Loss (BCE with pos_weight)
        pos_weight = torch.tensor([self.edge_pos_weight], device=edge_logits.device)
        l_edge = F.binary_cross_entropy_with_logits(edge_logits, e, pos_weight=pos_weight)
        
        total_loss = l_mask + self.lam * l_edge
        return total_loss, l_mask.detach(), l_edge.detach(), e