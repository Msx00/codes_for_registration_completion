"""Supervised losses and metrics for aligned registration points."""

import torch
import torch.nn.functional as F


def point_rmse(pred, gt):
    """Return point-wise RMSE in millimetres for aligned point clouds."""
    squared_l2 = (pred.float() - gt.float()).square().sum(dim=-1)
    return torch.sqrt(squared_l2.mean())


def pointwise_huber_loss(pred, gt, beta_mm=5.0):
    """Apply Smooth L1 to each aligned point pair's Euclidean error."""
    per_point_l2 = (pred.float() - gt.float()).norm(p=2, dim=-1)
    return F.smooth_l1_loss(
        per_point_l2,
        torch.zeros_like(per_point_l2),
        beta=beta_mm,
        reduction="mean",
    )
