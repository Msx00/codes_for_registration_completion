"""Pure-PyTorch point-cloud indexing and sampling operations."""

from __future__ import annotations

import torch


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    view_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    return values[batch.view(view_shape), indices]


@torch.no_grad()
def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-PyTorch kNN indices; geometry is always evaluated in FP32."""
    num_points = points.shape[1]
    if num_points < 2:
        return torch.zeros(
            points.shape[0],
            num_points,
            1,
            dtype=torch.long,
            device=points.device,
        )
    k = min(int(k), num_points - 1)
    with torch.autocast(device_type=points.device.type, enabled=False):
        distances = torch.cdist(points.float(), points.float())
        diagonal = torch.arange(num_points, device=points.device)
        distances[:, diagonal, diagonal] = float("inf")
        return distances.topk(k=k, dim=-1, largest=False).indices


@torch.no_grad()
def farthest_point_indices(
    points: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Deterministic batched FPS in FP32."""
    batch_size, point_count, _ = points.shape
    sample_count = min(max(int(sample_count), 1), point_count)
    if sample_count == point_count:
        return torch.arange(point_count, device=points.device).expand(
            batch_size, -1
        )
    points_fp32 = points.float()
    centroid = points_fp32.mean(dim=1, keepdim=True)
    farthest = (points_fp32 - centroid).square().sum(dim=-1).argmax(dim=1)
    batch = torch.arange(batch_size, device=points.device)
    minimum_distance = torch.full(
        (batch_size, point_count),
        float("inf"),
        dtype=torch.float32,
        device=points.device,
    )
    selected = torch.empty(
        batch_size,
        sample_count,
        dtype=torch.long,
        device=points.device,
    )
    for index in range(sample_count):
        selected[:, index] = farthest
        center = points_fp32[batch, farthest].unsqueeze(1)
        distance = (points_fp32 - center).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        farthest = minimum_distance.argmax(dim=1)
    return selected


def fps_points(
    points: torch.Tensor,
    sample_count: int,
    features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    indices = farthest_point_indices(points, sample_count)
    sampled_points = _batch_gather(points, indices)
    sampled_features = (
        _batch_gather(features, indices) if features is not None else None
    )
    return sampled_points, sampled_features
