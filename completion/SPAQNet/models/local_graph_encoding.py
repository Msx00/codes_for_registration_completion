"""Local graph encoding for source and partial liver point clouds."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .point_ops import _batch_gather, farthest_point_indices, fps_points, knn_indices


class LocalGraphEncodingInput(NamedTuple):
    source: torch.Tensor  # (B, Ns, 3), source-normalized
    partial: torch.Tensor  # (B, Np, 3), source-normalized and padded
    partial_mask: torch.Tensor  # (B, Np)


class LocalGraphEncodingOutput(NamedTuple):
    source_context: torch.Tensor  # (B, Cs, 3)
    partial_context: torch.Tensor  # (B, Cp, 3)
    source_features: torch.Tensor  # (B, Cs, D)
    partial_features: torch.Tensor  # (B, Cp, D)


class LocalGraphEncoder(nn.Module):
    """Point MLP plus an EdgeConv-style local aggregation."""

    def __init__(self, feature_dim: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        points: torch.Tensor,
        neighbor_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neighbor_indices is None:
            neighbor_indices = knn_indices(points, self.k_neighbors)
        point_features = self.point_mlp(points)
        neighbor_features = _batch_gather(point_features, neighbor_indices)
        neighbor_points = _batch_gather(points, neighbor_indices)
        center_features = point_features.unsqueeze(2).expand_as(neighbor_features)
        center_points = points.unsqueeze(2).expand_as(neighbor_points)
        edge_features = torch.cat(
            [
                neighbor_features - center_features,
                center_features,
                neighbor_points - center_points,
            ],
            dim=-1,
        )
        local_features = self.edge_mlp(edge_features).amax(dim=2)
        return self.norm(point_features + local_features), neighbor_indices


class LocalGraphEncoding(nn.Module):
    def __init__(self, feature_dim: int, k_neighbors: int, context_points: int):
        super().__init__()
        self.context_points = int(context_points)
        self.source_encoder = LocalGraphEncoder(feature_dim, k_neighbors)
        self.partial_encoder = LocalGraphEncoder(feature_dim, k_neighbors)

    def _select_partial_context(
        self, partial: torch.Tensor, partial_mask: torch.Tensor
    ) -> torch.Tensor:
        valid_counts = partial_mask.sum(dim=1)
        if bool((valid_counts < 2).any()):
            raise ValueError("Every partial cloud must contain at least 2 points")
        context_count = min(self.context_points, int(valid_counts.min().item()))
        selected = []
        for batch_index in range(partial.shape[0]):
            valid = partial[batch_index, partial_mask[batch_index]].unsqueeze(0)
            indices = farthest_point_indices(valid, context_count)[0]
            selected.append(valid[0, indices])
        return torch.stack(selected)

    def forward(self, inputs: LocalGraphEncodingInput) -> LocalGraphEncodingOutput:
        source_context, _ = fps_points(inputs.source, self.context_points)
        partial_context = self._select_partial_context(inputs.partial, inputs.partial_mask)
        source_features, _ = self.source_encoder(source_context)
        partial_features, _ = self.partial_encoder(partial_context)
        return LocalGraphEncodingOutput(
            source_context, partial_context, source_features, partial_features
        )
