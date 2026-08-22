"""Geometry-aware encoding of source and partial feature memory."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .point_ops import _batch_gather, knn_indices


class GeometricMemoryEncoderInput(NamedTuple):
    source_context: torch.Tensor  # (B, Cs, 3)
    partial_context: torch.Tensor  # (B, Cp, 3)
    initialized_memory: torch.Tensor  # (B, Cs + Cp, D)


class GeometricMemoryEncoderOutput(NamedTuple):
    memory: torch.Tensor  # (B, Cs + Cp, D)
    source_features: torch.Tensor  # (B, Cs, D)
    partial_features: torch.Tensor  # (B, Cp, D)
    global_features: torch.Tensor  # (B, 1, D)


class GeometryEncoderBlock(nn.Module):
    """Fuse global self-attention with kNN-relative local geometry."""

    def __init__(self, feature_dim: int, num_heads: int, k_neighbors: int):
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.self_norm = nn.LayerNorm(feature_dim)
        self.self_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.local_norm = nn.LayerNorm(feature_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.merge = nn.Linear(feature_dim * 2, feature_dim)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(features)
        global_features, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        local_input = self.local_norm(features)
        neighbors = _batch_gather(local_input, neighbor_indices)
        neighbor_positions = _batch_gather(positions, neighbor_indices)
        centers = local_input.unsqueeze(2).expand_as(neighbors)
        relative_positions = neighbor_positions - positions.unsqueeze(2)
        local_features = self.local_mlp(
            torch.cat([neighbors - centers, centers, relative_positions], dim=-1)
        ).amax(dim=2)
        features = features + self.merge(
            torch.cat([global_features, local_features], dim=-1)
        )
        return features + self.ffn(self.ffn_norm(features))


class GeometricMemoryEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        k_neighbors: int,
        encoder_depth: int,
    ):
        super().__init__()
        self.memory_encoder = nn.ModuleList(
            [
                GeometryEncoderBlock(feature_dim, num_heads, k_neighbors)
                for _ in range(max(int(encoder_depth), 1))
            ]
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self, inputs: GeometricMemoryEncoderInput
    ) -> GeometricMemoryEncoderOutput:
        memory_positions = torch.cat(
            [inputs.source_context, inputs.partial_context], dim=1
        )
        memory = inputs.initialized_memory
        memory_neighbors = knn_indices(
            memory_positions, self.memory_encoder[0].k_neighbors
        )
        for block in self.memory_encoder:
            memory = block(memory, memory_positions, memory_neighbors)
        source_count = inputs.source_context.shape[1]
        source_features = memory[:, :source_count]
        partial_features = memory[:, source_count:]
        source_global = source_features.amax(dim=1)
        partial_global = partial_features.amax(dim=1)
        global_features = self.global_fusion(
            torch.cat([source_global, partial_global], dim=-1)
        ).unsqueeze(1)
        return GeometricMemoryEncoderOutput(
            memory, source_features, partial_features, global_features
        )
