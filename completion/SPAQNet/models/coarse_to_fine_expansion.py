"""Coarse-to-mid-to-fine expansion of generated liver points."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn


class CoarseToFineExpansionInput(NamedTuple):
    coarse: torch.Tensor  # (B, Q, 3)
    coarse_features: torch.Tensor  # (B, Q, D)
    partial_context: torch.Tensor  # (B, Cp, 3)
    partial_features: torch.Tensor  # (B, Cp, D)
    global_features: torch.Tensor  # (B, 1, D)


class CoarseToFineExpansionOutput(NamedTuple):
    mid: torch.Tensor  # (B, Nmid, 3)
    fine: torch.Tensor  # (B, Nout, 3)


class PointExpansionBlock(nn.Module):
    """Expand generated parent points with learned local child offsets."""

    def __init__(
        self,
        feature_dim: int,
        max_children: int,
        max_offset_ratio: float,
    ):
        super().__init__()
        self.max_children = int(max_children)
        self.max_offset_ratio = float(max_offset_ratio)
        child_dim = 16
        self.child_embedding = nn.Embedding(self.max_children, child_dim)
        input_dim = feature_dim * 2 + child_dim + 3
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.offset_head = nn.Linear(feature_dim, 3)

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_features: torch.Tensor,
        global_features: torch.Tensor,
        target_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_count = parent_xyz.shape[1]
        ratio = math.ceil(int(target_count) / parent_count)
        if ratio > self.max_children:
            raise ValueError(f"Expansion ratio {ratio} exceeds {self.max_children}")
        child_ids = torch.arange(ratio, device=parent_xyz.device)
        child_embedding = self.child_embedding(child_ids)
        child_embedding = child_embedding.view(1, 1, ratio, -1).expand(
            parent_xyz.shape[0], parent_count, -1, -1
        )
        parent_xyz_expanded = parent_xyz.unsqueeze(2).expand(-1, -1, ratio, -1)
        parent_features_expanded = parent_features.unsqueeze(2).expand(
            -1, -1, ratio, -1
        )
        global_expanded = global_features.unsqueeze(2).expand(
            -1, parent_count, ratio, -1
        )
        child_features = self.feature_mlp(
            torch.cat(
                [
                    parent_features_expanded,
                    global_expanded,
                    child_embedding,
                    parent_xyz_expanded,
                ],
                dim=-1,
            )
        )
        child_offsets = torch.tanh(
            self.offset_head(child_features)
        ) * self.max_offset_ratio
        child_xyz = parent_xyz_expanded + child_offsets
        child_xyz = child_xyz.flatten(1, 2)[:, :target_count]
        child_features = child_features.flatten(1, 2)[:, :target_count]
        return child_xyz, child_features


class CoarseToFineExpansion(nn.Module):
    def __init__(self, feature_dim: int, mid_points: int, num_output_points: int):
        super().__init__()
        self.mid_points = int(mid_points)
        self.num_output_points = int(num_output_points)
        self.partial_seed_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.mid_expansion = PointExpansionBlock(
            feature_dim, max_children=8, max_offset_ratio=0.25
        )
        self.fine_expansion = PointExpansionBlock(
            feature_dim, max_children=8, max_offset_ratio=0.10
        )

    def forward(
        self, inputs: CoarseToFineExpansionInput
    ) -> CoarseToFineExpansionOutput:
        partial_seed_features = self.partial_seed_projection(
            inputs.partial_features + inputs.global_features
        )
        seed_xyz = torch.cat([inputs.coarse, inputs.partial_context], dim=1)
        seed_features = torch.cat(
            [inputs.coarse_features, partial_seed_features], dim=1
        )
        mid, mid_features = self.mid_expansion(
            seed_xyz,
            seed_features,
            inputs.global_features,
            self.mid_points,
        )
        fine, _ = self.fine_expansion(
            mid,
            mid_features,
            inputs.global_features,
            self.num_output_points,
        )
        return CoarseToFineExpansionOutput(mid, fine)
