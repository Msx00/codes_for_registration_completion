"""Geometry-aware decoding and refinement of adaptive queries."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .point_ops import _batch_gather, knn_indices


class GeometricQueryDecoderInput(NamedTuple):
    query_xyz: torch.Tensor  # (B, Q [+ Qd], 3)
    query_features: torch.Tensor  # (B, Q [+ Qd], D)
    memory: torch.Tensor  # (B, M, D)
    regular_query_count: int
    denoise_target: torch.Tensor | None  # (B, Qd, 3)
    attention_mask: torch.Tensor | None  # (Q + Qd, Q + Qd)


class GeometricQueryDecoderOutput(NamedTuple):
    coarse: torch.Tensor  # (B, Q, 3)
    coarse_features: torch.Tensor  # (B, Q, D)
    denoised: torch.Tensor | None  # (B, Qd, 3)


class GeometryDecoderBlock(nn.Module):
    """Geometry-aware query self-attention followed by memory cross-attention."""

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
        self.self_merge = nn.Linear(feature_dim * 2, feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.memory_norm = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        query_positions: torch.Tensor,
        memory: torch.Tensor,
        query_neighbors: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        global_features, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        local_input = self.local_norm(queries)
        neighbors = _batch_gather(local_input, query_neighbors)
        neighbor_positions = _batch_gather(query_positions, query_neighbors)
        centers = local_input.unsqueeze(2).expand_as(neighbors)
        relative_positions = neighbor_positions - query_positions.unsqueeze(2)
        local_features = self.local_mlp(
            torch.cat([neighbors - centers, centers, relative_positions], dim=-1)
        ).amax(dim=2)
        queries = queries + self.self_merge(
            torch.cat([global_features, local_features], dim=-1)
        )
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class GeometricQueryDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        k_neighbors: int,
        decoder_depth: int,
    ):
        super().__init__()
        self.query_decoder = nn.ModuleList(
            [
                GeometryDecoderBlock(feature_dim, num_heads, k_neighbors)
                for _ in range(max(int(decoder_depth), 1))
            ]
        )
        self.coarse_refine_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        nn.init.zeros_(self.coarse_refine_head[-1].weight)
        nn.init.zeros_(self.coarse_refine_head[-1].bias)
        self.denoise_refine_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        nn.init.zeros_(self.denoise_refine_head[-1].weight)
        nn.init.zeros_(self.denoise_refine_head[-1].bias)

    def forward(
        self, inputs: GeometricQueryDecoderInput
    ) -> GeometricQueryDecoderOutput:
        query_k = min(
            self.query_decoder[0].k_neighbors,
            max(inputs.regular_query_count - 1, 1),
        )
        query_neighbors = knn_indices(inputs.query_xyz, query_k)
        if inputs.denoise_target is not None:
            regular_neighbors = knn_indices(
                inputs.query_xyz[:, : inputs.regular_query_count], query_k
            )
            query_neighbors = torch.cat(
                [
                    regular_neighbors,
                    query_neighbors[:, inputs.regular_query_count :],
                ],
                dim=1,
            )
        query_features = inputs.query_features
        for block in self.query_decoder:
            query_features = block(
                query_features,
                inputs.query_xyz,
                inputs.memory,
                query_neighbors,
                inputs.attention_mask,
            )
        coarse = inputs.query_xyz[:, : inputs.regular_query_count] + torch.tanh(
            self.coarse_refine_head(
                query_features[:, : inputs.regular_query_count]
            )
        ) * 0.10
        coarse_features = query_features[:, : inputs.regular_query_count]
        denoised = (
            inputs.query_xyz[:, inputs.regular_query_count :]
            + torch.tanh(
                self.denoise_refine_head(
                    query_features[:, inputs.regular_query_count :]
                )
            )
            * 0.10
            if inputs.denoise_target is not None
            else None
        )
        return GeometricQueryDecoderOutput(coarse, coarse_features, denoised)
