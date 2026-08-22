"""Source-conditioned adaptive reconstruction and denoising queries."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .point_ops import _batch_gather, fps_points


class MemoryInitializationInput(NamedTuple):
    source_context: torch.Tensor  # (B, Cs, 3)
    partial_context: torch.Tensor  # (B, Cp, 3)
    source_features: torch.Tensor  # (B, Cs, D)
    partial_features: torch.Tensor  # (B, Cp, D)


class AdaptiveQueryGeneratorInput(NamedTuple):
    source_context: torch.Tensor  # (B, Cs, 3)
    partial_context: torch.Tensor  # (B, Cp, 3)
    source_features: torch.Tensor  # (B, Cs, D), encoded memory slice
    partial_features: torch.Tensor  # (B, Cp, D), encoded memory slice
    global_features: torch.Tensor  # (B, 1, D)


class AdaptiveQueryGeneratorOutput(NamedTuple):
    query_xyz: torch.Tensor  # (B, Q [+ Qd in training], 3)
    query_features: torch.Tensor  # (B, Q [+ Qd in training], D)
    ranking_scores: torch.Tensor  # (B, 2 * A + P)
    regular_query_count: int
    denoise_target: torch.Tensor | None  # (B, Qd, 3) in training
    attention_mask: torch.Tensor | None  # (Q + Qd, Q + Qd)


class AdaptiveQueryGenerator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        coarse_points: int,
        denoise_queries: int,
        denoise_jitter: float,
    ):
        super().__init__()
        self.coarse_points = int(coarse_points)
        self.anchor_candidates = max(2, self.coarse_points // 2)
        self.anchor_queries = max(1, self.coarse_points // 4)
        self.predicted_candidates = self.coarse_points
        self.denoise_queries = max(int(denoise_queries), 0)
        self.denoise_jitter = float(denoise_jitter)

        self.position_embedding = nn.Sequential(
            nn.Linear(3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.source_type = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.partial_type = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.predicted_type = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.denoise_type = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.denoise_geometry_projection = nn.Sequential(
            nn.Linear(7, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.predicted_candidate_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, self.predicted_candidates * 3),
        )
        self.predicted_feature_head = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.query_ranker = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def initialize_memory(self, inputs: MemoryInitializationInput) -> torch.Tensor:
        memory_positions = torch.cat(
            [inputs.source_context, inputs.partial_context], dim=1
        )
        return torch.cat(
            [
                inputs.source_features + self.source_type,
                inputs.partial_features + self.partial_type,
            ],
            dim=1,
        ) + self.position_embedding(memory_positions)

    def _adaptive_queries(
        self, inputs: AdaptiveQueryGeneratorInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_xyz, source_anchor_features = fps_points(
            inputs.source_context, self.anchor_candidates, inputs.source_features
        )
        partial_xyz, partial_anchor_features = fps_points(
            inputs.partial_context, self.anchor_candidates, inputs.partial_features
        )
        global_vector = inputs.global_features.squeeze(1)
        predicted_xyz = torch.tanh(
            self.predicted_candidate_head(global_vector).reshape(
                inputs.source_context.shape[0], self.predicted_candidates, 3
            )
        ) * 1.25
        predicted_features = self.predicted_feature_head(
            torch.cat(
                [
                    inputs.global_features.expand(
                        -1, self.predicted_candidates, -1
                    ),
                    predicted_xyz,
                ],
                dim=-1,
            )
        )
        candidate_xyz = torch.cat([source_xyz, partial_xyz, predicted_xyz], dim=1)
        candidate_features = torch.cat(
            [
                source_anchor_features + self.source_type,
                partial_anchor_features + self.partial_type,
                predicted_features + self.predicted_type,
            ],
            dim=1,
        )
        ranking_scores = self.query_ranker(
            torch.cat([candidate_features, candidate_xyz], dim=-1)
        ).squeeze(-1)
        source_count = source_xyz.shape[1]
        partial_count = partial_xyz.shape[1]
        source_keep = min(self.anchor_queries, source_count)
        partial_keep = min(self.anchor_queries, partial_count)
        predicted_keep = self.coarse_points - source_keep - partial_keep
        source_indices = ranking_scores[:, :source_count].topk(
            source_keep, dim=1, largest=True
        ).indices
        partial_indices = ranking_scores[
            :, source_count : source_count + partial_count
        ].topk(partial_keep, dim=1, largest=True).indices + source_count
        predicted_indices = ranking_scores[
            :, source_count + partial_count :
        ].topk(predicted_keep, dim=1, largest=True).indices
        predicted_indices = predicted_indices + source_count + partial_count
        selected_indices = torch.cat(
            [source_indices, partial_indices, predicted_indices], dim=1
        )
        selected_xyz = _batch_gather(candidate_xyz, selected_indices)
        selected_features = _batch_gather(candidate_features, selected_indices)
        selected_scores = _batch_gather(
            ranking_scores.unsqueeze(-1), selected_indices
        )
        selected_features = selected_features * (1.0 + selected_scores)
        selected_features = (
            selected_features
            + self.position_embedding(selected_xyz)
            + inputs.global_features
        )
        return selected_xyz, selected_features, ranking_scores

    def forward(
        self, inputs: AdaptiveQueryGeneratorInput
    ) -> AdaptiveQueryGeneratorOutput:
        query_xyz, query_features, ranking_scores = self._adaptive_queries(inputs)
        regular_query_count = query_xyz.shape[1]
        denoise_target = None
        if self.training and self.denoise_queries > 0:
            denoise_target, denoise_features = fps_points(
                inputs.partial_context,
                self.denoise_queries,
                inputs.partial_features,
            )
            denoise_xyz = denoise_target + torch.randn_like(
                denoise_target
            ) * self.denoise_jitter
            with torch.no_grad(), torch.autocast(
                device_type=denoise_xyz.device.type, enabled=False
            ):
                nearest_indices = torch.cdist(
                    denoise_xyz.float(), inputs.partial_context.float()
                ).argmin(dim=-1)
            nearest_partial = _batch_gather(inputs.partial_context, nearest_indices)
            denoise_target = nearest_partial
            nearest_offset = nearest_partial - denoise_xyz
            nearest_distance = torch.linalg.vector_norm(
                nearest_offset.float(), dim=-1, keepdim=True
            ).to(denoise_xyz.dtype)
            denoise_geometry = self.denoise_geometry_projection(
                torch.cat(
                    [denoise_xyz, nearest_offset, nearest_distance], dim=-1
                )
            )
            denoise_features = (
                denoise_features
                + self.position_embedding(denoise_xyz)
                + denoise_geometry
                + inputs.global_features
                + self.denoise_type
            )
            query_xyz = torch.cat([query_xyz, denoise_xyz], dim=1)
            query_features = torch.cat([query_features, denoise_features], dim=1)
            attention_mask = torch.zeros(
                query_xyz.shape[1],
                query_xyz.shape[1],
                dtype=torch.bool,
                device=query_xyz.device,
            )
            attention_mask[:regular_query_count, regular_query_count:] = True
        else:
            attention_mask = None
        return AdaptiveQueryGeneratorOutput(
            query_xyz,
            query_features,
            ranking_scores,
            regular_query_count,
            denoise_target,
            attention_mask,
        )
