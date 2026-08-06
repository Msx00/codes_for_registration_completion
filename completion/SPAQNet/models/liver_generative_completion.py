"""AdaPoinTr-style source-conditioned liver point-cloud completion.

The source is used as a complete-shape feature prior and as adaptive query
anchors. The network still generates absolute completed coordinates; it does
not predict a source displacement field.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .liver_completion import LocalGraphEncoder, knn_indices


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    view_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    return values[batch.view(view_shape), indices]


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
            torch.cat(
                [neighbors - centers, centers, relative_positions], dim=-1
            )
        ).amax(dim=2)
        features = features + self.merge(
            torch.cat([global_features, local_features], dim=-1)
        )
        return features + self.ffn(self.ffn_norm(features))


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
            torch.cat(
                [neighbors - centers, centers, relative_positions], dim=-1
            )
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
            raise ValueError(
                f"Expansion ratio {ratio} exceeds {self.max_children}"
            )
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


class LiverGenerativeCompletionSPAQNet(nn.Module):
    """Generate a complete point set with source-conditioned adaptive queries."""

    def __init__(
        self,
        feature_dim: int = 192,
        num_heads: int = 6,
        k_neighbors: int = 12,
        context_points: int = 256,
        num_output_points: int = 2048,
        coarse_points: int = 256,
        encoder_depth: int = 3,
        decoder_depth: int = 4,
        denoise_queries: int = 64,
        denoise_jitter: float = 0.005,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.context_points = int(context_points)
        self.num_output_points = int(num_output_points)
        self.coarse_points = min(
            int(coarse_points),
            max(2, self.num_output_points // 8),
            self.num_output_points,
        )
        self.mid_points = min(
            1024,
            max(self.coarse_points, self.num_output_points // 2),
            self.num_output_points,
        )
        self.anchor_candidates = max(2, self.coarse_points // 2)
        self.anchor_queries = max(1, self.coarse_points // 4)
        self.predicted_candidates = self.coarse_points
        self.denoise_queries = max(int(denoise_queries), 0)
        self.denoise_jitter = float(denoise_jitter)

        self.source_encoder = LocalGraphEncoder(feature_dim, k_neighbors)
        self.partial_encoder = LocalGraphEncoder(feature_dim, k_neighbors)
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

    def _select_partial_context(
        self,
        partial: torch.Tensor,
        partial_mask: torch.Tensor,
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

    @staticmethod
    def normalize_by_source(
        source: torch.Tensor,
        moving: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid = source.mean(dim=1, keepdim=True)
        source_centered = source - centroid
        scale = torch.linalg.vector_norm(
            source_centered.float(), dim=-1
        ).amax(dim=1, keepdim=True).clamp_min(eps).unsqueeze(-1)
        scale = scale.to(source.dtype)
        return (
            source_centered / scale,
            (moving - centroid) / scale,
            centroid,
            scale,
        )

    def _adaptive_queries(
        self,
        source_context: torch.Tensor,
        partial_context: torch.Tensor,
        source_features: torch.Tensor,
        partial_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_xyz, source_anchor_features = fps_points(
            source_context, self.anchor_candidates, source_features
        )
        partial_xyz, partial_anchor_features = fps_points(
            partial_context, self.anchor_candidates, partial_features
        )
        global_vector = global_features.squeeze(1)
        predicted_xyz = torch.tanh(
            self.predicted_candidate_head(global_vector).reshape(
                source_context.shape[0], self.predicted_candidates, 3
            )
        ) * 1.25
        predicted_features = self.predicted_feature_head(
            torch.cat(
                [
                    global_features.expand(-1, self.predicted_candidates, -1),
                    predicted_xyz,
                ],
                dim=-1,
            )
        )
        candidate_xyz = torch.cat(
            [source_xyz, partial_xyz, predicted_xyz], dim=1
        )
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
        # Fixed group quotas prevent hard top-k from discarding the complete
        # source prior while ranking remains active inside every candidate set.
        selected_indices = torch.cat(
            [source_indices, partial_indices, predicted_indices], dim=1
        )
        selected_xyz = _batch_gather(candidate_xyz, selected_indices)
        selected_features = _batch_gather(candidate_features, selected_indices)
        selected_scores = _batch_gather(
            ranking_scores.unsqueeze(-1), selected_indices
        )
        # The gate gives the ranking head a differentiable learning signal,
        # while top-k performs AdaPoinTr-style candidate selection.
        selected_features = selected_features * (1.0 + selected_scores)
        selected_features = (
            selected_features
            + self.position_embedding(selected_xyz)
            + global_features
        )
        return selected_xyz, selected_features, ranking_scores

    def forward(
        self,
        source: torch.Tensor,
        partial: torch.Tensor,
        partial_mask: torch.Tensor,
        partial_dense: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del partial_dense, observed_mask
        source_norm, partial_norm, centroid, scale = self.normalize_by_source(
            source, partial
        )
        source_context, _ = fps_points(source_norm, self.context_points)
        partial_context = self._select_partial_context(partial_norm, partial_mask)
        source_features, _ = self.source_encoder(source_context)
        partial_features, _ = self.partial_encoder(partial_context)
        memory_positions = torch.cat([source_context, partial_context], dim=1)
        memory = torch.cat(
            [
                source_features + self.source_type,
                partial_features + self.partial_type,
            ],
            dim=1,
        ) + self.position_embedding(memory_positions)
        memory_neighbors = knn_indices(memory_positions, self.memory_encoder[0].k_neighbors)
        for block in self.memory_encoder:
            memory = block(memory, memory_positions, memory_neighbors)
        source_count = source_context.shape[1]
        source_features = memory[:, :source_count]
        partial_features = memory[:, source_count:]
        source_global = source_features.amax(dim=1)
        partial_global = partial_features.amax(dim=1)
        global_features = self.global_fusion(
            torch.cat([source_global, partial_global], dim=-1)
        ).unsqueeze(1)

        query_xyz, query_features, ranking_scores = self._adaptive_queries(
            source_context,
            partial_context,
            source_features,
            partial_features,
            global_features,
        )
        regular_query_count = query_xyz.shape[1]
        denoise_target = None
        if self.training and self.denoise_queries > 0:
            denoise_target, denoise_features = fps_points(
                partial_context,
                self.denoise_queries,
                partial_features,
            )
            denoise_xyz = denoise_target + torch.randn_like(
                denoise_target
            ) * self.denoise_jitter
            with torch.no_grad(), torch.autocast(
                device_type=denoise_xyz.device.type, enabled=False
            ):
                nearest_indices = torch.cdist(
                    denoise_xyz.float(), partial_context.float()
                ).argmin(dim=-1)
            nearest_partial = _batch_gather(
                partial_context, nearest_indices
            )
            # As in AdaPoinTr's neighborhood denoising target, supervise the
            # geometrically nearest clean surface point rather than the
            # originally jittered index, which can become ambiguous on a
            # densely sampled surface after perturbation.
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
                + global_features
                + self.denoise_type
            )
            query_xyz = torch.cat([query_xyz, denoise_xyz], dim=1)
            query_features = torch.cat(
                [query_features, denoise_features], dim=1
            )
            attention_mask = torch.zeros(
                query_xyz.shape[1],
                query_xyz.shape[1],
                dtype=torch.bool,
                device=query_xyz.device,
            )
            attention_mask[:regular_query_count, regular_query_count:] = True
        else:
            attention_mask = None
        query_k = min(
            self.query_decoder[0].k_neighbors,
            max(regular_query_count - 1, 1),
        )
        query_neighbors = knn_indices(query_xyz, query_k)
        if denoise_target is not None:
            # Regular reconstruction queries must not consume training-only
            # denoising tokens through the local kNN branch either.
            regular_neighbors = knn_indices(
                query_xyz[:, :regular_query_count],
                query_k,
            )
            query_neighbors = torch.cat(
                [regular_neighbors, query_neighbors[:, regular_query_count:]],
                dim=1,
            )
        for block in self.query_decoder:
            query_features = block(
                query_features,
                query_xyz,
                memory,
                query_neighbors,
                attention_mask,
            )
        coarse = query_xyz[:, :regular_query_count] + torch.tanh(
            self.coarse_refine_head(
                query_features[:, :regular_query_count]
            )
        ) * 0.10
        coarse_features = query_features[:, :regular_query_count]
        denoised = (
            query_xyz[:, regular_query_count:]
            + torch.tanh(
                self.denoise_refine_head(
                    query_features[:, regular_query_count:]
                )
            )
            * 0.10
            if denoise_target is not None
            else None
        )

        partial_seed_features = self.partial_seed_projection(
            partial_features + global_features
        )
        seed_xyz = torch.cat([coarse, partial_context], dim=1)
        seed_features = torch.cat(
            [coarse_features, partial_seed_features], dim=1
        )
        mid, mid_features = self.mid_expansion(
            seed_xyz, seed_features, global_features, self.mid_points
        )
        fine, _ = self.fine_expansion(
            mid, mid_features, global_features, self.num_output_points
        )
        return {
            "coarse_normalized": coarse,
            "mid_normalized": mid,
            "fine_normalized": fine,
            "completed_xyz": fine * scale + centroid,
            "partial_normalized": partial_norm,
            "denoised_normalized": denoised,
            "denoise_target_normalized": denoise_target,
            "ranking_scores": ranking_scores,
            "centroid": centroid,
            "scale": scale,
        }
