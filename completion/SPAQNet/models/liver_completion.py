"""Source-conditioned SPAQNet-style liver point-cloud completion network.

This branch is intentionally independent from the original image-conditioned
SPAQNet entry point.  It preserves source/GT point correspondence and predicts
a coarse displacement followed by two residual refinement stages.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather values shaped (B, N, C) with indices shaped (B, ...)."""
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
        neighbor_features = _batch_gather(
            point_features,
            neighbor_indices,
        )
        neighbor_points = _batch_gather(points, neighbor_indices)
        center_features = point_features.unsqueeze(2).expand_as(
            neighbor_features
        )
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


class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        output = queries + attended
        return output + self.mlp(self.output_norm(output))


class ResidualRefinementBlock(nn.Module):
    """Refine a dense deformation using partial and graph-local context."""

    def __init__(
        self,
        feature_dim: int,
        max_step_ratio: float,
    ):
        super().__init__()
        self.max_step_ratio = float(max_step_ratio)
        input_dim = feature_dim * 2 + 13
        self.refiner = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 3),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    @staticmethod
    def _nearest_partial_features(
        current: torch.Tensor,
        partial_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), torch.autocast(
            device_type=current.device.type,
            enabled=False,
        ):
            nearest_indices = torch.cdist(
                current.float(),
                partial_context.float(),
            ).argmin(dim=-1)
        nearest = _batch_gather(partial_context, nearest_indices)
        offset = nearest - current
        distance = torch.linalg.vector_norm(offset.float(), dim=-1).to(
            current.dtype
        )
        return offset, distance.unsqueeze(-1)

    def forward(
        self,
        current: torch.Tensor,
        source: torch.Tensor,
        fused_features: torch.Tensor,
        global_features: torch.Tensor,
        partial_context: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        displacement = current - source
        neighbor_displacement = _batch_gather(
            displacement,
            neighbor_indices,
        ).mean(dim=2)
        nearest_offset, nearest_distance = self._nearest_partial_features(
            current,
            partial_context,
        )
        features = torch.cat(
            [
                fused_features,
                global_features.expand(-1, current.shape[1], -1),
                current,
                displacement,
                neighbor_displacement,
                nearest_offset,
                nearest_distance,
            ],
            dim=-1,
        )
        residual = torch.tanh(self.refiner(features)) * self.max_step_ratio
        return current + residual


class LiverCompletionSPAQNet(nn.Module):
    """Predict a correspondence-preserving completed liver point cloud."""

    def __init__(
        self,
        feature_dim: int = 192,
        num_heads: int = 6,
        k_neighbors: int = 12,
        context_points: int = 256,
        max_coarse_ratio: float = 0.75,
        max_mid_ratio: float = 0.25,
        max_fine_ratio: float = 0.10,
        aligned_observed: bool = True,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.context_points = int(context_points)
        self.max_coarse_ratio = float(max_coarse_ratio)
        self.aligned_observed = bool(aligned_observed)

        self.source_encoder = LocalGraphEncoder(feature_dim, k_neighbors)
        self.partial_encoder = LocalGraphEncoder(feature_dim, k_neighbors)
        self.cross_attention = CrossAttentionBlock(feature_dim, num_heads)
        self.global_fusion = nn.Sequential(
            nn.Linear(
                feature_dim * (3 if self.aligned_observed else 2),
                feature_dim,
            ),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        if self.aligned_observed:
            self.observed_encoder = nn.Sequential(
                nn.Linear(7, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                feature_dim * (4 if self.aligned_observed else 3),
                feature_dim,
            ),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.coarse_head = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 3),
        )
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)

        self.mid_refiner = ResidualRefinementBlock(
            feature_dim,
            max_mid_ratio,
        )
        self.fine_refiner = ResidualRefinementBlock(
            feature_dim,
            max_fine_ratio,
        )

    def _select_partial_context(
        self,
        partial: torch.Tensor,
        partial_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_counts = partial_mask.sum(dim=1)
        if bool((valid_counts < 2).any()):
            raise ValueError("Every partial cloud must contain at least 2 points")
        context_count = min(
            self.context_points,
            int(valid_counts.min().item()),
        )
        selected = []
        for batch_index in range(partial.shape[0]):
            valid_indices = torch.nonzero(
                partial_mask[batch_index],
                as_tuple=False,
            ).squeeze(1)
            positions = torch.linspace(
                0,
                valid_indices.numel() - 1,
                steps=context_count,
                device=partial.device,
            ).round().long()
            selected.append(partial[batch_index, valid_indices[positions]])
        return torch.stack(selected, dim=0)

    @staticmethod
    def normalize_by_source(
        source: torch.Tensor,
        moving: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid = source.mean(dim=1, keepdim=True)
        source_centered = source - centroid
        scale = torch.linalg.vector_norm(
            source_centered.float(),
            dim=-1,
        ).amax(dim=1, keepdim=True).clamp_min(eps).unsqueeze(-1)
        scale = scale.to(source.dtype)
        return (
            source_centered / scale,
            (moving - centroid) / scale,
            centroid,
            scale,
        )

    def forward(
        self,
        source: torch.Tensor,
        partial: torch.Tensor,
        partial_mask: torch.Tensor,
        partial_dense: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_norm, partial_norm, centroid, scale = self.normalize_by_source(
            source,
            partial,
        )
        if self.aligned_observed:
            if partial_dense is None or observed_mask is None:
                raise ValueError(
                    "Aligned-observed mode requires partial_dense and "
                    "observed_mask"
                )
            partial_dense_norm = (partial_dense - centroid) / scale
            observed = observed_mask.unsqueeze(-1)
            # Missing dense entries are placeholders. Replacing them by source
            # points makes their offset zero before the mask removes the feature.
            aligned_partial_norm = torch.where(
                observed,
                partial_dense_norm,
                source_norm,
            )
            observed_features = self.observed_encoder(
                torch.cat(
                    [
                        aligned_partial_norm,
                        aligned_partial_norm - source_norm,
                        observed.to(source_norm.dtype),
                    ],
                    dim=-1,
                )
            ) * observed.to(source_norm.dtype)
        partial_context = self._select_partial_context(
            partial_norm,
            partial_mask,
        )

        source_features, neighbor_indices = self.source_encoder(source_norm)
        partial_features, _ = self.partial_encoder(partial_context)
        attended_source = self.cross_attention(
            source_features,
            partial_features,
        )

        source_global = source_features.amax(dim=1)
        partial_global = partial_features.amax(dim=1)
        global_inputs = [source_global, partial_global]
        if self.aligned_observed:
            observed_global = observed_features.sum(dim=1) / observed.to(
                source_norm.dtype
            ).sum(dim=1).clamp_min(1.0)
            global_inputs.append(observed_global)
        global_features = self.global_fusion(
            torch.cat(global_inputs, dim=-1)
        ).unsqueeze(1)

        fusion_inputs = [
            source_features,
            attended_source,
            global_features.expand(-1, source.shape[1], -1),
        ]
        if self.aligned_observed:
            neighbor_observed = _batch_gather(
                observed.to(source_norm.dtype),
                neighbor_indices,
            )
            neighbor_observed_features = _batch_gather(
                observed_features,
                neighbor_indices,
            )
            local_observed_features = (
                neighbor_observed_features * neighbor_observed
            ).sum(dim=2) / neighbor_observed.sum(dim=2).clamp_min(1.0)
            aligned_observed_features = torch.where(
                observed,
                observed_features,
                local_observed_features,
            )
            fusion_inputs.append(aligned_observed_features)
        fused_features = self.feature_fusion(
            torch.cat(fusion_inputs, dim=-1)
        )

        coarse_displacement = torch.tanh(
            self.coarse_head(
                torch.cat([fused_features, source_norm], dim=-1)
            )
        ) * self.max_coarse_ratio
        coarse = source_norm + coarse_displacement
        if self.aligned_observed:
            coarse = torch.where(observed, aligned_partial_norm, coarse)
        mid = self.mid_refiner(
            coarse,
            source_norm,
            fused_features,
            global_features,
            partial_context,
            neighbor_indices,
        )
        if self.aligned_observed:
            mid = torch.where(observed, aligned_partial_norm, mid)
        fine = self.fine_refiner(
            mid,
            source_norm,
            fused_features,
            global_features,
            partial_context,
            neighbor_indices,
        )
        if self.aligned_observed:
            fine = torch.where(observed, aligned_partial_norm, fine)

        outputs = {
            "coarse_normalized": coarse,
            "mid_normalized": mid,
            "fine_normalized": fine,
            "completed_xyz": fine * scale + centroid,
            "source_normalized": source_norm,
            "partial_normalized": partial_norm,
            "centroid": centroid,
            "scale": scale,
            "neighbor_indices": neighbor_indices,
        }
        if self.aligned_observed:
            outputs["partial_dense_normalized"] = aligned_partial_norm
        return outputs
