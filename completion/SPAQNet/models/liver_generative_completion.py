"""Source-normalized orchestration for generative liver completion."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .adaptive_query_generator import (
    AdaptiveQueryGenerator,
    AdaptiveQueryGeneratorInput,
    MemoryInitializationInput,
)
from .coarse_to_fine_expansion import (
    CoarseToFineExpansion,
    CoarseToFineExpansionInput,
)
from .geometric_memory_encoder import (
    GeometricMemoryEncoder,
    GeometricMemoryEncoderInput,
)
from .geometric_query_decoder import (
    GeometricQueryDecoder,
    GeometricQueryDecoderInput,
)
from .local_graph_encoding import LocalGraphEncoding, LocalGraphEncodingInput


class LiverGenerativeCompletionSPAQNet(nn.Module):
    """Generate a complete point set with source-conditioned adaptive queries."""

    LEGACY_STATE_DICT_PREFIXES = {
        "source_encoder.": "local_graph_encoding.source_encoder.",
        "partial_encoder.": "local_graph_encoding.partial_encoder.",
        "position_embedding.": "adaptive_query_generator.position_embedding.",
        "source_type": "adaptive_query_generator.source_type",
        "partial_type": "adaptive_query_generator.partial_type",
        "predicted_type": "adaptive_query_generator.predicted_type",
        "denoise_type": "adaptive_query_generator.denoise_type",
        "denoise_geometry_projection.": (
            "adaptive_query_generator.denoise_geometry_projection."
        ),
        "memory_encoder.": "geometric_memory_encoder.memory_encoder.",
        "global_fusion.": "geometric_memory_encoder.global_fusion.",
        "predicted_candidate_head.": (
            "adaptive_query_generator.predicted_candidate_head."
        ),
        "predicted_feature_head.": (
            "adaptive_query_generator.predicted_feature_head."
        ),
        "query_ranker.": "adaptive_query_generator.query_ranker.",
        "query_decoder.": "geometric_query_decoder.query_decoder.",
        "coarse_refine_head.": "geometric_query_decoder.coarse_refine_head.",
        "denoise_refine_head.": "geometric_query_decoder.denoise_refine_head.",
        "partial_seed_projection.": (
            "coarse_to_fine_expansion.partial_seed_projection."
        ),
        "mid_expansion.": "coarse_to_fine_expansion.mid_expansion.",
        "fine_expansion.": "coarse_to_fine_expansion.fine_expansion.",
    }

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

        self.local_graph_encoding = LocalGraphEncoding(
            feature_dim, k_neighbors, self.context_points
        )
        self.adaptive_query_generator = AdaptiveQueryGenerator(
            feature_dim,
            self.coarse_points,
            self.denoise_queries,
            self.denoise_jitter,
        )
        self.geometric_memory_encoder = GeometricMemoryEncoder(
            feature_dim, num_heads, k_neighbors, encoder_depth
        )
        self.geometric_query_decoder = GeometricQueryDecoder(
            feature_dim, num_heads, k_neighbors, decoder_depth
        )
        self.coarse_to_fine_expansion = CoarseToFineExpansion(
            feature_dim, self.mid_points, self.num_output_points
        )

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

    @classmethod
    def _migrate_legacy_keys(
        cls, state_dict: Mapping[str, torch.Tensor], prefix: str
    ) -> None:
        # PyTorch passes a mutable OrderedDict here. Mutating only exact known
        # legacy prefixes preserves strict=True diagnostics for every other key.
        for old_prefix, new_prefix in cls.LEGACY_STATE_DICT_PREFIXES.items():
            old_full_prefix = prefix + old_prefix
            new_full_prefix = prefix + new_prefix
            for key in list(state_dict.keys()):
                if key == old_full_prefix or key.startswith(old_full_prefix):
                    suffix = key[len(old_full_prefix) :]
                    new_key = new_full_prefix + suffix
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._migrate_legacy_keys(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        local = self.local_graph_encoding(
            LocalGraphEncodingInput(source_norm, partial_norm, partial_mask)
        )
        initialized_memory = self.adaptive_query_generator.initialize_memory(
            MemoryInitializationInput(
                local.source_context,
                local.partial_context,
                local.source_features,
                local.partial_features,
            )
        )
        encoded = self.geometric_memory_encoder(
            GeometricMemoryEncoderInput(
                local.source_context, local.partial_context, initialized_memory
            )
        )
        queries = self.adaptive_query_generator(
            AdaptiveQueryGeneratorInput(
                local.source_context,
                local.partial_context,
                encoded.source_features,
                encoded.partial_features,
                encoded.global_features,
            )
        )
        decoded = self.geometric_query_decoder(
            GeometricQueryDecoderInput(
                queries.query_xyz,
                queries.query_features,
                encoded.memory,
                queries.regular_query_count,
                queries.denoise_target,
                queries.attention_mask,
            )
        )
        expanded = self.coarse_to_fine_expansion(
            CoarseToFineExpansionInput(
                decoded.coarse,
                decoded.coarse_features,
                local.partial_context,
                encoded.partial_features,
                encoded.global_features,
            )
        )
        return {
            "coarse_normalized": decoded.coarse,
            "mid_normalized": expanded.mid,
            "fine_normalized": expanded.fine,
            "completed_xyz": expanded.fine * scale + centroid,
            "partial_normalized": partial_norm,
            "denoised_normalized": decoded.denoised,
            "denoise_target_normalized": queries.denoise_target,
            "ranking_scores": queries.ranking_scores,
            "centroid": centroid,
            "scale": scale,
        }
