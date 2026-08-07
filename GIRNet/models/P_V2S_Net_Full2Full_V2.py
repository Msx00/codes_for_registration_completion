import torch
import torch.nn as nn

from models.T_layer_dgcnn import LayerDGCNN_v3
from models.farthest_point_sampling import farthest_point_sampling
from models.global_matcher import GlobalMatcherV2, interpolate_flow_v2
from models.iterative_refiner import IterativeFlowRefiner
from models.k_nearest_neighbors import k_nearest_neighbors
from models.select import select_point_regions, select_points
from models.symmetric_preprocessor import SymmetricFullToFullPreprocessor


def _farthest_point_sampling_with_indices(
    coords_cf,
    num_samples,
    original_indices=None,
):
    """FPS that also tracks original source indices through pyramid levels.

    Args:
        coords_cf: (B, 3, N_current) coordinates in channel-first format.
        num_samples: number of points to sample.
        original_indices: (B, N_current) long tensor of original source
            indices, or None to start with arange(N).

    Returns:
        sampled_coords_cf: (B, 3, num_samples)
        fps_indices: (B, num_samples) indices into the CURRENT level.
        sampled_original_indices: (B, num_samples) tracked original indices.
    """
    B, C, N = coords_cf.shape
    if original_indices is None:
        original_indices = torch.arange(N, device=coords_cf.device).long().expand(B, N)
    fps_indices, sampled_coords_cf = farthest_point_sampling(
        coords_cf, num_samples, random=False
    )
    batch_idx = torch.arange(B, device=coords_cf.device).view(-1, 1)
    sampled_original_indices = original_indices[batch_idx, fps_indices]
    return sampled_coords_cf, fps_indices, sampled_original_indices


class PV2SNetFull2FullV2(nn.Module):
    """Full2Full V2 with score-normalized GlobalMatcher, soft gate, and
    index-tracked coarse matching.

    Key differences from V1:
    - GlobalMatcherV2 with z-score normalization and learnable score fusion.
    - Soft confidence gate (sigmoid with temperature).
    - FPS tracks original source indices through all pyramid levels.
    - Coarse flow interpolated with squared-distance weights.
    - Returns source_global_indices, target_global_xyz, score_weights
      for L_match supervision.
    """

    PYRAMID_POINT_COUNTS = {4: 239, 3: 144, 2: 92, 1: 35, 0: 8}

    def __init__(
        self,
        feature_dim=50,
        points_per_region=35,
        global_match_level=2,
        global_match_temperature=0.1,
        global_match_dim=64,
        global_spatial_sigma=0.3,
        max_coarse_flow_normalized=0.25,
        gate_temperature=0.02,
        num_refinement_steps=3,
        refinement_k=35,
        enc_freq=(2e-2, 2e-1, 2, 4, 8, 16, 32, 64),
        enc_freq_scale=1.0,
        debug_refinement=False,
    ):
        super().__init__()
        if global_match_level not in self.PYRAMID_POINT_COUNTS:
            raise ValueError(
                "global_match_level must be one of 0,1,2,3,4; "
                f"got {global_match_level}"
            )
        self.feature_dim = int(feature_dim)
        self.points_per_region = int(points_per_region)
        self.global_match_level = int(global_match_level)
        self.preprocessor = SymmetricFullToFullPreprocessor(
            enc_freq=enc_freq,
            enc_freq_scale=enc_freq_scale,
        )

        self.reduce_channels = nn.Conv1d(56, self.feature_dim, kernel_size=1)
        self.dgcnn_4 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_3 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_2 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_1 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_0 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)

        self.global_matcher = GlobalMatcherV2(
            feature_dim=self.feature_dim,
            projection_dim=global_match_dim,
            temperature=global_match_temperature,
            spatial_sigma=global_spatial_sigma,
            max_coarse_flow=max_coarse_flow_normalized,
            gate_temperature=gate_temperature,
        )
        self.iterative_refiner = IterativeFlowRefiner(
            feature_dim=self.feature_dim,
            num_steps=num_refinement_steps,
            k=refinement_k,
            debug=debug_refinement,
        )
        print(
            "Building PV2SNetFull2FullV2: "
            f"global match level={self.global_match_level}, "
            f"global match points={self.global_match_points}, "
            f"refinement steps={num_refinement_steps}, k={refinement_k}, "
            f"spatial_sigma={global_spatial_sigma}, "
            f"gate_temperature={gate_temperature}"
        )

    @property
    def global_match_points(self):
        return self.PYRAMID_POINT_COUNTS[self.global_match_level]

    def _downsample_level(self, coords_cf, features_cf, original_indices, point_count, dgcnn):
        """Downsample one pyramid level while tracking original indices."""
        if coords_cf.shape[-1] < point_count:
            raise ValueError(
                f"Input has {coords_cf.shape[-1]} points but pyramid level "
                f"requires {point_count}"
            )
        sampled_coords_cf, fps_indices, sampled_original_indices = (
            _farthest_point_sampling_with_indices(
                coords_cf, point_count, original_indices
            )
        )
        center_features = select_points(features_cf, fps_indices)
        k = min(self.points_per_region, coords_cf.shape[-1])
        _, neighbor_indices = k_nearest_neighbors(
            pos_source=coords_cf,
            pos_queries=sampled_coords_cf,
            k=k,
        )
        neighbor_features = select_point_regions(features_cf, neighbor_indices)
        sampled_features = dgcnn(center_features, neighbor_features)
        return sampled_coords_cf, sampled_features, sampled_original_indices

    def _shared_pyramid(self, coords_cf, features_cf):
        """Build pyramid and track original indices at each level."""
        B, _, N = coords_cf.shape
        original_indices = torch.arange(N, device=coords_cf.device).long().expand(B, N)
        levels = {5: (coords_cf, features_cf, original_indices)}
        current_coords, current_features, current_indices = coords_cf, features_cf, original_indices
        for level in (4, 3, 2, 1, 0):
            dgcnn = getattr(self, f"dgcnn_{level}")
            current_coords, current_features, current_indices = self._downsample_level(
                current_coords,
                current_features,
                current_indices,
                self.PYRAMID_POINT_COUNTS[level],
                dgcnn,
            )
            levels[level] = (current_coords, current_features, current_indices)
        return levels

    def forward(self, source_xyz, target_xyz):
        if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
            raise ValueError("source_xyz must have shape (B, Nsource, 3)")
        if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3:
            raise ValueError("target_xyz must have shape (B, Ntarget, 3)")

        source_input, target_input = self.preprocessor(source_xyz, target_xyz)
        source_features_cf = self.reduce_channels(source_input)
        target_features_cf = self.reduce_channels(target_input)
        source_coords_cf = source_xyz.transpose(1, 2).contiguous()
        target_coords_cf = target_xyz.transpose(1, 2).contiguous()

        source_levels = self._shared_pyramid(source_coords_cf, source_features_cf)
        target_levels = self._shared_pyramid(target_coords_cf, target_features_cf)

        source_global_coords_cf, source_global_features_cf, source_global_indices = (
            source_levels[self.global_match_level]
        )
        target_global_coords_cf, target_global_features_cf, _target_global_indices = (
            target_levels[self.global_match_level]
        )

        # GlobalMatcherV2 expects (B, N, 3) coords and (B, N, C) features.
        match = self.global_matcher(
            source_global_coords_cf.transpose(1, 2).contiguous(),
            target_global_coords_cf.transpose(1, 2).contiguous(),
            source_global_features_cf.transpose(1, 2).contiguous(),
            target_global_features_cf.transpose(1, 2).contiguous(),
        )

        # Interpolate coarse flow to full source resolution using squared-distance weights.
        coarse_flow_full = interpolate_flow_v2(
            query_coords=source_xyz,
            support_coords=source_global_coords_cf.transpose(1, 2).contiguous(),
            support_flow=match["coarse_flow"],
            k=3,
        )

        final_flow, refined_flows, refined_warps = self.iterative_refiner(
            source_xyz,
            target_xyz,
            source_features_cf.transpose(1, 2).contiguous(),
            target_features_cf.transpose(1, 2).contiguous(),
            coarse_flow_full,
        )
        flow_stages = [coarse_flow_full, *refined_flows]
        warped_source_stages = [
            source_xyz + coarse_flow_full,
            *refined_warps,
        ]

        # target_global_xyz is in GIRNet normalized coordinates.
        # The wrapper (GIRNet_text_model.forward) inverse-transforms
        # it back to mm before passing it to compute_match_loss.
        target_global_xyz = target_global_coords_cf.transpose(1, 2).contiguous()

        return {
            "result": final_flow,
            "flow_stages": flow_stages,
            "warped_source_stages": warped_source_stages,
            "global_assignment": match["assignment"],
            "global_match_confidence": match["match_confidence"],
            "source_global_indices": source_global_indices,
            "target_global_xyz": target_global_xyz,
            "score_weights": match["score_weights"],
        }
