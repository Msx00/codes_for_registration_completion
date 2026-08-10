import torch
import torch.nn as nn

from .dgcnn import LayerDGCNN_v3
from .knn import k_nearest_neighbors
from .matcher import GlobalMatcherV3, interpolate_source_features
from .preprocessing import SymmetricFullToFullPreprocessor
from .refiner import IterativeFlowRefinerV3
from .sampling import farthest_point_sampling
from .selection import select_point_regions, select_points


def _farthest_point_sampling_with_indices(
    coords_cf,
    num_samples,
    original_indices=None,
):
    bsz, _, n_points = coords_cf.shape
    if original_indices is None:
        original_indices = torch.arange(
            n_points, device=coords_cf.device
        ).long().expand(bsz, n_points)
    fps_indices, sampled_coords_cf = farthest_point_sampling(
        coords_cf, num_samples, random=False
    )
    batch_idx = torch.arange(
        bsz, device=coords_cf.device
    ).view(-1, 1)
    sampled_original_indices = original_indices[batch_idx, fps_indices]
    return sampled_coords_cf, fps_indices, sampled_original_indices


class PV2SNetFull2FullV3(nn.Module):
    """Source-indexed correspondence deformation network.

    V3 uses global matching to form persistent target context, decodes a
    source-indexed coarse flow, and refines it with both fixed source topology
    and dynamic target evidence.
    """

    PYRAMID_POINT_COUNTS = {4: 239, 3: 144, 2: 92, 1: 35, 0: 8}

    def __init__(
        self,
        feature_dim=50,
        points_per_region=35,
        global_match_level=4,
        global_match_dim=64,
        feature_temperature=1.0,
        spatial_temperature=1.0,
        num_refinement_steps=3,
        refinement_k=35,
        source_graph_k=16,
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

        self.global_matcher = GlobalMatcherV3(
            feature_dim=self.feature_dim,
            projection_dim=global_match_dim,
            feature_temperature=feature_temperature,
            spatial_temperature=spatial_temperature,
        )
        self.iterative_refiner = IterativeFlowRefinerV3(
            feature_dim=self.feature_dim,
            memory_dim=self.global_matcher.memory_dim,
            num_steps=num_refinement_steps,
            target_k=refinement_k,
            source_graph_k=source_graph_k,
            debug=debug_refinement,
        )

        print(
            "Building PV2SNetFull2FullV3: "
            f"global match level={self.global_match_level}, "
            f"global match points={self.global_match_points}, "
            f"feature_temperature={feature_temperature}, "
            f"spatial_temperature={spatial_temperature}, "
            f"refinement steps={num_refinement_steps}, "
            f"target_k={refinement_k}, source_graph_k={source_graph_k}"
        )

    @property
    def global_match_points(self):
        return self.PYRAMID_POINT_COUNTS[self.global_match_level]

    def _downsample_level(
        self,
        coords_cf,
        features_cf,
        original_indices,
        point_count,
        dgcnn,
    ):
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
        neighbor_features = select_point_regions(
            features_cf, neighbor_indices
        )
        sampled_features = dgcnn(center_features, neighbor_features)
        return (
            sampled_coords_cf,
            sampled_features,
            sampled_original_indices,
        )

    def _shared_pyramid(self, coords_cf, features_cf):
        bsz, _, n_points = coords_cf.shape
        original_indices = torch.arange(
            n_points, device=coords_cf.device
        ).long().expand(bsz, n_points)
        levels = {5: (coords_cf, features_cf, original_indices)}
        current_coords = coords_cf
        current_features = features_cf
        current_indices = original_indices
        for level in (4, 3, 2, 1, 0):
            dgcnn = getattr(self, f"dgcnn_{level}")
            (
                current_coords,
                current_features,
                current_indices,
            ) = self._downsample_level(
                current_coords,
                current_features,
                current_indices,
                self.PYRAMID_POINT_COUNTS[level],
                dgcnn,
            )
            levels[level] = (
                current_coords,
                current_features,
                current_indices,
            )
        return levels

    def forward(self, source_xyz, target_xyz):
        if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
            raise ValueError("source_xyz must have shape (B, Nsource, 3)")
        if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3:
            raise ValueError("target_xyz must have shape (B, Ntarget, 3)")

        source_input, target_input = self.preprocessor(
            source_xyz, target_xyz
        )
        source_features_cf = self.reduce_channels(source_input)
        target_features_cf = self.reduce_channels(target_input)
        source_coords_cf = source_xyz.transpose(1, 2).contiguous()
        target_coords_cf = target_xyz.transpose(1, 2).contiguous()

        source_levels = self._shared_pyramid(
            source_coords_cf, source_features_cf
        )
        target_levels = self._shared_pyramid(
            target_coords_cf, target_features_cf
        )

        (
            source_global_coords_cf,
            source_global_features_cf,
            source_global_indices,
        ) = source_levels[self.global_match_level]
        (
            target_global_coords_cf,
            target_global_features_cf,
            _target_global_indices,
        ) = target_levels[self.global_match_level]

        source_global_coords = (
            source_global_coords_cf.transpose(1, 2).contiguous()
        )
        target_global_coords = (
            target_global_coords_cf.transpose(1, 2).contiguous()
        )
        source_global_features = (
            source_global_features_cf.transpose(1, 2).contiguous()
        )
        target_global_features = (
            target_global_features_cf.transpose(1, 2).contiguous()
        )

        match = self.global_matcher(
            source_global_coords,
            target_global_coords,
            source_global_features,
            target_global_features,
        )

        coarse_flow_full = interpolate_source_features(
            query_coords=source_xyz,
            support_coords=source_global_coords,
            support_features=match["coarse_flow"],
            k=3,
        )
        global_memory_full = interpolate_source_features(
            query_coords=source_xyz,
            support_coords=source_global_coords,
            support_features=match["global_memory"],
            k=3,
        )

        (
            final_flow,
            refined_flows,
            refined_warps,
            refinement_residuals,
            source_knn_indices,
        ) = self.iterative_refiner(
            source_xyz,
            target_xyz,
            source_features_cf.transpose(1, 2).contiguous(),
            target_features_cf.transpose(1, 2).contiguous(),
            coarse_flow_full,
            global_memory_full,
        )

        flow_stages = [coarse_flow_full, *refined_flows]
        warped_source_stages = [
            source_xyz + coarse_flow_full,
            *refined_warps,
        ]

        return {
            "result": final_flow,
            "flow_stages": flow_stages,
            "warped_source_stages": warped_source_stages,
            "global_assignment": match["assignment"],
            "global_match_confidence": match["match_confidence"],
            "source_global_indices": source_global_indices,
            "target_global_xyz": target_global_coords,
            "score_weights": match["score_weights"],
            "global_coarse_flow": match["coarse_flow"],
            "global_memory": global_memory_full,
            "refinement_residuals": refinement_residuals,
            "source_knn_indices": source_knn_indices,
        }
