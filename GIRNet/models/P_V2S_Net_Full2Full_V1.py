import torch
import torch.nn as nn

from models.T_layer_dgcnn import LayerDGCNN_v3
from models.farthest_point_sampling import farthest_point_sampling
from models.global_matcher import GlobalMatcher, interpolate_flow
from models.iterative_refiner import IterativeFlowRefiner
from models.k_nearest_neighbors import k_nearest_neighbors
from models.select import select_point_regions, select_points
from models.symmetric_preprocessor import SymmetricFullToFullPreprocessor


class PV2SNetFull2FullV1(nn.Module):
    """Minimal symmetric full-to-full PIVOTS V1.

    Tensor layout at this public boundary is (B, N, 3).  All returned flows
    keep the original source order and are expressed in normalized units.
    """

    PYRAMID_POINT_COUNTS = {4: 239, 3: 144, 2: 92, 1: 35, 0: 8}

    def __init__(
        self,
        feature_dim=50,
        points_per_region=35,
        global_match_level=2,
        global_match_temperature=0.1,
        global_match_dim=64,
        global_spatial_sigma=0.2,
        max_coarse_flow_normalized=0.25,
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

        # Names intentionally match compatible legacy PIVOTS encoder modules.
        self.reduce_channels = nn.Conv1d(56, self.feature_dim, kernel_size=1)
        self.dgcnn_4 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_3 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_2 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_1 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)
        self.dgcnn_0 = LayerDGCNN_v3(self.feature_dim, self.feature_dim)

        self.global_matcher = GlobalMatcher(
            feature_dim=self.feature_dim,
            projection_dim=global_match_dim,
            temperature=global_match_temperature,
            spatial_sigma=global_spatial_sigma,
            max_coarse_flow=max_coarse_flow_normalized,
        )
        self.iterative_refiner = IterativeFlowRefiner(
            feature_dim=self.feature_dim,
            num_steps=num_refinement_steps,
            k=refinement_k,
            debug=debug_refinement,
        )
        print(
            "Building PV2SNetFull2FullV1: "
            f"global match level={self.global_match_level}, "
            f"global match points={self.global_match_points}, "
            f"refinement steps={num_refinement_steps}, k={refinement_k}"
        )

    @property
    def global_match_points(self):
        return self.PYRAMID_POINT_COUNTS[self.global_match_level]

    def _downsample_level(self, coords, features, point_count, dgcnn):
        if coords.shape[-1] < point_count:
            raise ValueError(
                f"Input has {coords.shape[-1]} points but pyramid level "
                f"requires {point_count}"
            )
        indices, sampled_coords = farthest_point_sampling(
            coords, point_count, random=False
        )
        center_features = select_points(features, indices)
        k = min(self.points_per_region, coords.shape[-1])
        _, neighbor_indices = k_nearest_neighbors(
            pos_source=coords,
            pos_queries=sampled_coords,
            k=k,
        )
        neighbor_features = select_point_regions(features, neighbor_indices)
        sampled_features = dgcnn(center_features, neighbor_features)
        return sampled_coords, sampled_features

    def _shared_pyramid(self, coords, features):
        levels = {5: (coords, features)}
        current_coords, current_features = coords, features
        for level in (4, 3, 2, 1, 0):
            dgcnn = getattr(self, f"dgcnn_{level}")
            current_coords, current_features = self._downsample_level(
                current_coords,
                current_features,
                self.PYRAMID_POINT_COUNTS[level],
                dgcnn,
            )
            levels[level] = (current_coords, current_features)
        return levels

    def forward(self, source_xyz, target_xyz):
        if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
            raise ValueError("source_xyz must have shape (B, Nsource, 3)")
        if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3:
            raise ValueError("target_xyz must have shape (B, Ntarget, 3)")

        source_input, target_input = self.preprocessor(source_xyz, target_xyz)
        # The same convolution and DGCNN modules encode both clouds.
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

        source_global_coords_cf, source_global_features_cf = source_levels[
            self.global_match_level
        ]
        target_global_coords_cf, target_global_features_cf = target_levels[
            self.global_match_level
        ]
        match = self.global_matcher(
            source_global_coords_cf.transpose(1, 2).contiguous(),
            target_global_coords_cf.transpose(1, 2).contiguous(),
            source_global_features_cf.transpose(1, 2).contiguous(),
            target_global_features_cf.transpose(1, 2).contiguous(),
        )
        coarse_flow_full = interpolate_flow(
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
        return {
            "result": final_flow,
            "flow_stages": flow_stages,
            "warped_source_stages": warped_source_stages,
            "global_assignment": match["assignment"],
            "global_match_confidence": match["match_confidence"],
        }
