import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMatcherV3(nn.Module):
    """Global correspondence context for source-indexed deformation.

    Matching produces target context. A learned decoder predicts displacement;
    confidence is an input feature and never multiplicatively gates flow.
    """

    def __init__(
        self,
        feature_dim=50,
        projection_dim=64,
        feature_temperature=1.0,
        spatial_temperature=1.0,
    ):
        super().__init__()
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if feature_temperature <= 0:
            raise ValueError("feature_temperature must be positive")
        if spatial_temperature <= 0:
            raise ValueError("spatial_temperature must be positive")

        self.feature_dim = int(feature_dim)
        self.feature_temperature = float(feature_temperature)
        self.spatial_temperature = float(spatial_temperature)
        self.memory_dim = self.feature_dim + 4

        self.query_projection = nn.Linear(self.feature_dim, projection_dim)
        self.key_projection = nn.Linear(self.feature_dim, projection_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.geometry_mlp[-1].weight)
        nn.init.zeros_(self.geometry_mlp[-1].bias)

        # [spatial, feature, geometry] -> approximately [0.42, 0.42, 0.16].
        self.score_weight_logits = nn.Parameter(torch.tensor([0.0, 0.0, -1.0]))

        decoder_dim = 2 * self.feature_dim + 4
        self.coarse_flow_decoder = nn.Sequential(
            nn.Linear(decoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        nn.init.zeros_(self.coarse_flow_decoder[-1].weight)
        nn.init.zeros_(self.coarse_flow_decoder[-1].bias)

    @staticmethod
    def _normalize_score(score, eps=1e-5):
        mean = score.mean(dim=-1, keepdim=True)
        std = score.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (score - mean) / std

    def forward(
        self,
        source_coords,
        target_coords,
        source_features,
        target_features,
    ):
        output_dtype = source_features.dtype

        with torch.cuda.amp.autocast(enabled=False):
            source_coords_fp32 = source_coords.float()
            target_coords_fp32 = target_coords.float()
            source_features_fp32 = source_features.float()
            target_features_fp32 = target_features.float()

            q = F.normalize(self.query_projection(source_features_fp32), dim=-1)
            k = F.normalize(self.key_projection(target_features_fp32), dim=-1)

            # Apply temperatures after z-score normalization; unlike V2 this
            # keeps the temperature effective.
            feature_raw = torch.matmul(q, k.transpose(-1, -2))
            feature_score = (
                self._normalize_score(feature_raw) / self.feature_temperature
            )

            relative = (
                target_coords_fp32.unsqueeze(1)
                - source_coords_fp32.unsqueeze(2)
            )
            squared_distance = relative.square().sum(dim=-1)
            relative_norm = torch.sqrt(
                squared_distance.clamp_min(1e-12)
            ).unsqueeze(-1)
            geometry_input = torch.cat([relative, relative_norm], dim=-1)
            geometry_raw = self.geometry_mlp(geometry_input).squeeze(-1)
            geometry_score = self._normalize_score(geometry_raw)

            spatial_raw = -squared_distance
            spatial_score = (
                self._normalize_score(spatial_raw) / self.spatial_temperature
            )

            score_weights = torch.softmax(
                self.score_weight_logits.float(), dim=0
            )
            score = (
                score_weights[0] * spatial_score
                + score_weights[1] * feature_score
                + score_weights[2] * geometry_score
            )

            assignment = (
                torch.softmax(score, dim=-1)
                * torch.softmax(score, dim=-2)
            )
            assignment = assignment / assignment.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

            confidence = assignment.max(dim=-1).values
            target_feature_context = torch.matmul(
                assignment, target_features_fp32
            )
            matched_target = torch.matmul(assignment, target_coords_fp32)
            expected_relative_xyz = matched_target - source_coords_fp32

            global_memory = torch.cat(
                [
                    target_feature_context,
                    expected_relative_xyz,
                    confidence.unsqueeze(-1),
                ],
                dim=-1,
            )
            decoder_input = torch.cat(
                [source_features_fp32, global_memory], dim=-1
            )
            coarse_flow = self.coarse_flow_decoder(decoder_input)

        return {
            "coarse_flow": coarse_flow.to(dtype=output_dtype),
            "assignment": assignment.to(dtype=output_dtype),
            "match_confidence": confidence.to(dtype=output_dtype),
            "target_feature_context": target_feature_context.to(dtype=output_dtype),
            "expected_relative_xyz": expected_relative_xyz.to(dtype=output_dtype),
            "global_memory": global_memory.to(dtype=output_dtype),
            "score_weights": score_weights.detach(),
        }
