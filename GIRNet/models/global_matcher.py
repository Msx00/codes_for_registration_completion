import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMatcher(nn.Module):
    """All-to-all dual-softmax matching at a coarse pyramid level."""

    def __init__(
        self,
        feature_dim=50,
        projection_dim=64,
        temperature=0.1,
        spatial_sigma=0.2,
        max_coarse_flow=0.25,
    ):
        super().__init__()
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if spatial_sigma <= 0:
            raise ValueError("spatial_sigma must be positive")
        if max_coarse_flow <= 0:
            raise ValueError("max_coarse_flow must be positive")
        self.temperature = float(temperature)
        self.spatial_sigma = float(spatial_sigma)
        self.max_coarse_flow = float(max_coarse_flow)
        self.query_projection = nn.Linear(feature_dim, projection_dim)
        self.key_projection = nn.Linear(feature_dim, projection_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.geometry_mlp[-1].weight)
        nn.init.zeros_(self.geometry_mlp[-1].bias)

        # Near-zero (but not exactly zero) sigmoid scales keep spatial matching
        # deterministic at initialization while avoiding a permanently dead
        # geometry branch when its final layer is zero-initialized.
        self.feature_score_logit = nn.Parameter(torch.tensor(-8.0))
        self.geometry_score_logit = nn.Parameter(torch.tensor(-8.0))
        self.coarse_gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        source_coords,
        target_coords,
        source_features,
        target_features,
    ):
        output_dtype = source_features.dtype
        device_type = "cuda" if source_coords.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            source_coords_fp32 = source_coords.float()
            target_coords_fp32 = target_coords.float()
            source_features_fp32 = source_features.float()
            target_features_fp32 = target_features.float()

            q = F.normalize(
                self.query_projection(source_features_fp32), dim=-1
            )
            k = F.normalize(
                self.key_projection(target_features_fp32), dim=-1
            )
            feature_score = (
                torch.matmul(q, k.transpose(-1, -2)) / self.temperature
            )

            relative = (
                target_coords_fp32.unsqueeze(1)
                - source_coords_fp32.unsqueeze(2)
            )
            squared_distance = relative.square().sum(dim=-1)
            relative_norm = torch.sqrt(squared_distance.clamp_min(1e-12)).unsqueeze(-1)
            geometry_input = torch.cat([relative, relative_norm], dim=-1)
            geometry_score = self.geometry_mlp(geometry_input).squeeze(-1)

            spatial_score = -squared_distance / (2.0 * self.spatial_sigma ** 2)
            feature_score_scale = torch.sigmoid(self.feature_score_logit.float())
            geometry_score_scale = torch.sigmoid(self.geometry_score_logit.float())
            score = (
                spatial_score
                + feature_score_scale * feature_score
                + geometry_score_scale * geometry_score
            )

            assignment = (
                torch.softmax(score, dim=-1)
                * torch.softmax(score, dim=-2)
            )
            assignment = assignment / assignment.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

            confidence = assignment.max(dim=-1).values
            num_target_points = target_coords_fp32.shape[1]
            confidence_threshold = min(
                2.0 / max(num_target_points, 1),
                1.0 - 1e-6,
            )
            confidence_gate = (
                (confidence - confidence_threshold)
                / max(1.0 - confidence_threshold, 1e-6)
            ).clamp(0.0, 1.0)
            learned_gate = torch.sigmoid(self.coarse_gate_logit.float())
            coarse_gate = confidence_gate * learned_gate

            matched_target = torch.matmul(assignment, target_coords_fp32)
            raw_coarse_flow = matched_target - source_coords_fp32
            coarse_flow = coarse_gate.unsqueeze(-1) * raw_coarse_flow
            max_flow = max(self.max_coarse_flow, 1e-6)
            coarse_flow = max_flow * torch.tanh(coarse_flow / max_flow)

        return {
            "coarse_flow": coarse_flow.to(dtype=output_dtype),
            "assignment": assignment.to(dtype=output_dtype),
            "match_confidence": confidence.to(dtype=output_dtype),
            "coarse_gate": coarse_gate.to(dtype=output_dtype),
        }


def interpolate_flow(
    query_coords,
    support_coords,
    support_flow,
    k=3,
    eps=1e-8,
):
    """Interpolate source-indexed flow with inverse-distance weighted 3-NN."""
    if support_coords.shape[1] < 1:
        raise ValueError("support_coords must contain at least one point")
    k = min(int(k), support_coords.shape[1])
    if k < 1:
        raise ValueError("k must be positive")

    distance = torch.cdist(query_coords.float(), support_coords.float(), p=2)
    nearest_distance, nearest_index = torch.topk(
        distance,
        k=k,
        dim=-1,
        largest=False,
        sorted=False,
    )
    batch_index = torch.arange(
        support_flow.shape[0], device=support_flow.device
    ).view(-1, 1, 1)
    neighbor_flow = support_flow[batch_index, nearest_index]
    inverse_distance = 1.0 / nearest_distance.clamp_min(eps)
    weights = inverse_distance / inverse_distance.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (weights.unsqueeze(-1) * neighbor_flow).sum(dim=-2)
