import torch
import torch.nn as nn


def _batched_gather(points, indices):
    """Gather (B, M, C) points with (B, N, K) indices."""
    batch_index = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    return points[batch_index, indices]


class LocalFlowRefiner(nn.Module):
    """Predict one source-indexed residual flow from a dynamic target KNN."""

    def __init__(self, feature_dim=50, hidden_dim=128):
        super().__init__()
        query_dim = feature_dim + 3
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(feature_dim + 4, hidden_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        warped_source_coords,
        target_coords,
        source_features,
        target_features,
        current_flow,
        k,
    ):
        k = min(int(k), target_coords.shape[1])
        if k < 1:
            raise ValueError("refinement k must be positive")

        # This search intentionally uses the newly warped source coordinates.
        # It is called again by IterativeFlowRefiner on every iteration.
        # Neighbor selection is discrete. Avoid retaining the full N x M
        # cdist graph; selected relative distances below remain differentiable
        # with respect to the current warped coordinates.
        with torch.no_grad():
            pairwise_distance = torch.cdist(
                warped_source_coords.float(), target_coords.float(), p=2
            )
            nearest_index = torch.topk(
                pairwise_distance,
                k=k,
                dim=-1,
                largest=False,
                sorted=False,
            ).indices
        target_neighbor_coords = _batched_gather(target_coords, nearest_index)
        target_neighbor_features = _batched_gather(target_features, nearest_index)
        relative = target_neighbor_coords.float() - warped_source_coords.float().unsqueeze(2)
        nearest_distance = torch.linalg.vector_norm(
            relative, dim=-1
        )
        neighbor_input = torch.cat(
            [
                target_neighbor_features.float(),
                relative,
                nearest_distance.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        query_input = torch.cat(
            [source_features.float(), current_flow.float()], dim=-1
        )
        query = self.query_projection(query_input).unsqueeze(2)
        key = self.key_projection(neighbor_input)
        attention = torch.softmax(
            (query * key).sum(dim=-1).float() * self.scale,
            dim=-1,
        )
        value = self.value_projection(neighbor_input)
        aggregated = (attention.unsqueeze(-1) * value).sum(dim=2)
        return self.residual_head(torch.cat([query_input, aggregated], dim=-1))


class IterativeFlowRefiner(nn.Module):
    """Shared recurrent warp/KNN/residual refinement."""

    def __init__(
        self,
        feature_dim=50,
        num_steps=3,
        k=35,
        hidden_dim=128,
        debug=False,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.num_steps = int(num_steps)
        self.k = int(k)
        self.debug = bool(debug)
        # One shared parameter set is recurrently reused for every step.
        self.local_refiner = LocalFlowRefiner(feature_dim, hidden_dim)

    def forward(
        self,
        source_coords,
        target_coords,
        source_features,
        target_features,
        initial_flow,
    ):
        flow = initial_flow
        flow_stages = []
        warped_stages = []
        for step in range(self.num_steps):
            warped_source = source_coords + flow
            if self.debug:
                print(
                    f"[Debug] refinement iteration {step + 1}/{self.num_steps}: "
                    "recomputing target KNN from warped source"
                )
            residual = self.local_refiner(
                warped_source,
                target_coords,
                source_features,
                target_features,
                flow,
                self.k,
            )
            flow = flow + residual
            flow_stages.append(flow)
            warped_stages.append(source_coords + flow)
        return flow, flow_stages, warped_stages
