"""Source-indexed iterative flow refinement."""

import torch
import torch.nn as nn


def _batched_gather(points, indices):
    batch_index = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    return points[batch_index, indices]


def build_fixed_source_knn(source_coords, k):
    """Build source topology once from the original source coordinates."""
    n = source_coords.shape[1]
    k = min(int(k), max(n - 1, 1))
    if n < 2:
        raise ValueError("source graph requires at least two points")
    with torch.no_grad():
        distance = torch.cdist(
            source_coords.float(), source_coords.float(), p=2
        )
        diagonal = torch.eye(
            n, device=source_coords.device, dtype=torch.bool
        ).unsqueeze(0)
        distance = distance.masked_fill(diagonal, float("inf"))
        indices = torch.topk(
            distance,
            k=k,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
    return indices


class LocalFlowRefinerV3(nn.Module):
    """Source-indexed residual refinement with dual topology contexts."""

    def __init__(self, feature_dim=50, memory_dim=54, hidden_dim=128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.memory_dim = int(memory_dim)

        # [feature_j-feature_i, xyz_j-xyz_i, flow_j-flow_i]
        self.source_edge_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + 6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        query_dim = self.feature_dim + self.memory_dim + 3
        target_neighbor_dim = self.feature_dim + 4
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(target_neighbor_dim, hidden_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(target_neighbor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = hidden_dim ** -0.5

        residual_dim = (
            self.feature_dim
            + hidden_dim
            + hidden_dim
            + self.memory_dim
            + 3
        )
        self.residual_head = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _source_context(
        self,
        source_coords,
        source_features,
        current_flow,
        source_knn_indices,
    ):
        neighbor_features = _batched_gather(
            source_features, source_knn_indices
        ).float()
        neighbor_coords = _batched_gather(
            source_coords, source_knn_indices
        ).float()
        neighbor_flow = _batched_gather(
            current_flow, source_knn_indices
        ).float()

        edge_input = torch.cat(
            [
                neighbor_features - source_features.float().unsqueeze(2),
                neighbor_coords - source_coords.float().unsqueeze(2),
                neighbor_flow - current_flow.float().unsqueeze(2),
            ],
            dim=-1,
        )
        return self.source_edge_mlp(edge_input).amax(dim=2)

    def _target_context(
        self,
        warped_source_coords,
        target_coords,
        source_features,
        target_features,
        global_memory,
        current_flow,
        k,
    ):
        k = min(int(k), target_coords.shape[1])
        if k < 1:
            raise ValueError("target refinement k must be positive")

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

        target_neighbor_coords = _batched_gather(
            target_coords, nearest_index
        ).float()
        target_neighbor_features = _batched_gather(
            target_features, nearest_index
        ).float()
        relative = (
            target_neighbor_coords
            - warped_source_coords.float().unsqueeze(2)
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        neighbor_input = torch.cat(
            [
                target_neighbor_features,
                relative,
                distance.unsqueeze(-1),
            ],
            dim=-1,
        )

        query_input = torch.cat(
            [
                source_features.float(),
                global_memory.float(),
                current_flow.float(),
            ],
            dim=-1,
        )
        query = self.query_projection(query_input).unsqueeze(2)
        key = self.key_projection(neighbor_input)
        attention = torch.softmax(
            (query * key).sum(dim=-1).float() * self.scale,
            dim=-1,
        )
        value = self.value_projection(neighbor_input)
        return (attention.unsqueeze(-1) * value).sum(dim=2)

    def forward(
        self,
        source_coords,
        warped_source_coords,
        target_coords,
        source_features,
        target_features,
        global_memory,
        current_flow,
        source_knn_indices,
        target_k,
    ):
        source_context = self._source_context(
            source_coords,
            source_features,
            current_flow,
            source_knn_indices,
        )
        target_context = self._target_context(
            warped_source_coords,
            target_coords,
            source_features,
            target_features,
            global_memory,
            current_flow,
            target_k,
        )
        residual_input = torch.cat(
            [
                source_features.float(),
                source_context,
                target_context,
                global_memory.float(),
                current_flow.float(),
            ],
            dim=-1,
        )
        return self.residual_head(residual_input)


class IterativeFlowRefinerV3(nn.Module):
    """Recurrent V3 refinement with fixed source graph and dynamic target KNN."""

    def __init__(
        self,
        feature_dim=50,
        memory_dim=54,
        num_steps=3,
        target_k=35,
        source_graph_k=16,
        hidden_dim=128,
        debug=False,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if target_k < 1:
            raise ValueError("target_k must be at least 1")
        if source_graph_k < 1:
            raise ValueError("source_graph_k must be at least 1")
        self.num_steps = int(num_steps)
        self.target_k = int(target_k)
        self.source_graph_k = int(source_graph_k)
        self.debug = bool(debug)
        self.local_refiner = LocalFlowRefinerV3(
            feature_dim=feature_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        source_coords,
        target_coords,
        source_features,
        target_features,
        initial_flow,
        global_memory,
    ):
        source_knn_indices = build_fixed_source_knn(
            source_coords, self.source_graph_k
        )
        flow = initial_flow
        flow_stages = []
        warped_stages = []
        residual_stages = []

        for step in range(self.num_steps):
            warped_source = source_coords + flow
            if self.debug:
                print(
                    f"[Debug] V3 refinement {step + 1}/{self.num_steps}: "
                    "fixed source graph + dynamic target KNN"
                )
            residual = self.local_refiner(
                source_coords,
                warped_source,
                target_coords,
                source_features,
                target_features,
                global_memory,
                flow,
                source_knn_indices,
                self.target_k,
            )
            flow = flow + residual
            residual_stages.append(residual)
            flow_stages.append(flow)
            warped_stages.append(source_coords + flow)

        return (
            flow,
            flow_stages,
            warped_stages,
            residual_stages,
            source_knn_indices,
        )
