import torch
import torch.nn.functional as F


def _batched_gather(points, indices):
    batch_index = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    return points[batch_index, indices]


def _edge_vectors(points, source_knn_indices, edge_k=None):
    if edge_k is not None:
        source_knn_indices = source_knn_indices[:, :, :min(int(edge_k), source_knn_indices.shape[-1])]
    neighbors = _batched_gather(points, source_knn_indices).float()
    centers = points.float().unsqueeze(2)
    return neighbors - centers


def source_edge_consistency_loss(
    pred_xyz,
    gt_xyz,
    source_knn_indices,
    beta_mm=2.0,
    edge_k=None,
):
    """Huber loss on fixed source-indexed edge vectors in millimetres."""
    if source_knn_indices is None:
        return pred_xyz.new_zeros((), dtype=torch.float32)
    pred_edges = _edge_vectors(pred_xyz, source_knn_indices, edge_k=edge_k)
    gt_edges = _edge_vectors(gt_xyz, source_knn_indices, edge_k=edge_k)
    return F.smooth_l1_loss(
        pred_edges,
        gt_edges,
        beta=float(beta_mm),
        reduction="mean",
    )


def source_edge_error_mm(pred_xyz, gt_xyz, source_knn_indices, edge_k=None):
    """Mean Euclidean error between predicted and GT source-topology edges."""
    if source_knn_indices is None:
        return pred_xyz.new_zeros((), dtype=torch.float32)
    pred_edges = _edge_vectors(pred_xyz, source_knn_indices, edge_k=edge_k)
    gt_edges = _edge_vectors(gt_xyz, source_knn_indices, edge_k=edge_k)
    return torch.linalg.vector_norm(
        pred_edges - gt_edges, dim=-1
    ).mean()
