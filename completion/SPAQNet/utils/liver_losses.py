"""Numerically stable losses for correspondence-preserving completion."""

from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn as nn

try:
    from pytorch3d.ops import knn_points

    KNN_BACKEND = "pytorch3d"
except ImportError:
    # Keep training usable in environments where PyTorch3D is unavailable.
    # The public loss implementations below still use the knn_points API;
    # installing PyTorch3D automatically selects its optimized implementation.
    _KNN = namedtuple("_KNN", ("dists", "idx", "knn"))
    KNN_BACKEND = "torch-fallback"

    def knn_points(
        p1: torch.Tensor,
        p2: torch.Tensor,
        lengths1: torch.Tensor | None = None,
        lengths2: torch.Tensor | None = None,
        norm: int = 2,
        K: int = 1,
        **_: object,
    ) -> _KNN:
        """Differentiable fallback matching the used PyTorch3D KNN API."""
        if norm not in {1, 2}:
            raise ValueError(f"norm must be 1 or 2, got {norm}")
        batch_size, p1_count, _ = p1.shape
        p2_count = p2.shape[1]
        if p2_count < 1 or K < 1:
            raise ValueError("p2 and K must be nonempty")
        lengths1 = _point_lengths(p1, lengths1)
        lengths2 = _point_lengths(p2, lengths2)
        distance = torch.cdist(p1.float(), p2.float(), p=float(norm))
        if norm == 2:
            distance = distance.square()
        p2_padding = (
            torch.arange(p2_count, device=p2.device)[None]
            >= lengths2[:, None]
        )
        distance = distance.masked_fill(
            p2_padding[:, None, :], float("inf")
        )
        used_k = min(int(K), p2_count)
        dists, indices = distance.topk(
            used_k, dim=-1, largest=False, sorted=True
        )
        if used_k < K:
            pad_shape = (batch_size, p1_count, int(K) - used_k)
            dists = torch.cat(
                [dists, dists.new_full(pad_shape, float("inf"))], dim=-1
            )
            indices = torch.cat(
                [indices, indices.new_zeros(pad_shape)], dim=-1
            )
        p1_padding = (
            torch.arange(p1_count, device=p1.device)[None]
            >= lengths1[:, None]
        )
        dists = dists.masked_fill(p1_padding[:, :, None], 0.0)
        return _KNN(dists=dists, idx=indices, knn=None)


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    view_shape = (values.shape[0],) + (1,) * (indices.ndim - 1)
    return values[batch.view(view_shape), indices]


def weighted_point_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    delta_normalized: torch.Tensor,
    missing_weight: float,
    visible_weight: float,
) -> torch.Tensor:
    """Huber loss on corresponding 3D points with missing-region emphasis."""
    prediction = prediction.float()
    target = target.float()
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    delta = delta_normalized.float().reshape(-1, 1).clamp_min(1e-6)
    quadratic = 0.5 * error.square() / delta
    linear = error - 0.5 * delta
    point_loss = torch.where(error <= delta, quadratic, linear)
    weights = torch.where(
        observed_mask,
        torch.as_tensor(visible_weight, device=error.device),
        torch.as_tensor(missing_weight, device=error.device),
    ).float()
    return (point_loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_point_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
    missing_weight: float,
    visible_weight: float,
) -> torch.Tensor:
    """Squared L2 loss on corresponding points with region weighting."""
    squared_l2 = (prediction.float() - target.float()).square().sum(dim=-1)
    weights = torch.where(
        observed_mask,
        torch.as_tensor(visible_weight, device=squared_l2.device),
        torch.as_tensor(missing_weight, device=squared_l2.device),
    ).float()
    return (squared_l2 * weights).sum() / weights.sum().clamp_min(1.0)


def _nearest_squared_distances(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional nearest squared distances, always computed in FP32."""
    with torch.autocast(device_type=first.device.type, enabled=False):
        distances = torch.cdist(first.float(), second.float()).square()
        return distances.amin(dim=-1), distances.amin(dim=-2)


def symmetric_chamfer_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_to_gt, gt_to_pred = _nearest_squared_distances(prediction, target)
    return 0.5 * (pred_to_gt.mean() + gt_to_pred.mean())


def symmetric_chamfer_rmse_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Differentiable symmetric Chamfer RMSE in normalized coordinates."""
    chamfer_mse = symmetric_chamfer_fp32(prediction, target)
    return torch.sqrt(chamfer_mse.clamp_min(1e-12))


def symmetric_chamfer_l1_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional mean nearest-neighbour Euclidean distance in FP32."""
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        distances = torch.cdist(prediction.float(), target.float())
        return 0.5 * (
            distances.amin(dim=-1).mean()
            + distances.amin(dim=-2).mean()
        )


def _point_lengths(
    points: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    if points.ndim != 3:
        raise ValueError(
            f"point clouds must have shape (B, P, D), got {points.shape}"
        )
    if lengths is None:
        return torch.full(
            (points.shape[0],),
            points.shape[1],
            dtype=torch.long,
            device=points.device,
        )
    lengths = lengths.to(device=points.device, dtype=torch.long)
    if lengths.shape != (points.shape[0],):
        raise ValueError(
            f"lengths must have shape ({points.shape[0]},), got {lengths.shape}"
        )
    if bool(((lengths < 0) | (lengths > points.shape[1])).any()):
        raise ValueError("point-cloud lengths lie outside the padded shape")
    return lengths


def correntropy_chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    y_lengths: torch.Tensor | None = None,
    norm: int = 1,
    sigma: float = 1.0,
    trunc: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Nonnegative bidirectional correntropy Chamfer loss.

    Padding and distances beyond ``trunc`` contribute zero similarity and are
    excluded from each sample's denominator.  A perfect match approaches 0.
    """
    if norm not in {1, 2}:
        raise ValueError(f"norm must be 1 or 2, got {norm}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if trunc is not None and trunc <= 0:
        raise ValueError(f"trunc must be positive or None, got {trunc}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("x and y must have matching batch and feature sizes")

    x_lengths = _point_lengths(x, x_lengths)
    y_lengths = _point_lengths(y, y_lengths)
    if bool((x_lengths == 0).any()) or bool((y_lengths == 0).any()):
        raise ValueError("KNN requires at least one valid point per cloud")

    # PyTorch3D KNN is the primary backend. Float32 distance, exp, masks, and
    # reductions avoid AMP underflow while preserving the autograd graph.
    x_nn = knn_points(
        x.float(), y.float(), lengths1=x_lengths, lengths2=y_lengths,
        norm=norm, K=1,
    )
    y_nn = knn_points(
        y.float(), x.float(), lengths1=y_lengths, lengths2=x_lengths,
        norm=norm, K=1,
    )
    distance_x = x_nn.dists[..., 0].float()
    distance_y = y_nn.dists[..., 0].float()
    valid_x = (
        torch.arange(x.shape[1], device=x.device)[None]
        < x_lengths[:, None]
    ) & torch.isfinite(distance_x)
    valid_y = (
        torch.arange(y.shape[1], device=y.device)[None]
        < y_lengths[:, None]
    ) & torch.isfinite(distance_y)
    if trunc is not None:
        valid_x = valid_x & (distance_x <= float(trunc))
        valid_y = valid_y & (distance_y <= float(trunc))

    similarity_x = torch.exp(-distance_x / float(sigma)).masked_fill(
        ~valid_x, 0.0
    )
    similarity_y = torch.exp(-distance_y / float(sigma)).masked_fill(
        ~valid_y, 0.0
    )
    count_x = valid_x.sum(dim=1).float().clamp_min(1.0)
    count_y = valid_y.sum(dim=1).float().clamp_min(1.0)
    mean_similarity_x = (similarity_x.sum(dim=1) / count_x).mean()
    mean_similarity_y = (similarity_y.sum(dim=1) / count_y).mean()
    return (2.0 - mean_similarity_x - mean_similarity_y).clamp_min(0.0)


def correntropy_chamfer_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    truncation: float | None = None,
) -> torch.Tensor:
    """Backward-compatible wrapper for existing completion loss callers."""
    return correntropy_chamfer_distance(
        prediction,
        target,
        norm=1,
        sigma=sigma,
        trunc=truncation,
    )


def oa_correntropy_chamfer_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma2: float = 1.0,
    truncation: float = 0.2,
    norm: int = 1,
) -> torch.Tensor:
    """OAReg-style negative correntropy CD for fixed-length point clouds.

    This follows the supplied implementation: L1/L2 KNN distance, hard
    truncation, exp(-distance/sigma2), and negative bidirectional correlation.
    Batch reduction is a mean instead of a sum so its scale is independent of
    the number of crop views in a batch.
    """
    if norm not in {1, 2}:
        raise ValueError(f"OA correntropy norm must be 1 or 2, got {norm}")
    sigma2 = max(float(sigma2), 1e-8)
    truncation = float(truncation)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        distances = torch.cdist(
            prediction.float(),
            target.float(),
            p=float(norm),
        )
        pred_to_gt = distances.amin(dim=-1)
        gt_to_pred = distances.amin(dim=-2)

        def negative_correlation(nearest: torch.Tensor) -> torch.Tensor:
            if truncation > 0:
                nearest = torch.where(
                    nearest < truncation,
                    nearest,
                    torch.zeros_like(nearest),
                )
            correlation = torch.exp(-nearest / sigma2).mean(dim=1)
            return correlation

        correlation = negative_correlation(pred_to_gt)
        correlation = correlation + negative_correlation(gt_to_pred)
        return -correlation.mean()


def one_sided_partial_loss(
    partial: torch.Tensor,
    pred: torch.Tensor,
    partial_lengths: torch.Tensor | None = None,
    pred_lengths: torch.Tensor | None = None,
    norm: int = 1,
) -> torch.Tensor:
    """Mean partial-to-prediction nearest-neighbour distance."""
    if norm not in {1, 2}:
        raise ValueError(f"norm must be 1 or 2, got {norm}")
    if partial.shape[0] != pred.shape[0] or partial.shape[2] != pred.shape[2]:
        raise ValueError(
            "partial and pred must have matching batch and feature sizes"
        )
    partial_lengths = _point_lengths(partial, partial_lengths)
    pred_lengths = _point_lengths(pred, pred_lengths)
    if bool((pred_lengths == 0).any()):
        raise ValueError("pred must contain at least one valid point per cloud")
    nearest = knn_points(
        partial.float(),
        pred.float(),
        lengths1=partial_lengths,
        lengths2=pred_lengths,
        norm=norm,
        K=1,
    ).dists[..., 0].float()
    valid = (
        torch.arange(partial.shape[1], device=partial.device)[None]
        < partial_lengths[:, None]
    ) & torch.isfinite(nearest)
    nearest = nearest.masked_fill(~valid, 0.0)
    per_sample = nearest.sum(dim=1) / valid.sum(dim=1).float().clamp_min(1.0)
    return per_sample.mean()


def one_sided_partial_coverage_fp32(
    partial: torch.Tensor,
    prediction: torch.Tensor,
    partial_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible wrapper using squared Euclidean distance."""
    return one_sided_partial_loss(
        partial,
        prediction,
        partial_lengths=partial_mask.sum(dim=1),
        norm=2,
    )


def directed_topk_hausdorff_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    topk_ratio: float = 0.05,
) -> torch.Tensor:
    """Mean of the worst prediction-to-target nearest-neighbour distances.

    This directed robust Hausdorff term targets floating generated points while
    avoiding the single-point instability of a strict maximum Hausdorff loss.
    Squared distances keep its units consistent with the Chamfer terms.
    """
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must lie in (0, 1], got {topk_ratio}")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        nearest = torch.cdist(
            prediction.float(), target.float()
        ).square().amin(dim=-1)
        k = max(1, math.ceil(nearest.shape[1] * float(topk_ratio)))
        return nearest.topk(k, dim=1, largest=True).values.mean()


def _even_subsample(points: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Deterministically reduce point count without biasing toward a prefix."""
    count = min(max(int(sample_count), 1), points.shape[1])
    if count == points.shape[1]:
        return points.float()
    indices = torch.linspace(
        0,
        points.shape[1] - 1,
        steps=count,
        device=points.device,
    ).round().long()
    return points[:, indices].float()


def sinkhorn_transport_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_count: int = 512,
    epsilon: float = 0.01,
    iterations: int = 20,
) -> torch.Tensor:
    """Entropic optimal-transport cost with uniform marginals in FP32.

    Log-domain Sinkhorn updates prevent underflow. A bounded point subset keeps
    the O(N^2) transport matrix practical for 2048-point completion training.
    """
    epsilon = max(float(epsilon), 1e-6)
    iterations = max(int(iterations), 1)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = _even_subsample(prediction, sample_count)
        gt = _even_subsample(target, sample_count)
        cost = torch.cdist(pred, gt).square()
        batch_size, pred_count, gt_count = cost.shape
        log_a = cost.new_full(
            (batch_size, pred_count), -math.log(pred_count)
        )
        log_b = cost.new_full(
            (batch_size, gt_count), -math.log(gt_count)
        )
        dual_pred = torch.zeros_like(log_a)
        dual_gt = torch.zeros_like(log_b)
        for _ in range(iterations):
            dual_pred = epsilon * (
                log_a
                - torch.logsumexp(
                    (dual_gt.unsqueeze(1) - cost) / epsilon,
                    dim=2,
                )
            )
            dual_gt = epsilon * (
                log_b
                - torch.logsumexp(
                    (dual_pred.unsqueeze(2) - cost) / epsilon,
                    dim=1,
                )
            )
        transport = torch.exp(
            (dual_pred.unsqueeze(2) + dual_gt.unsqueeze(1) - cost) / epsilon
        )
        return (transport * cost).sum(dim=(1, 2)).mean()


def repulsion_loss(
    points: torch.Tensor,
    lengths: torch.Tensor | None = None,
    k: int = 5,
    radius: float = 0.02,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize valid point pairs closer than ``radius``."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if radius < 0:
        raise ValueError(f"radius must be nonnegative, got {radius}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    lengths = _point_lengths(points, lengths)
    point_count = points.shape[1]
    if point_count < 2 or radius == 0:
        return points.float().sum() * 0.0

    used_k = min(int(k), point_count - 1)
    squared_distances = knn_points(
        points.float(),
        points.float(),
        lengths1=lengths,
        lengths2=lengths,
        norm=2,
        K=used_k + 1,
    ).dists[..., 1:].float()
    distances = torch.sqrt(squared_distances.clamp_min(float(eps)))
    valid_points = (
        torch.arange(point_count, device=points.device)[None]
        < lengths[:, None]
    )
    valid_neighbors = (
        torch.arange(used_k, device=points.device)[None, None, :]
        < (lengths - 1).clamp_min(0)[:, None, None]
    )
    valid = (
        valid_points[:, :, None]
        & valid_neighbors
        & torch.isfinite(squared_distances)
    )
    penalties = torch.relu(float(radius) - distances).square()
    penalties = penalties.masked_fill(~valid, 0.0)
    return penalties.sum() / valid.sum().float().clamp_min(1.0)


def repulsion_loss_fp32(
    prediction: torch.Tensor,
    sample_count: int = 512,
    neighbors: int = 5,
    margin: float = 0.02,
) -> torch.Tensor:
    """Backward-compatible wrapper for the previous optional loss API."""
    points = _even_subsample(prediction, sample_count)
    return repulsion_loss(points, k=neighbors, radius=margin)


def displacement_smoothness(
    prediction: torch.Tensor,
    source: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    displacement = prediction.float() - source.float()
    neighbor_displacement = _batch_gather(displacement, neighbor_indices)
    center = displacement.unsqueeze(2)
    return (neighbor_displacement - center).square().sum(dim=-1).mean()


def local_edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    neighbor_indices: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """Match local GT edge lengths without requiring normals or a mesh."""
    prediction = prediction.float()
    target = target.float()
    pred_edges = _batch_gather(prediction, neighbor_indices) - prediction.unsqueeze(2)
    target_edges = _batch_gather(target, neighbor_indices) - target.unsqueeze(2)
    pred_lengths = torch.linalg.vector_norm(pred_edges, dim=-1)
    target_lengths = torch.linalg.vector_norm(target_edges, dim=-1)
    error = (pred_lengths - target_lengths).abs()
    beta = max(float(beta), 1e-6)
    return torch.where(
        error < beta,
        0.5 * error.square() / beta,
        error - 0.5 * beta,
    ).mean()


class CompletionLoss(nn.Module):
    def __init__(
        self,
        correspondence_loss: str = "hybrid",
        huber_beta_mm: float = 5.0,
        missing_weight: float = 1.5,
        visible_weight: float = 1.0,
        set_loss_mode: str = "chamfer",
        correntropy_sigma: float = 0.10,
        correntropy_truncation: float = -1.0,
        w_huber: float = 1.0,
        w_set: float = 0.20,
        w_partial: float = 0.10,
        w_smooth: float = 0.05,
        w_edge: float = 0.05,
    ):
        super().__init__()
        if correspondence_loss not in {"mse", "huber", "hybrid"}:
            raise ValueError(
                f"Unknown correspondence_loss: {correspondence_loss}"
            )
        if set_loss_mode not in {"chamfer", "correntropy", "hybrid"}:
            raise ValueError(f"Unknown set_loss_mode: {set_loss_mode}")
        self.correspondence_loss = correspondence_loss
        self.huber_beta_mm = float(huber_beta_mm)
        self.missing_weight = float(missing_weight)
        self.visible_weight = float(visible_weight)
        self.set_loss_mode = set_loss_mode
        self.correntropy_sigma = float(correntropy_sigma)
        self.correntropy_truncation = (
            float(correntropy_truncation)
            if correntropy_truncation > 0
            else None
        )
        self.w_huber = float(w_huber)
        self.w_set = float(w_set)
        self.w_partial = float(w_partial)
        self.w_smooth = float(w_smooth)
        self.w_edge = float(w_edge)

    def _set_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        chamfer = symmetric_chamfer_fp32(prediction, target)
        if self.set_loss_mode == "chamfer":
            return chamfer
        correntropy = correntropy_chamfer_fp32(
            prediction,
            target,
            sigma=self.correntropy_sigma,
            truncation=self.correntropy_truncation,
        )
        if self.set_loss_mode == "correntropy":
            return correntropy
        return 0.5 * (chamfer + correntropy)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        gt_xyz: torch.Tensor,
        partial_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scale = outputs["scale"].float()
        centroid = outputs["centroid"].float()
        gt_normalized = (gt_xyz.float() - centroid) / scale
        delta_normalized = self.huber_beta_mm / scale.squeeze(-1).squeeze(-1)

        stage_predictions = (
            outputs["coarse_normalized"],
            outputs["mid_normalized"],
            outputs["fine_normalized"],
        )
        stage_weights = (0.25, 0.50, 1.00)
        huber = sum(
            stage_weight
            * weighted_point_huber(
                stage_prediction,
                gt_normalized,
                observed_mask,
                delta_normalized,
                self.missing_weight,
                self.visible_weight,
            )
            for stage_prediction, stage_weight in zip(
                stage_predictions,
                stage_weights,
            )
        )
        mse = sum(
            stage_weight
            * weighted_point_mse(
                stage_prediction,
                gt_normalized,
                observed_mask,
                self.missing_weight,
                self.visible_weight,
            )
            for stage_prediction, stage_weight in zip(
                stage_predictions,
                stage_weights,
            )
        )
        if self.correspondence_loss == "mse":
            correspondence = mse
        elif self.correspondence_loss == "huber":
            correspondence = huber
        else:
            correspondence = mse + 0.25 * huber

        fine = outputs["fine_normalized"]
        set_loss = self._set_loss(fine, gt_normalized)
        partial = outputs["partial_normalized"]
        partial_coverage = one_sided_partial_coverage_fp32(
            partial,
            fine,
            partial_mask,
        )
        smoothness = displacement_smoothness(
            fine,
            outputs["source_normalized"],
            outputs["neighbor_indices"],
        )
        edge = local_edge_loss(
            fine,
            gt_normalized,
            outputs["neighbor_indices"],
        )
        total = (
            self.w_huber * correspondence
            + self.w_set * set_loss
            + self.w_partial * partial_coverage
            + self.w_smooth * smoothness
            + self.w_edge * edge
        )
        return {
            "loss": total,
            "correspondence": correspondence,
            "mse": mse,
            "huber": huber,
            "set": set_loss,
            "partial": partial_coverage,
            "smooth": smoothness,
            "edge": edge,
        }


class GenerativeCompletionLoss(nn.Module):
    """Permutation-invariant loss for generated, non-corresponding points."""

    def __init__(
        self,
        set_loss_mode: str = "chamfer",
        correntropy_sigma: float = 1.0,
        correntropy_trunc: float | None = 0.2,
        w_coarse: float = 0.25,
        w_mid: float = 0.50,
        w_fine: float = 1.0,
        w_denoise: float = 0.5,
        w_partial: float = 0.5,
        w_repulsion: float = 0.01,
        repulsion_k: int = 5,
        repulsion_radius: float = 0.02,
    ):
        super().__init__()
        if set_loss_mode not in {"chamfer", "correntropy", "hybrid"}:
            raise ValueError(f"Unknown set_loss_mode: {set_loss_mode}")
        self.set_loss_mode = set_loss_mode
        self.correntropy_sigma = float(correntropy_sigma)
        self.correntropy_truncation = (
            float(correntropy_trunc)
            if correntropy_trunc is not None and correntropy_trunc > 0
            else None
        )
        self.w_coarse = float(w_coarse)
        self.w_mid = float(w_mid)
        self.w_fine = float(w_fine)
        self.w_denoise = float(w_denoise)
        self.w_partial = float(w_partial)
        self.w_repulsion = float(w_repulsion)
        self.repulsion_k = int(repulsion_k)
        self.repulsion_radius = float(repulsion_radius)

    def _set_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        chamfer = symmetric_chamfer_l1_fp32(prediction, target)
        if self.set_loss_mode == "chamfer":
            return chamfer
        correntropy = correntropy_chamfer_fp32(
            prediction,
            target,
            sigma=self.correntropy_sigma,
            truncation=self.correntropy_truncation,
        )
        if self.set_loss_mode == "correntropy":
            return correntropy
        return 0.5 * (chamfer + correntropy)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        gt_xyz: torch.Tensor,
        partial_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del observed_mask
        scale = outputs["scale"].float()
        centroid = outputs["centroid"].float()
        gt_normalized = (gt_xyz.float() - centroid) / scale
        coarse_set = self._set_loss(
            outputs["coarse_normalized"], gt_normalized
        )
        mid_set = self._set_loss(outputs["mid_normalized"], gt_normalized)
        fine_set = self._set_loss(outputs["fine_normalized"], gt_normalized)
        reconstruction = (
            self.w_coarse * coarse_set
            + self.w_mid * mid_set
            + self.w_fine * fine_set
        )
        zero = fine_set * 0.0
        pred_lengths = torch.full(
            (outputs["fine_normalized"].shape[0],),
            outputs["fine_normalized"].shape[1],
            dtype=torch.long,
            device=outputs["fine_normalized"].device,
        )
        partial_coverage = one_sided_partial_loss(
            outputs["partial_normalized"],
            outputs["fine_normalized"],
            partial_lengths=partial_mask.sum(dim=1),
            pred_lengths=pred_lengths,
            norm=1,
        )
        repulsion = repulsion_loss(
            outputs["fine_normalized"],
            lengths=pred_lengths,
            k=self.repulsion_k,
            radius=self.repulsion_radius,
        )
        denoised = outputs.get("denoised_normalized")
        denoise_target = outputs.get("denoise_target_normalized")
        if (
            self.w_denoise != 0.0
            and denoised is not None
            and denoise_target is not None
        ):
            with torch.autocast(
                device_type=denoised.device.type, enabled=False
            ):
                denoise = torch.linalg.vector_norm(
                    denoised.float() - denoise_target.float(), dim=-1
                ).mean()
        else:
            denoise = zero
        total = (
            self.w_coarse * coarse_set
            + self.w_mid * mid_set
            + self.w_fine * fine_set
            + self.w_denoise * denoise
            + self.w_partial * partial_coverage
            + self.w_repulsion * repulsion
        )
        return {
            "loss": total,
            "total_loss": total,
            "reconstruction": reconstruction,
            "coarse_set": coarse_set,
            "mid_set": mid_set,
            "set": fine_set,
            "loss_coarse": coarse_set,
            "loss_mid": mid_set,
            "loss_fine": fine_set,
            "repulsion": repulsion,
            "partial": partial_coverage,
            "denoise": denoise,
            "loss_repulsion": repulsion,
            "loss_partial": partial_coverage,
            "loss_denoise": denoise,
            # Compatibility keys keep shared trainer/checkpoint logging simple.
            "correspondence": zero,
            "mse": zero,
            "huber": zero,
            "smooth": zero,
            "edge": zero,
        }


@torch.no_grad()
def completion_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    squared_l2 = (prediction.float() - target.float()).square().sum(dim=-1)
    missing_mask = ~observed_mask
    observed_count = observed_mask.sum().clamp_min(1)
    missing_count = missing_mask.sum().clamp_min(1)
    return {
        "squared_error_sum": squared_l2.sum(),
        "point_count": torch.as_tensor(
            squared_l2.numel(),
            device=prediction.device,
            dtype=torch.float64,
        ),
        "observed_squared_error_sum": (
            squared_l2 * observed_mask.float()
        ).sum(),
        "observed_point_count": observed_count.to(torch.float64),
        "missing_squared_error_sum": (
            squared_l2 * missing_mask.float()
        ).sum(),
        "missing_point_count": missing_count.to(torch.float64),
    }


@torch.no_grad()
def generative_completion_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Symmetric Chamfer RMSE plus GT-region coverage RMSE accumulators."""
    pred_to_gt, gt_to_pred = _nearest_squared_distances(prediction, target)
    missing_mask = ~observed_mask
    return {
        "squared_error_sum": pred_to_gt.sum() + gt_to_pred.sum(),
        "point_count": torch.as_tensor(
            pred_to_gt.numel() + gt_to_pred.numel(),
            device=prediction.device,
            dtype=torch.float64,
        ),
        "observed_squared_error_sum": (
            gt_to_pred * observed_mask.float()
        ).sum(),
        "observed_point_count": observed_mask.sum().clamp_min(1).to(
            torch.float64
        ),
        "missing_squared_error_sum": (
            gt_to_pred * missing_mask.float()
        ).sum(),
        "missing_point_count": missing_mask.sum().clamp_min(1).to(
            torch.float64
        ),
    }
