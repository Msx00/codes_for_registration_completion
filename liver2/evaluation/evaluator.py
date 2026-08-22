import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from completion.SPAQNet.utils.liver_losses import symmetric_chamfer_l1_fp32
from liver2.losses.correspondence import source_edge_error_mm
from liver2.losses.registration import pointwise_huber_loss


def registration_chamfer(source, prediction, target, eps=1e-6):
    """Symmetric CD in the same source-normalized frame as GIRNet."""
    centroid = source.float().mean(dim=1, keepdim=True)
    source_centered = source.float() - centroid
    scale = torch.linalg.vector_norm(source_centered, dim=-1).amax(
        dim=1, keepdim=True
    ).clamp_min(eps).unsqueeze(-1)
    return symmetric_chamfer_l1_fp32(
        (prediction.float() - centroid) / scale,
        (target.float() - centroid) / scale,
    )


def _stage_metric_names(num_refinement_steps):
    names = ["coarse"]
    for step in range(1, num_refinement_steps + 1):
        suffix = "_final" if step == num_refinement_steps else ""
        names.append(f"refine_{step}{suffix}")
    return names


def _auxiliary_stage_loss(pred_stages, gt, weights, beta_mm):
    auxiliary_predictions = pred_stages[:-1]
    if not auxiliary_predictions:
        return pred_stages[-1].new_zeros((), dtype=torch.float32)
    if len(auxiliary_predictions) != len(weights):
        raise ValueError(
            "aux_stage_weights count must equal the number of non-final "
            f"stages: got {len(weights)} weights for "
            f"{len(auxiliary_predictions)} stages"
        )
    return sum(
        float(weight) * pointwise_huber_loss(pred, gt, beta_mm=beta_mm)
        for weight, pred in zip(weights, auxiliary_predictions)
    )


def _batched_gather(points, indices):
    """Gather (B, M, C) points with (B, N) long indices -> (B, N, C)."""
    batch_index = torch.arange(points.shape[0], device=points.device).view(-1, 1)
    return points[batch_index, indices]


def _compute_eval_match_loss(
    global_assignment,
    source_global_indices,
    target_global_xyz,
    gt_xyz,
    match_sigma_mm,
):
    """Compute L_match for evaluation (same formula as training).

    Returns dict with loss + diagnostics. All coords must be in mm.
    """
    zero = torch.tensor(0.0, device=gt_xyz.device, dtype=torch.float32)
    zero_diag = {
        "loss": zero,
        "match_weight_mean": zero,
        "match_weight_batch_min": zero,
        "match_weight_batch_max": zero,
        "match_nn_distance_mm_mean": zero,
        "match_nn_distance_mm_median": zero,
        "match_probability_mean": zero,
    }
    if global_assignment is None or source_global_indices is None:
        return zero_diag

    if not torch.isfinite(target_global_xyz).all():
        return zero_diag
    if not torch.isfinite(gt_xyz).all():
        return zero_diag

    gt_coarse = _batched_gather(gt_xyz, source_global_indices)
    with torch.no_grad():
        distance2 = torch.cdist(
            gt_coarse.float(), target_global_xyz.float(), p=2
        ).square()
        pseudo_target_index = distance2.argmin(dim=-1)
        min_distance2 = distance2.gather(
            dim=-1, index=pseudo_target_index.unsqueeze(-1)
        ).squeeze(-1)
        match_weight = torch.exp(
            -min_distance2 / (2.0 * match_sigma_mm ** 2)
        )
        match_nn_distance_mm = torch.sqrt(min_distance2.clamp_min(1e-12))

    matched_probability = global_assignment.float().gather(
        dim=-1, index=pseudo_target_index.unsqueeze(-1)
    ).squeeze(-1)

    weight_sum = match_weight.sum().clamp_min(1e-8)
    L_match = -(
        match_weight * torch.log(matched_probability.clamp_min(1e-8))
    ).sum() / weight_sum

    return {
        "loss": L_match,
        "match_weight_mean": match_weight.mean(),
        "match_weight_batch_min": match_weight.min(),
        "match_weight_batch_max": match_weight.max(),
        "match_nn_distance_mm_mean": match_nn_distance_mm.mean(),
        "match_nn_distance_mm_median": match_nn_distance_mm.median(),
        "match_probability_mean": matched_probability.mean(),
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    """DDP-aware validation with full-to-full stage metrics."""
    model.eval()
    stage_names = _stage_metric_names(args.num_refinement_steps)

    # --- stats layout ---
    #  0: reg_cd             1: reg_mse           2: reg_huber
    #  3: aux_loss           4: source_sq         5: completion_gt_cd
    #  6: pred_completed_cd  7: pred_gt_cd        8: match_loss
    #  9: confidence_sum    10: confidence_count 11: point_count
    # 12: sample_count      13: point_l1_sum     14-16: score_weights
    # 17-22: match diag (6 values)
    BASE = 23
    stage_count = len(stage_names)
    V3_DIAG_BASE = BASE + stage_count
    # one edge-error scalar + one motion-magnitude sum per prediction stage
    V3_DIAG_COUNT = 1 + stage_count
    overlap_values = [float(value) for value in args.data_overlaps]
    OVERLAP_BASE = V3_DIAG_BASE + V3_DIAG_COUNT
    stats = torch.zeros(
        OVERLAP_BASE + 2 * len(overlap_values),
        device=device,
        dtype=torch.float64,
    )

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    iterator = tqdm(loader, desc="Eval", leave=False) if rank == 0 else loader

    for batch in iterator:
        for key in batch:
            batch[key] = batch[key].to(device)

        registration_target_xyz = (
            batch["gt_xyz"]
            if args.registration_target_mode == "gt"
            else None
        )
        out = model(
            batch["src_xyz"], batch["part_xyz"],
            E_kPa=batch["E_kPa"],
            nu=batch["nu"],
            return_completion=True,
            registration_target_xyz=registration_target_xyz,
            freeze_completion=True,
            partial_mask=batch.get("partial_mask"),
            overlap=batch.get("overlap"),
        )
        pred = out["pred_xyz"]
        pred_stages = out["pred_stages_xyz"]
        completed = out["completed_xyz"]
        gt = batch["gt_xyz"]
        src = batch["src_xyz"]
        if len(pred_stages) != len(stage_names):
            raise RuntimeError(
                f"Expected {len(stage_names)} prediction stages, "
                f"got {len(pred_stages)}"
            )

        batch_size, points_per_sample, _ = pred.shape

        # ---- existing metrics ----
        reg_cd = registration_chamfer(src, pred, gt)
        reg_mse = F.mse_loss(pred, gt)
        reg_huber = pointwise_huber_loss(
            pred, gt, beta_mm=args.huber_beta_mm
        )
        aux_loss = _auxiliary_stage_loss(
            pred_stages, gt, args.aux_stage_weights, args.huber_beta_mm,
        )
        source_squared = (src.float() - gt.float()).square().sum(dim=-1)
        completion_gt_cd = symmetric_chamfer_l1_fp32(completed.float(), gt.float())
        pred_completed_cd = symmetric_chamfer_l1_fp32(pred.float(), completed.float())
        pred_gt_cd = symmetric_chamfer_l1_fp32(pred.float(), gt.float())
        point_l1 = (pred.float() - gt.float()).abs().sum(dim=-1)

        match_out = _compute_eval_match_loss(
            out.get("global_assignment"),
            out.get("source_global_indices"),
            out.get("target_global_xyz"),
            gt,
            args.match_sigma_mm,
        )

        stats[0] += reg_cd.double() * batch_size
        stats[1] += reg_mse.double() * batch_size
        stats[2] += reg_huber.double() * batch_size
        stats[3] += aux_loss.double() * batch_size
        stats[4] += source_squared.sum().double()
        stats[5] += completion_gt_cd.double() * batch_size
        stats[6] += pred_completed_cd.double() * batch_size
        stats[7] += pred_gt_cd.double() * batch_size
        stats[8] += match_out["loss"].double() * batch_size

        confidence = out.get("global_match_confidence")
        if confidence is not None:
            stats[9] += confidence.float().sum().double()
            stats[10] += confidence.numel()
        stats[11] += batch_size * points_per_sample
        stats[12] += batch_size
        stats[13] += point_l1.sum().double()

        score_weights = out.get("score_weights")
        if score_weights is not None:
            stats[14] += score_weights[0].double() * batch_size
            stats[15] += score_weights[1].double() * batch_size
            stats[16] += score_weights[2].double() * batch_size

        for idx, key in enumerate(
            ["match_weight_mean", "match_weight_batch_min", "match_weight_batch_max",
             "match_nn_distance_mm_mean", "match_nn_distance_mm_median",
             "match_probability_mean"],
            start=17,
        ):
            val = match_out.get(key, torch.tensor(0.0, device=device))
            stats[idx] += val.double() * batch_size

        for index, stage_pred in enumerate(pred_stages):
            squared = (stage_pred.float() - gt.float()).square().sum(dim=-1)
            stats[BASE + index] += squared.sum().double()

        # ---- V3 correspondence/refinement diagnostics ----
        edge_error = source_edge_error_mm(
            pred, gt, out.get("source_knn_indices"), edge_k=args.edge_k
        )
        stats[V3_DIAG_BASE] += edge_error.double() * batch_size
        for stage_index, stage_pred in enumerate(pred_stages):
            if stage_index == 0:
                motion = stage_pred.float() - src.float()
            else:
                motion = (
                    stage_pred.float() - pred_stages[stage_index - 1].float()
                )
            stats[V3_DIAG_BASE + 1 + stage_index] += (
                torch.linalg.vector_norm(motion, dim=-1).sum().double()
            )

        final_squared = (pred.float() - gt.float()).square().sum(dim=-1)
        for overlap_index, overlap_value in enumerate(overlap_values):
            selected = torch.isclose(
                batch["overlap"].float(),
                batch["overlap"].new_tensor(overlap_value),
                atol=1e-6,
                rtol=0.0,
            )
            if selected.any():
                stats[OVERLAP_BASE + 2 * overlap_index] += (
                    final_squared[selected].sum().double()
                )
                stats[OVERLAP_BASE + 2 * overlap_index + 1] += (
                    selected.sum().double() * points_per_sample
                )

    # ---- all_reduce ----
    if is_dist:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    point_count = stats[11].clamp_min(1.0)
    sample_count = stats[12].clamp_min(1.0)
    confidence_count = stats[10].clamp_min(1.0)

    metrics = {
        "eval_reg_huber": (stats[2] / sample_count).item(),
        "eval_aux_stage_loss": (stats[3] / sample_count).item(),
        "eval_reg_mse": (stats[1] / sample_count).item(),
        "eval_reg_cd": (stats[0] / sample_count).item(),
        "eval_source_point_rmse": torch.sqrt(stats[4] / point_count).item(),
        "eval_completion_gt_cd": (stats[5] / sample_count).item(),
        "eval_pred_completed_cd": (stats[6] / sample_count).item(),
        "eval_pred_gt_cd": (stats[7] / sample_count).item(),
        "eval_match_loss": (stats[8] / sample_count).item(),
        "eval_point_mae": (stats[13] / point_count).item(),
        "eval_global_match_confidence": (stats[9] / confidence_count).item(),
        "eval_completion_cd": (stats[5] / sample_count).item(),
        "score_weight_spatial": (stats[14] / sample_count).item(),
        "score_weight_feature": (stats[15] / sample_count).item(),
        "score_weight_geometry": (stats[16] / sample_count).item(),
        "eval_match_weight_mean": (stats[17] / sample_count).item(),
        "eval_match_weight_batch_min_mean": (stats[18] / sample_count).item(),
        "eval_match_weight_batch_max_mean": (stats[19] / sample_count).item(),
        "eval_match_nn_distance_mm_mean": (stats[20] / sample_count).item(),
        "eval_match_nn_distance_mm_median": (stats[21] / sample_count).item(),
        "eval_match_probability_mean": (stats[22] / sample_count).item(),
    }
    for index, stage_name in enumerate(stage_names):
        metrics[f"eval_{stage_name}_point_rmse"] = torch.sqrt(
            stats[BASE + index] / point_count
        ).item()
    final_rmse = metrics[f"eval_{stage_names[-1]}_point_rmse"]
    metrics["eval_final_point_rmse"] = final_rmse
    metrics["eval_reg_point_rmse"] = final_rmse
    for overlap_index, overlap_value in enumerate(overlap_values):
        squared_sum = stats[OVERLAP_BASE + 2 * overlap_index]
        overlap_point_count = stats[
            OVERLAP_BASE + 2 * overlap_index + 1
        ]
        metrics[
            f"eval_overlap_{overlap_value:.2f}_point_rmse"
        ] = (
            torch.sqrt(squared_sum / overlap_point_count).item()
            if overlap_point_count.item() > 0
            else float("nan")
        )

    metrics["eval_v3_edge_error_mm"] = (
        stats[V3_DIAG_BASE] / sample_count
    ).item()
    metrics["eval_v3_coarse_displacement_mm_mean"] = (
        stats[V3_DIAG_BASE + 1] / point_count
    ).item()
    for step in range(1, len(stage_names)):
        metrics[f"eval_v3_refine{step}_residual_mm_mean"] = (
            stats[V3_DIAG_BASE + 1 + step] / point_count
        ).item()

    return metrics
