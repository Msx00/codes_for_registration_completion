import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from completion.SPAQNet.utils.liver_losses import symmetric_chamfer_l1_fp32
from loss import pointwise_huber_loss


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


def _stage_metric_names(GIRNet_arch, num_refinement_steps):
    if GIRNet_arch == "legacy":
        return ["final"]
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
    stage_names = _stage_metric_names(
        args.GIRNet_arch, args.num_refinement_steps
    )
    is_full2full = args.GIRNet_arch in ("full2full_v1", "full2full_v2")
    # learned_gate is a model-parameter scalar — identical across ranks.
    # Capture it from the last batch (no all_reduce needed).
    learned_gate_value = None

    # --- stats layout ---
    #  0: reg_cd             1: reg_mse           2: reg_huber
    #  3: aux_loss           4: source_sq         5: completion_gt_cd
    #  6: pred_completed_cd  7: pred_gt_cd        8: match_loss
    #  9: confidence_sum    10: confidence_count 11: point_count
    # 12: sample_count      13: point_l1_sum     14-16: score_weights
    # 17-22: match diag (6 values)
    BASE = 23
    stage_count = len(stage_names)
    # --- coarse diagnostics (only full2full) ---
    # source_global_sq, raw_match_sq, pre_tanh_match_sq, gated_match_sq
    # oracle_candidate_sq, oracle_candidate_sum (for mean)
    # source_global_pt, oracle_pt
    # raw_flow_sum, pre_tanh_flow_sum, coarse_flow_sum, flow_pt
    # conf_gate_sum, coarse_gate_sum, gate_pt
    # conf_gate_min_sum, conf_gate_max_sum, coarse_gate_min_sum, coarse_gate_max_sum
    COARSE_DIAG_COUNT = 20
    COARSE_DIAG_BASE = BASE + stage_count
    stats = torch.zeros(
        COARSE_DIAG_BASE + COARSE_DIAG_COUNT,
        device=device,
        dtype=torch.float64,
    )

    _C = lambda idx: COARSE_DIAG_BASE + idx  # noqa: E731
    CD_SRC_SQ, CD_RAW_SQ, CD_PRE_SQ, CD_GATE_SQ = _C(0), _C(1), _C(2), _C(3)
    CD_ORACLE_SQ, CD_ORACLE_SUM = _C(4), _C(5)
    CD_SRC_PT, CD_ORACLE_PT = _C(6), _C(7)
    CD_RAW_FLOW, CD_PRE_FLOW, CD_COARSE_FLOW, CD_FLOW_PT = _C(8), _C(9), _C(10), _C(11)
    CD_CONF_GATE, CD_COARSE_GATE, CD_GATE_PT = _C(12), _C(13), _C(14)
    CD_CONF_MIN, CD_CONF_MAX, CD_COARSE_MIN, CD_COARSE_MAX = _C(15), _C(16), _C(17), _C(18)
    # _C(19) reserved

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
            batch["input_ids"], batch["attn_mask"],
            E_kPa=batch["E_kPa"],
            nu=batch["nu"],
            return_completion=True,
            registration_target_xyz=registration_target_xyz,
            freeze_completion=True,
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

        match_out = {"loss": torch.tensor(0.0, device=device)}
        if args.GIRNet_arch == "full2full_v2":
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

        # ---- coarse diagnostics (full2full only) ----
        if is_full2full:
            src_g_idx = out.get("source_global_indices")
            tgt_g_xyz = out.get("target_global_xyz")
            raw_flow_mm = out.get("global_raw_coarse_flow_mm")
            pre_flow_mm = out.get("global_pre_tanh_coarse_flow_mm")
            coarse_flow_mm = out.get("global_coarse_flow_mm")
            conf_gate = out.get("global_confidence_gate")
            coarse_gate = out.get("global_coarse_gate")

            if src_g_idx is not None and tgt_g_xyz is not None:
                src_global = _batched_gather(src, src_g_idx)
                gt_global = _batched_gather(gt, src_g_idx)
                n_global = src_global.shape[1]

                # source_global RMSE
                src_g_sq = (src_global.float() - gt_global.float()).square().sum(dim=-1)
                stats[CD_SRC_SQ] += src_g_sq.sum().double()
                stats[CD_SRC_PT] += batch_size * n_global

                # oracle candidate
                d2_all = torch.cdist(
                    gt_global.float(), tgt_g_xyz.float(), p=2
                ).square()
                oracle_min_d2 = d2_all.min(dim=-1).values  # (B, N_coarse)
                stats[CD_ORACLE_SQ] += oracle_min_d2.sum().double()
                stats[CD_ORACLE_SUM] += torch.sqrt(oracle_min_d2.clamp_min(1e-12)).sum().double()
                stats[CD_ORACLE_PT] += batch_size * n_global

                # raw match RMSE
                if raw_flow_mm is not None:
                    raw_pred = src_global.float() + raw_flow_mm.float()
                    raw_sq = (raw_pred - gt_global.float()).square().sum(dim=-1)
                    stats[CD_RAW_SQ] += raw_sq.sum().double()

                # pre_tanh match RMSE
                if pre_flow_mm is not None:
                    pre_pred = src_global.float() + pre_flow_mm.float()
                    pre_sq = (pre_pred - gt_global.float()).square().sum(dim=-1)
                    stats[CD_PRE_SQ] += pre_sq.sum().double()

                # gated match RMSE
                if coarse_flow_mm is not None:
                    gated_pred = src_global.float() + coarse_flow_mm.float()
                    gated_sq = (gated_pred - gt_global.float()).square().sum(dim=-1)
                    stats[CD_GATE_SQ] += gated_sq.sum().double()

            # flow magnitude diagnostics
            if raw_flow_mm is not None:
                raw_n = raw_flow_mm.float().norm(p=2, dim=-1)  # (B, Nc)
                stats[CD_RAW_FLOW] += raw_n.sum().double()
            if pre_flow_mm is not None:
                pre_n = pre_flow_mm.float().norm(p=2, dim=-1)
                stats[CD_PRE_FLOW] += pre_n.sum().double()
            if coarse_flow_mm is not None:
                cflow_n = coarse_flow_mm.float().norm(p=2, dim=-1)
                stats[CD_COARSE_FLOW] += cflow_n.sum().double()
            if raw_flow_mm is not None:
                stats[CD_FLOW_PT] += raw_flow_mm.shape[0] * raw_flow_mm.shape[1]

            # gate diagnostics
            if conf_gate is not None:
                stats[CD_CONF_GATE] += conf_gate.float().sum().double()
                stats[CD_CONF_MIN] += conf_gate.float().min().double() * batch_size
                stats[CD_CONF_MAX] += conf_gate.float().max().double() * batch_size
            if coarse_gate is not None:
                stats[CD_COARSE_GATE] += coarse_gate.float().sum().double()
                stats[CD_COARSE_MIN] += coarse_gate.float().min().double() * batch_size
                stats[CD_COARSE_MAX] += coarse_gate.float().max().double() * batch_size
            if conf_gate is not None:
                stats[CD_GATE_PT] += conf_gate.numel()

            learned_gate_raw = out.get("global_learned_gate")
            if learned_gate_raw is not None:
                learned_gate_value = learned_gate_raw.detach().float().item()

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

    # ---- coarse diagnostics (point-RMSE style) ----
    if is_full2full:
        src_gp = stats[CD_SRC_PT].clamp_min(1.0)
        oracle_gp = stats[CD_ORACLE_PT].clamp_min(1.0)
        flow_gp = stats[CD_FLOW_PT].clamp_min(1.0)
        gate_gp = stats[CD_GATE_PT].clamp_min(1.0)

        metrics["eval_source_global_point_rmse"] = (
            torch.sqrt(stats[CD_SRC_SQ] / src_gp).item()
        )
        metrics["eval_oracle_candidate_mean_distance_mm"] = (
            stats[CD_ORACLE_SUM] / oracle_gp
        ).item()
        metrics["eval_oracle_candidate_point_rmse"] = (
            torch.sqrt(stats[CD_ORACLE_SQ] / oracle_gp).item()
        )
        metrics["eval_raw_match_point_rmse"] = (
            torch.sqrt(stats[CD_RAW_SQ] / src_gp).item()
        )
        metrics["eval_pre_tanh_match_point_rmse"] = (
            torch.sqrt(stats[CD_PRE_SQ] / src_gp).item()
        )
        metrics["eval_gated_global_match_point_rmse"] = (
            torch.sqrt(stats[CD_GATE_SQ] / src_gp).item()
        )
        metrics["eval_global_raw_flow_mm_mean"] = (
            stats[CD_RAW_FLOW] / flow_gp
        ).item()
        metrics["eval_global_pre_tanh_flow_mm_mean"] = (
            stats[CD_PRE_FLOW] / flow_gp
        ).item()
        metrics["eval_global_coarse_flow_mm_mean"] = (
            stats[CD_COARSE_FLOW] / flow_gp
        ).item()
        metrics["eval_confidence_gate_mean"] = (
            stats[CD_CONF_GATE] / gate_gp
        ).item()
        metrics["eval_confidence_gate_batch_min"] = (
            stats[CD_CONF_MIN] / sample_count
        ).item()
        metrics["eval_confidence_gate_batch_max"] = (
            stats[CD_CONF_MAX] / sample_count
        ).item()
        metrics["eval_coarse_gate_mean"] = (
            stats[CD_COARSE_GATE] / gate_gp
        ).item()
        metrics["eval_coarse_gate_batch_min"] = (
            stats[CD_COARSE_MIN] / sample_count
        ).item()
        metrics["eval_coarse_gate_batch_max"] = (
            stats[CD_COARSE_MAX] / sample_count
        ).item()
        metrics["eval_learned_gate"] = learned_gate_value
    else:
        for key in (
            "eval_source_global_point_rmse",
            "eval_oracle_candidate_mean_distance_mm",
            "eval_oracle_candidate_point_rmse",
            "eval_raw_match_point_rmse",
            "eval_pre_tanh_match_point_rmse",
            "eval_gated_global_match_point_rmse",
            "eval_global_raw_flow_mm_mean",
            "eval_global_pre_tanh_flow_mm_mean",
            "eval_global_coarse_flow_mm_mean",
            "eval_confidence_gate_mean",
            "eval_confidence_gate_batch_min",
            "eval_confidence_gate_batch_max",
            "eval_coarse_gate_mean",
            "eval_coarse_gate_batch_min",
            "eval_coarse_gate_batch_max",
            "eval_learned_gate",
        ):
            metrics[key] = 0.0

    return metrics
