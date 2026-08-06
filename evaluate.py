import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from completion.SPAQNet.utils.liver_losses import symmetric_chamfer_l1_fp32
from loss import pointwise_huber_loss


def registration_chamfer(source, prediction, target, eps=1e-6):
    """Symmetric CD in the same source-normalized frame as PIVOTS."""
    centroid = source.float().mean(dim=1, keepdim=True)
    source_centered = source.float() - centroid
    scale = torch.linalg.vector_norm(source_centered, dim=-1).amax(
        dim=1, keepdim=True
    ).clamp_min(eps).unsqueeze(-1)
    return symmetric_chamfer_l1_fp32(
        (prediction.float() - centroid) / scale,
        (target.float() - centroid) / scale,
    )


def _stage_metric_names(pivots_arch, num_refinement_steps):
    if pivots_arch == "legacy":
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


@torch.no_grad()
def evaluate(model, loader, device, args):
    """DDP-aware validation with full-to-full stage metrics."""
    model.eval()
    stage_names = _stage_metric_names(
        args.pivots_arch, args.num_refinement_steps
    )

    # Base layout: CD, MSE, Huber, auxiliary, source squared L2,
    # completion CD, confidence sum, point count, sample count,
    # confidence count; followed by squared L2 for every prediction stage.
    base_count = 10
    stats = torch.zeros(
        base_count + len(stage_names),
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
        cd_target = gt if args.registration_target_mode == "gt" else completed.detach()
        reg_cd = registration_chamfer(src, pred, cd_target)
        reg_mse = F.mse_loss(pred, gt)
        reg_huber = pointwise_huber_loss(
            pred, gt, beta_mm=args.huber_beta_mm
        )
        aux_loss = _auxiliary_stage_loss(
            pred_stages,
            gt,
            args.aux_stage_weights,
            args.huber_beta_mm,
        )
        source_squared = (src.float() - gt.float()).square().sum(dim=-1)
        completion_cd = symmetric_chamfer_l1_fp32(
            completed.float(), gt.float()
        )

        stats[0] += reg_cd.double() * batch_size
        stats[1] += reg_mse.double() * batch_size
        stats[2] += reg_huber.double() * batch_size
        stats[3] += aux_loss.double() * batch_size
        stats[4] += source_squared.sum().double()
        stats[5] += completion_cd.double() * batch_size
        confidence = out.get("global_match_confidence")
        if confidence is not None:
            stats[6] += confidence.float().sum().double()
            stats[9] += confidence.numel()
        stats[7] += batch_size * points_per_sample
        stats[8] += batch_size

        for index, stage_pred in enumerate(pred_stages):
            squared = (stage_pred.float() - gt.float()).square().sum(dim=-1)
            stats[base_count + index] += squared.sum().double()

    if is_dist:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    point_count = stats[7].clamp_min(1.0)
    sample_count = stats[8].clamp_min(1.0)
    confidence_count = stats[9].clamp_min(1.0)
    metrics = {
        "eval_reg_huber": (stats[2] / sample_count).item(),
        "eval_aux_stage_loss": (stats[3] / sample_count).item(),
        "eval_reg_mse": (stats[1] / sample_count).item(),
        "eval_reg_cd": (stats[0] / sample_count).item(),
        "eval_source_point_rmse": torch.sqrt(stats[4] / point_count).item(),
        "eval_completion_cd": (stats[5] / sample_count).item(),
        "eval_global_match_confidence": (
            stats[6] / confidence_count
        ).item(),
    }
    for index, stage_name in enumerate(stage_names):
        metrics[f"eval_{stage_name}_point_rmse"] = torch.sqrt(
            stats[base_count + index] / point_count
        ).item()
    final_rmse = metrics[f"eval_{stage_names[-1]}_point_rmse"]
    metrics["eval_final_point_rmse"] = final_rmse
    metrics["eval_reg_point_rmse"] = final_rmse
    return metrics
