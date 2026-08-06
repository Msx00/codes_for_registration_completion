#!/usr/bin/env python
"""Inference and RMSE evaluation for liver completion checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
SPLATTN_ROOT = ROOT / "SPAQNet"
if str(SPLATTN_ROOT) not in sys.path:
    sys.path.insert(0, str(SPLATTN_ROOT))

from models.liver_completion import LiverCompletionSplAttN  # noqa: E402
from models.liver_generative_completion import (  # noqa: E402
    LiverGenerativeCompletionSplAttN,
)
from utils.liver_data import (  # noqa: E402
    CROP_TYPES,
    LiverCaseDataset,
    build_partial_views,
    collate_liver_cases,
    sample_corresponding_points,
)


DEFAULT_CHECKPOINT = (
    ROOT / "logs" / "full_aug_20260804_112526" / "best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="0 uses the checkpoint split limit; -1 uses the complete split",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=0,
        help="0 uses the checkpoint value",
    )
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--crops_per_case", type=int, default=1)
    parser.add_argument(
        "--crop_types",
        default=",".join(CROP_TYPES),
        help="Comma-separated: ball,plane,multi_ball",
    )
    parser.add_argument("--batch_cases", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--save_points",
        action="store_true",
        help="Save completed/source/partial/GT point clouds as PLY",
    )
    parser.add_argument(
        "--anchor_observed",
        action="store_true",
        help="Also hard-copy observed points for a legacy checkpoint",
    )
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not 0 < args.overlap <= 1:
        parser.error("--overlap must lie inside (0, 1]")
    if args.crops_per_case < 1:
        parser.error("--crops_per_case must be at least 1")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def strip_ddp_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def rmse_from_squared(squared: torch.Tensor) -> float:
    return math.sqrt(float(squared.mean().item()))


def case_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    observed_mask: torch.Tensor,
    generative: bool,
) -> dict[str, float]:
    pointwise_squared = (
        prediction.float() - target.float()
    ).square().sum(dim=-1)
    distances = torch.cdist(prediction.float(), target.float()).square()
    pred_to_gt = distances.amin(dim=-1)
    gt_to_pred = distances.amin(dim=-2)
    pointwise_rmse = rmse_from_squared(pointwise_squared)
    symmetric_chamfer_rmse = math.sqrt(
        float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean()).item())
    )
    if generative:
        squared = gt_to_pred
        mae_mm = float(
            0.5 * (pred_to_gt.sqrt().mean() + squared.sqrt().mean()).item()
        )
        all_rmse = symmetric_chamfer_rmse
    else:
        squared = pointwise_squared
        mae_mm = float(squared.sqrt().mean().item())
        all_rmse = pointwise_rmse
    baseline_squared = (source.float() - target.float()).square().sum(dim=-1)
    missing_mask = ~observed_mask
    return {
        "rmse_mm": all_rmse,
        "pointwise_rmse_mm": pointwise_rmse,
        "symmetric_chamfer_rmse_mm": symmetric_chamfer_rmse,
        "mae_mm": mae_mm,
        "observed_rmse_mm": rmse_from_squared(squared[observed_mask]),
        "missing_rmse_mm": rmse_from_squared(squared[missing_mask]),
        "source_baseline_rmse_mm": rmse_from_squared(baseline_squared),
        "source_baseline_mae_mm": float(
            baseline_squared.sqrt().mean().item()
        ),
    }


def update_totals(
    totals: dict[str, float],
    metrics: dict[str, float],
    prediction_count: int,
    target_count: int,
    observed_mask: torch.Tensor,
    generative: bool,
) -> None:
    totals["pointwise_squared_error_sum"] += float(
        metrics["pointwise_rmse_mm"] ** 2 * target_count
    )
    totals["pointwise_point_count"] += target_count
    chamfer_count = prediction_count + target_count
    totals["chamfer_squared_error_sum"] += float(
        metrics["symmetric_chamfer_rmse_mm"] ** 2 * chamfer_count
    )
    totals["chamfer_point_count"] += chamfer_count
    if generative:
        primary_count = chamfer_count
        primary_rmse = metrics["symmetric_chamfer_rmse_mm"]
    else:
        primary_count = target_count
        primary_rmse = metrics["pointwise_rmse_mm"]
    totals["squared_error_sum"] += primary_rmse**2 * primary_count
    totals["point_count"] += primary_count
    totals["absolute_error_sum"] += metrics["mae_mm"] * primary_count
    totals["absolute_error_count"] += primary_count
    missing_mask = ~observed_mask
    observed_count = int(observed_mask.sum().item())
    missing_count = int(missing_mask.sum().item())
    totals["observed_squared_error_sum"] += (
        metrics["observed_rmse_mm"] ** 2 * observed_count
    )
    totals["observed_point_count"] += observed_count
    totals["missing_squared_error_sum"] += (
        metrics["missing_rmse_mm"] ** 2 * missing_count
    )
    totals["missing_point_count"] += missing_count
    totals["baseline_squared_error_sum"] += (
        metrics["source_baseline_rmse_mm"] ** 2 * target_count
    )
    totals["baseline_point_count"] += target_count


def aggregate_metrics(totals: dict[str, float]) -> dict[str, float]:
    def pooled_rmse(sum_name: str, count_name: str) -> float:
        return math.sqrt(totals[sum_name] / max(totals[count_name], 1))

    return {
        "rmse_mm": pooled_rmse("squared_error_sum", "point_count"),
        "pointwise_rmse_mm": pooled_rmse(
            "pointwise_squared_error_sum", "pointwise_point_count"
        ),
        "symmetric_chamfer_rmse_mm": pooled_rmse(
            "chamfer_squared_error_sum", "chamfer_point_count"
        ),
        "mae_mm": totals["absolute_error_sum"]
        / max(totals["absolute_error_count"], 1),
        "observed_rmse_mm": pooled_rmse(
            "observed_squared_error_sum", "observed_point_count"
        ),
        "missing_rmse_mm": pooled_rmse(
            "missing_squared_error_sum", "missing_point_count"
        ),
        "source_baseline_rmse_mm": pooled_rmse(
            "baseline_squared_error_sum", "baseline_point_count"
        ),
    }


def write_ascii_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    if colors is not None and colors.shape != points.shape:
        raise ValueError(
            f"PLY points/colors shape mismatch: {points.shape} vs {colors.shape}"
        )
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if colors is not None:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")
        if colors is None:
            np.savetxt(handle, points, fmt="%.8f %.8f %.8f")
        else:
            vertices = np.concatenate([points, colors], axis=1)
            np.savetxt(
                handle,
                vertices,
                fmt="%.8f %.8f %.8f %d %d %d",
            )


def save_point_outputs(
    output_root: Path,
    case_key: str,
    view_index: int,
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    partial: torch.Tensor,
    partial_mask: torch.Tensor,
    observed_mask: torch.Tensor,
) -> None:
    folder = output_root / f"{case_key}__view_{view_index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        "completed.ply": prediction,
        "source.ply": source,
        "gt.ply": target,
        "partial.ply": partial[partial_mask],
    }
    for name, tensor in arrays.items():
        points = tensor.detach().float().cpu().numpy()
        write_ascii_ply(folder / name, points)

    prediction_points = prediction.detach().float().cpu().numpy()
    target_points = target.detach().float().cpu().numpy()
    comparison_points = np.concatenate(
        [prediction_points, target_points],
        axis=0,
    )
    comparison_colors = np.concatenate(
        [
            np.tile(
                np.array([[255, 64, 64]], dtype=np.uint8),
                (prediction_points.shape[0], 1),
            ),
            np.tile(
                np.array([[64, 255, 128]], dtype=np.uint8),
                (target_points.shape[0], 1),
            ),
        ],
        axis=0,
    )
    write_ascii_ply(
        folder / "completed_vs_gt.ply",
        comparison_points,
        comparison_colors,
    )
    np.savetxt(
        folder / "observed_mask.txt",
        observed_mask.detach().cpu().numpy().astype(np.uint8),
        fmt="%d",
    )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = strip_ddp_prefix(checkpoint["model"])
    saved_config = checkpoint.get("config", {})
    saved_architecture = saved_config.get("architecture", "")
    generative = saved_architecture == "generative" or any(
        key == "coarse_queries"
        or key.startswith("predicted_candidate_head.")
        for key in state
    )
    aligned_observed = any(
        key.startswith("observed_encoder.") for key in state
    )
    architecture = (
        "generative"
        if generative
        else "aligned-observed"
        if aligned_observed
        else "legacy"
    )

    dataset_root = args.dataset_root or saved_config.get("dataset_root", "")
    if not dataset_root:
        raise ValueError("dataset_root is missing from both CLI and checkpoint")
    num_points = args.num_points or int(saved_config.get("num_points", 2048))
    seed = args.seed if args.seed >= 0 else int(saved_config.get("seed", 42))
    max_cases = args.max_cases
    if max_cases == 0:
        split_limit_key = {
            "train": "max_train_cases",
            "validation": "max_val_cases",
            "val": "max_val_cases",
            "test": "max_test_cases",
        }.get(args.split, f"max_{args.split}_cases")
        max_cases = int(saved_config.get(split_limit_key, -1))
    crop_types = tuple(
        item.strip() for item in args.crop_types.split(",") if item.strip()
    )
    unknown_crops = set(crop_types) - set(CROP_TYPES)
    if not crop_types or unknown_crops:
        raise ValueError(f"Invalid crop types: {sorted(unknown_crops)}")

    if generative:
        model = LiverGenerativeCompletionSplAttN(
            feature_dim=int(saved_config.get("feature_dim", 192)),
            num_heads=int(saved_config.get("num_heads", 6)),
            k_neighbors=int(saved_config.get("k_neighbors", 12)),
            context_points=int(saved_config.get("context_points", 256)),
            num_output_points=num_points,
            coarse_points=int(saved_config.get("coarse_points", 256)),
            encoder_depth=int(saved_config.get("encoder_depth", 3)),
            decoder_depth=int(saved_config.get("decoder_depth", 4)),
            denoise_queries=int(saved_config.get("denoise_queries", 64)),
            denoise_jitter=float(saved_config.get("denoise_jitter", 0.005)),
        ).to(device)
    else:
        model = LiverCompletionSplAttN(
            feature_dim=int(saved_config.get("feature_dim", 192)),
            num_heads=int(saved_config.get("num_heads", 6)),
            k_neighbors=int(saved_config.get("k_neighbors", 12)),
            context_points=int(saved_config.get("context_points", 256)),
            aligned_observed=aligned_observed,
        ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if generative and args.anchor_observed:
        raise ValueError(
            "--anchor_observed is invalid for unordered generative output"
        )

    dataset = LiverCaseDataset(
        dataset_root,
        args.split,
        max_cases=max_cases,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_cases,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_liver_cases,
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).parent
        / f"inference_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        **vars(args),
        "dataset_root": dataset_root,
        "num_points": num_points,
        "seed": seed,
        "max_cases": max_cases,
        "architecture": architecture,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_rmse_mm": float(
            checkpoint.get("best_rmse_mm", float("nan"))
        ),
    }
    (output_dir / "inference_config.json").write_text(
        json.dumps(resolved_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[Info] checkpoint={args.checkpoint}\n"
        f"[Info] architecture={architecture}; epoch="
        f"{checkpoint.get('epoch', -1) + 1}; device={device}\n"
        f"[Info] split={args.split}; cases={len(dataset)}; "
        f"points={num_points}; overlap={args.overlap:.2f}; "
        f"crops/case={args.crops_per_case}\n"
        f"[Info] output_dir={output_dir}"
    )

    totals = {
        "squared_error_sum": 0.0,
        "point_count": 0,
        "pointwise_squared_error_sum": 0.0,
        "pointwise_point_count": 0,
        "chamfer_squared_error_sum": 0.0,
        "chamfer_point_count": 0,
        "absolute_error_sum": 0.0,
        "absolute_error_count": 0,
        "observed_squared_error_sum": 0.0,
        "observed_point_count": 0,
        "missing_squared_error_sum": 0.0,
        "missing_point_count": 0,
        "baseline_squared_error_sum": 0.0,
        "baseline_point_count": 0,
    }
    view_records = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            source_full = batch["source_full"].to(device, non_blocking=True)
            gt_full = batch["gt_full"].to(device, non_blocking=True)
            case_indices = batch["case_index"].to(device, non_blocking=True)
            source, gt = sample_corresponding_points(
                source_full,
                gt_full,
                num_points,
            )
            views = build_partial_views(
                source,
                gt,
                case_indices,
                epoch=0,
                crops_per_gt=args.crops_per_case,
                overlap_min=args.overlap,
                overlap_max=args.overlap,
                anchor_overlap=args.overlap,
                anchor_probability=1.0,
                seed=seed,
                training=args.crops_per_case > 1,
                rotation_degrees=0.0,
                translation_mm=0.0,
                scale_min=1.0,
                scale_max=1.0,
                partial_jitter_mm=0.0,
                crop_types=crop_types,
            )
            with autocast_context(device, args.amp):
                outputs = model(
                    views["source_xyz"],
                    views["partial_xyz"],
                    views["partial_mask"],
                    views["partial_dense_xyz"],
                    views["observed_mask"],
                )
            prediction = outputs["completed_xyz"].float()
            if args.anchor_observed and not aligned_observed:
                prediction = torch.where(
                    views["observed_mask"].unsqueeze(-1),
                    views["partial_dense_xyz"].float(),
                    prediction,
                )

            views_per_case = prediction.shape[0] // source.shape[0]
            for flat_index in range(prediction.shape[0]):
                batch_index = flat_index // views_per_case
                view_index = flat_index % views_per_case
                observed_mask = views["observed_mask"][flat_index]
                metrics = case_metrics(
                    prediction[flat_index],
                    views["gt_xyz"][flat_index],
                    views["source_xyz"][flat_index],
                    observed_mask,
                    generative,
                )
                update_totals(
                    totals,
                    metrics,
                    prediction[flat_index].shape[0],
                    views["gt_xyz"][flat_index].shape[0],
                    observed_mask,
                    generative,
                )
                case_path = batch["case_path"][batch_index]
                case_key = Path(case_path).relative_to(
                    Path(dataset_root) / args.split
                ).as_posix().replace("/", "__")
                record = {
                    "case_path": case_path,
                    "case_key": case_key,
                    "view_index": view_index,
                    "crop_type": views["crop_type"][flat_index],
                    "overlap": float(views["overlap"][flat_index].item()),
                    **metrics,
                }
                view_records.append(record)
                if args.save_points:
                    save_point_outputs(
                        output_dir / "points",
                        case_key,
                        view_index,
                        prediction[flat_index],
                        views["source_xyz"][flat_index],
                        views["gt_xyz"][flat_index],
                        views["partial_xyz"][flat_index],
                        views["partial_mask"][flat_index],
                        observed_mask,
                    )

    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in view_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    def mean_case_metric(metric_name: str) -> float:
        values_by_case: dict[str, list[float]] = {}
        for record in view_records:
            values_by_case.setdefault(record["case_path"], []).append(
                float(record[metric_name])
            )
        per_case_values = [
            sum(values) / len(values) for values in values_by_case.values()
        ]
        return sum(per_case_values) / max(len(per_case_values), 1)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": architecture,
        "split": args.split,
        "case_count": len(dataset),
        "view_count": len(view_records),
        "overlap": args.overlap,
        "anchor_observed_postprocess": bool(args.anchor_observed),
        "rmse_definition": (
            "symmetric_chamfer" if generative else "corresponding_points"
        ),
        "pointwise_rmse_definition": (
            "sqrt(mean_i(||completed[i]-gt[i]||_2^2))"
        ),
        "symmetric_chamfer_rmse_definition": (
            "sqrt(0.5*(mean_pred_nn_squared+mean_gt_nn_squared))"
        ),
        "mae_definition": (
            "symmetric_nearest_neighbor_euclidean"
            if generative
            else "corresponding_point_euclidean"
        ),
        "mean_case_mae_mm": mean_case_metric("mae_mm"),
        "mean_case_pointwise_rmse_mm": mean_case_metric(
            "pointwise_rmse_mm"
        ),
        "mean_case_symmetric_chamfer_rmse_mm": mean_case_metric(
            "symmetric_chamfer_rmse_mm"
        ),
        **aggregate_metrics(totals),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[Result] " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
