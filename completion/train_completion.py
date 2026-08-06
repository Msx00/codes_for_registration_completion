#!/usr/bin/env python
"""Pure source-conditioned liver completion training (single GPU or DDP)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
SPAQNet_ROOT = ROOT / "SPAQNet"
import sys

if str(SPAQNet_ROOT) not in sys.path:
    sys.path.insert(0, str(SPAQNet_ROOT))

from models.liver_completion import LiverCompletionSPAQNet  # noqa: E402
from models.liver_generative_completion import (  # noqa: E402
    LiverGenerativeCompletionSPAQNet,
)
from utils.liver_data import (  # noqa: E402
    LiverCaseDataset,
    build_partial_views,
    collate_liver_cases,
    sample_corresponding_points,
)
from utils.liver_losses import (  # noqa: E402
    CompletionLoss,
    GenerativeCompletionLoss,
    KNN_BACKEND,
    completion_metrics,
    generative_completion_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/home/ma_sx/Project/Dataset/MedShapeNet-Liver",
    )
    parser.add_argument(
        "--save_dir",
        default=str(ROOT / "logs" / f"completion_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--max_train_cases", type=int, default=-1)
    parser.add_argument("--max_val_cases", type=int, default=-1)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--crops_per_gt", type=int, default=4)
    parser.add_argument("--overlap_min", type=float, default=0.15)
    parser.add_argument("--overlap_max", type=float, default=0.40)
    parser.add_argument("--anchor_overlap", type=float, default=0.25)
    parser.add_argument("--anchor_probability", type=float, default=0.50)
    parser.add_argument("--rotation_degrees", type=float, default=10.0)
    parser.add_argument("--translation_mm", type=float, default=2.0)
    parser.add_argument("--scale_min", type=float, default=0.98)
    parser.add_argument("--scale_max", type=float, default=1.02)
    parser.add_argument("--partial_jitter_mm", type=float, default=0.0)
    parser.add_argument("--augmentation_curriculum_epochs", type=int, default=0)

    parser.add_argument(
        "--architecture",
        choices=("generative", "displacement"),
        default="generative",
    )
    parser.add_argument("--feature_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=12)
    parser.add_argument("--context_points", type=int, default=256)
    parser.add_argument("--coarse_points", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--denoise_queries", type=int, default=64)
    parser.add_argument("--denoise_jitter", type=float, default=0.005)

    parser.add_argument(
        "--correspondence_loss",
        choices=("mse", "huber", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--huber_beta_mm", type=float, default=5.0)
    parser.add_argument("--missing_weight", type=float, default=1.5)
    parser.add_argument("--visible_weight", type=float, default=1.0)
    parser.add_argument(
        "--set_loss_mode",
        choices=("chamfer", "correntropy", "hybrid"),
        default="correntropy",
    )
    parser.add_argument("--correntropy_sigma", type=float, default=1.0)
    parser.add_argument(
        "--correntropy_trunc",
        "--correntropy_truncation",
        dest="correntropy_trunc",
        type=float,
        default=0.2,
    )
    parser.add_argument("--w_huber", type=float, default=1.0)
    parser.add_argument("--w_set", type=float, default=0.20)
    parser.add_argument("--w_partial", type=float, default=0.5)
    parser.add_argument("--w_smooth", type=float, default=0.05)
    parser.add_argument("--w_edge", type=float, default=0.05)
    parser.add_argument("--w_coarse_set", type=float, default=0.25)
    parser.add_argument("--w_mid_set", type=float, default=0.50)
    parser.add_argument("--w_fine_set", type=float, default=1.0)
    parser.add_argument("--w_denoise", type=float, default=0.5)
    parser.add_argument("--w_repulsion", type=float, default=0.01)
    parser.add_argument("--repulsion_k", type=int, default=5)
    parser.add_argument("--repulsion_radius", type=float, default=0.02)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_cases", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=8)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    parser.add_argument("--amp_init_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    if args.crops_per_gt < 1:
        parser.error("--crops_per_gt must be at least 1")
    if args.grad_accum_steps < 1:
        parser.error("--grad_accum_steps must be at least 1")
    if not 0 < args.overlap_min <= args.overlap_max <= 1:
        parser.error("overlap range must satisfy 0 < min <= max <= 1")
    if not args.overlap_min <= args.anchor_overlap <= args.overlap_max:
        parser.error("anchor_overlap must lie inside the overlap range")
    if not 0 <= args.anchor_probability <= 1:
        parser.error("anchor_probability must lie inside [0, 1]")
    if not 0 < args.scale_min <= args.scale_max:
        parser.error("scale range must satisfy 0 < min <= max")
    if args.feature_dim % args.num_heads != 0:
        parser.error("feature_dim must be divisible by num_heads")
    if args.encoder_depth < 1 or args.decoder_depth < 1:
        parser.error("encoder_depth and decoder_depth must be positive")
    if args.denoise_queries < 0 or args.denoise_jitter < 0:
        parser.error("denoise query count and jitter must be nonnegative")
    if args.correntropy_sigma <= 0:
        parser.error("correntropy_sigma must be positive")
    if args.repulsion_k < 1 or args.repulsion_radius < 0:
        parser.error("repulsion_k must be positive and radius nonnegative")
    if args.resume and not Path(args.resume).is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume}")
    return args


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        if world_size > 1:
            raise RuntimeError("Multi-process training requires CUDA")
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def distributed_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def all_reduce_min(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return value


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def make_loader(
    dataset: LiverCaseDataset,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_liver_cases,
    )
    return loader, sampler


def make_views(
    batch: dict[str, object],
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    training: bool,
) -> dict[str, object]:
    source_full = batch["source_full"].to(device, non_blocking=True)
    gt_full = batch["gt_full"].to(device, non_blocking=True)
    case_indices = batch["case_index"].to(device, non_blocking=True)
    source, gt = sample_corresponding_points(
        source_full,
        gt_full,
        args.num_points,
    )
    augmentation_strength = 1.0
    # Use spherical partial views only.  build_partial_views includes `epoch`
    # in the training-view seed, so every case receives new ball locations in
    # each epoch while validation remains deterministic (epoch 0).
    crop_types = ("ball",)
    if training and args.augmentation_curriculum_epochs > 0:
        augmentation_strength = min(
            1.0,
            float(epoch + 1) / args.augmentation_curriculum_epochs,
        )

    # Begin near the target 0.25 overlap and identity transform, then expose
    # the full overlap/crop/geometry distribution over the curriculum.
    overlap_min = args.anchor_overlap + augmentation_strength * (
        args.overlap_min - args.anchor_overlap
    )
    overlap_max = args.anchor_overlap + augmentation_strength * (
        args.overlap_max - args.anchor_overlap
    )
    anchor_probability = 1.0 - augmentation_strength * (
        1.0 - args.anchor_probability
    )
    scale_min = 1.0 + augmentation_strength * (args.scale_min - 1.0)
    scale_max = 1.0 + augmentation_strength * (args.scale_max - 1.0)
    return build_partial_views(
        source,
        gt,
        case_indices,
        epoch=epoch,
        crops_per_gt=args.crops_per_gt,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        anchor_overlap=args.anchor_overlap,
        anchor_probability=anchor_probability,
        seed=args.seed,
        training=training,
        rotation_degrees=args.rotation_degrees * augmentation_strength,
        translation_mm=args.translation_mm * augmentation_strength,
        scale_min=scale_min,
        scale_max=scale_max,
        partial_jitter_mm=args.partial_jitter_mm * augmentation_strength,
        crop_types=crop_types,
    )


def reduce_epoch_meters(
    meters: dict[str, torch.Tensor],
) -> dict[str, float]:
    names = list(meters)
    packed = torch.stack([meters[name].double() for name in names])
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = {name: packed[i] for i, name in enumerate(names)}
    sample_count = values["sample_count"].clamp_min(1.0)
    point_count = values["point_count"].clamp_min(1.0)
    observed_count = values["observed_point_count"].clamp_min(1.0)
    missing_count = values["missing_point_count"].clamp_min(1.0)
    return {
        "loss": (values["loss"] / sample_count).item(),
        "total_loss": (values["total_loss"] / sample_count).item(),
        "correspondence": (values["correspondence"] / sample_count).item(),
        "mse": (values["mse"] / sample_count).item(),
        "huber": (values["huber"] / sample_count).item(),
        "set": (values["set"] / sample_count).item(),
        "partial": (values["partial"] / sample_count).item(),
        "smooth": (values["smooth"] / sample_count).item(),
        "edge": (values["edge"] / sample_count).item(),
        "reconstruction": (
            values["reconstruction"] / sample_count
        ).item(),
        "coarse_set": (values["coarse_set"] / sample_count).item(),
        "mid_set": (values["mid_set"] / sample_count).item(),
        "oa_correntropy": (
            values["oa_correntropy"] / sample_count
        ).item(),
        "oa_coarse": (values["oa_coarse"] / sample_count).item(),
        "oa_mid": (values["oa_mid"] / sample_count).item(),
        "oa_fine": (values["oa_fine"] / sample_count).item(),
        "topk_hausdorff": (
            values["topk_hausdorff"] / sample_count
        ).item(),
        "sinkhorn": (values["sinkhorn"] / sample_count).item(),
        "repulsion": (values["repulsion"] / sample_count).item(),
        "denoise": (values["denoise"] / sample_count).item(),
        "loss_coarse": (values["loss_coarse"] / sample_count).item(),
        "loss_mid": (values["loss_mid"] / sample_count).item(),
        "loss_fine": (values["loss_fine"] / sample_count).item(),
        "loss_denoise": (values["loss_denoise"] / sample_count).item(),
        "loss_partial": (values["loss_partial"] / sample_count).item(),
        "loss_repulsion": (
            values["loss_repulsion"] / sample_count
        ).item(),
        "rmse_mm": torch.sqrt(
            values["squared_error_sum"] / point_count
        ).item(),
        "observed_rmse_mm": torch.sqrt(
            values["observed_squared_error_sum"] / observed_count
        ).item(),
        "missing_rmse_mm": torch.sqrt(
            values["missing_squared_error_sum"] / missing_count
        ).item(),
        "skipped_steps": values["skipped_steps"].item(),
    }


def run_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    rank: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter_names = (
        "loss",
        "total_loss",
        "correspondence",
        "mse",
        "huber",
        "set",
        "partial",
        "smooth",
        "edge",
        "reconstruction",
        "coarse_set",
        "mid_set",
        "oa_correntropy",
        "oa_coarse",
        "oa_mid",
        "oa_fine",
        "topk_hausdorff",
        "sinkhorn",
        "repulsion",
        "denoise",
        "loss_coarse",
        "loss_mid",
        "loss_fine",
        "loss_denoise",
        "loss_partial",
        "loss_repulsion",
        "sample_count",
        "squared_error_sum",
        "point_count",
        "observed_squared_error_sum",
        "observed_point_count",
        "missing_squared_error_sum",
        "missing_point_count",
        "skipped_steps",
    )
    meters = {
        name: torch.zeros((), dtype=torch.float64, device=device)
        for name in meter_names
    }
    iterator = loader
    if rank == 0:
        iterator = tqdm(
            loader,
            desc="Train" if training else "Eval",
            leave=False,
        )

    grad_context = torch.enable_grad if training else torch.no_grad
    if training:
        optimizer.zero_grad(set_to_none=True)
    with grad_context():
        for batch_index, batch in enumerate(iterator):
            views = make_views(batch, device, args, epoch, training)
            with autocast_context(device, args.amp):
                outputs = model(
                    views["source_xyz"],
                    views["partial_xyz"],
                    views["partial_mask"],
                    views["partial_dense_xyz"],
                    views["observed_mask"],
                )
                losses = criterion(
                    outputs,
                    views["gt_xyz"],
                    views["partial_mask"],
                    views["observed_mask"],
                )

            global_finite = all_reduce_min(
                torch.isfinite(losses["loss"]).to(torch.int32)
            )
            if not bool(global_finite.item()):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                meters["skipped_steps"] += 1
                continue

            if training:
                scaled_loss = losses["loss"] / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (
                    (batch_index + 1) % args.grad_accum_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                    )
                    global_grad_finite = all_reduce_min(
                        torch.isfinite(grad_norm).to(torch.int32)
                    )
                    if bool(global_grad_finite.item()):
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    else:
                        if scaler.is_enabled():
                            scaler.update(
                                new_scale=max(
                                    scaler.get_scale() / 2.0, 1.0
                                )
                            )
                        meters["skipped_steps"] += 1
                    optimizer.zero_grad(set_to_none=True)

            batch_size = int(views["gt_xyz"].shape[0])
            for name in (
                "loss",
                "total_loss",
                "correspondence",
                "mse",
                "huber",
                "set",
                "partial",
                "smooth",
                "edge",
                "reconstruction",
                "coarse_set",
                "mid_set",
                "oa_correntropy",
                "oa_coarse",
                "oa_mid",
                "oa_fine",
                "topk_hausdorff",
                "sinkhorn",
                "repulsion",
                "denoise",
                "loss_coarse",
                "loss_mid",
                "loss_fine",
                "loss_denoise",
                "loss_partial",
                "loss_repulsion",
            ):
                if name in losses:
                    meters[name] += losses[name].detach().double() * batch_size
            meters["sample_count"] += batch_size
            metric_function = (
                generative_completion_metrics
                if args.architecture == "generative"
                else completion_metrics
            )
            metrics = metric_function(
                outputs["completed_xyz"],
                views["gt_xyz"],
                views["observed_mask"],
            )
            for name, value in metrics.items():
                meters[name] += value.detach().double()

    return reduce_epoch_meters(meters)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_rmse_mm: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_rmse_mm": best_rmse_mm,
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_setup()
    try:
        set_seed(args.seed, rank)
        train_dataset = LiverCaseDataset(
            args.dataset_root,
            "train",
            max_cases=args.max_train_cases,
            seed=args.seed,
        )
        val_dataset = LiverCaseDataset(
            args.dataset_root,
            "validation",
            max_cases=args.max_val_cases,
            seed=args.seed,
        )
        train_loader, train_sampler = make_loader(
            train_dataset,
            args.batch_cases,
            args.num_workers,
            rank,
            world_size,
            shuffle=True,
        )
        val_loader, val_sampler = make_loader(
            val_dataset,
            args.batch_cases,
            args.num_workers,
            rank,
            world_size,
            shuffle=False,
        )

        if args.architecture == "generative":
            model = LiverGenerativeCompletionSPAQNet(
                feature_dim=args.feature_dim,
                num_heads=args.num_heads,
                k_neighbors=args.k_neighbors,
                context_points=args.context_points,
                num_output_points=args.num_points,
                coarse_points=args.coarse_points,
                encoder_depth=args.encoder_depth,
                decoder_depth=args.decoder_depth,
                denoise_queries=args.denoise_queries,
                denoise_jitter=args.denoise_jitter,
            ).to(device)
        else:
            model = LiverCompletionSPAQNet(
                feature_dim=args.feature_dim,
                num_heads=args.num_heads,
                k_neighbors=args.k_neighbors,
                context_points=args.context_points,
                aligned_observed=True,
            ).to(device)
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        if args.architecture == "generative":
            criterion = GenerativeCompletionLoss(
                set_loss_mode=args.set_loss_mode,
                correntropy_sigma=args.correntropy_sigma,
                correntropy_trunc=args.correntropy_trunc,
                w_coarse=args.w_coarse_set,
                w_mid=args.w_mid_set,
                w_fine=args.w_fine_set,
                w_denoise=args.w_denoise,
                w_partial=args.w_partial,
                w_repulsion=args.w_repulsion,
                repulsion_k=args.repulsion_k,
                repulsion_radius=args.repulsion_radius,
            ).to(device)
        else:
            criterion = CompletionLoss(
                correspondence_loss=args.correspondence_loss,
                huber_beta_mm=args.huber_beta_mm,
                missing_weight=args.missing_weight,
                visible_weight=args.visible_weight,
                set_loss_mode=args.set_loss_mode,
                correntropy_sigma=args.correntropy_sigma,
                correntropy_truncation=args.correntropy_trunc,
                w_huber=args.w_huber,
                w_set=args.w_set,
                w_partial=args.w_partial,
                w_smooth=args.w_smooth,
                w_edge=args.w_edge,
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and args.amp == "fp16",
            init_scale=args.amp_init_scale,
        )
        start_epoch = 0
        best_rmse_mm = float("inf")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint.get("scaler", {}))
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_rmse_mm = float(checkpoint["best_rmse_mm"])

        save_dir = Path(args.save_dir)
        if rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text(
                json.dumps(vars(args), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            total_parameters = sum(p.numel() for p in model.parameters())
            print(
                f"[Info] Pure completion training on {world_size} process(es); "
                f"architecture={args.architecture}; device={device}; "
                f"params={total_parameters:,}"
            )
            print(
                f"[Info] train cases={len(train_dataset):,}; "
                f"views/epoch={len(train_dataset) * args.crops_per_gt:,}; "
                f"validation cases={len(val_dataset):,}"
            )
            print(
                f"[Info] gradient accumulation={args.grad_accum_steps}; "
                f"effective global views/step≈"
                f"{args.batch_cases * args.crops_per_gt * world_size * args.grad_accum_steps}"
            )
            if args.architecture == "generative":
                print(
                    f"[Info] KNN backend={KNN_BACKEND}; loss coordinates="
                    "source-normalized (approximately unit sphere)"
                )
                print(
                    f"[Info] Correntropy sigma={args.correntropy_sigma}; "
                    f"trunc={args.correntropy_trunc}"
                )
                print(
                    f"[Info] W_PARTIAL={args.w_partial}; "
                    f"W_REPULSION={args.w_repulsion}; "
                    f"REPULSION_K={args.repulsion_k}; "
                    f"REPULSION_RADIUS={args.repulsion_radius}"
                )
            objective = (
                f"adaptive-query multi-stage {args.set_loss_mode} + "
                "denoising + partial consistency + repulsion; "
                "metric=Chamfer RMSE"
                if args.architecture == "generative"
                else f"correspondence={args.correspondence_loss}; "
                f"set_loss={args.set_loss_mode}"
            )
            print(
                f"[Info] overlaps={args.overlap_min:.2f}-"
                f"{args.overlap_max:.2f}; anchor={args.anchor_overlap:.2f} "
                f"(p={args.anchor_probability:.2f}); objective={objective}; "
                f"curriculum={args.augmentation_curriculum_epochs} epochs"
            )
        if dist.is_initialized():
            dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(0)
            if rank == 0:
                print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")

            train_metrics = run_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                scaler,
                device,
                args,
                epoch,
                rank,
            )
            val_metrics = run_epoch(
                model,
                criterion,
                val_loader,
                None,
                scaler,
                device,
                args,
                0,
                rank,
            )
            scheduler.step(val_metrics["rmse_mm"])

            if rank == 0:
                lr = optimizer.param_groups[0]["lr"]
                if args.architecture == "generative":
                    print(
                        "[Train] "
                        f"total_loss={train_metrics['total_loss']:.6f} "
                        f"loss(coarse/mid/fine)="
                        f"{train_metrics['loss_coarse']:.6f}/"
                        f"{train_metrics['loss_mid']:.6f}/"
                        f"{train_metrics['loss_fine']:.6f} "
                        f"loss(denoise/partial/repulsion)="
                        f"{train_metrics['loss_denoise']:.6f}/"
                        f"{train_metrics['loss_partial']:.6f}/"
                        f"{train_metrics['loss_repulsion']:.6f} "
                        f"chamfer_rmse(all/observed/missing)="
                        f"{train_metrics['rmse_mm']:.4f}/"
                        f"{train_metrics['observed_rmse_mm']:.4f}/"
                        f"{train_metrics['missing_rmse_mm']:.4f}mm "
                        f"skips={train_metrics['skipped_steps']:.0f}"
                    )
                else:
                    print(
                        "[Train] "
                        f"loss={train_metrics['loss']:.6f} "
                        f"corr(mse/huber)="
                        f"{train_metrics['correspondence']:.6f}"
                        f"({train_metrics['mse']:.6f}/"
                        f"{train_metrics['huber']:.6f}) "
                        f"set={train_metrics['set']:.6f} "
                        f"partial/smooth/edge="
                        f"{train_metrics['partial']:.6f}/"
                        f"{train_metrics['smooth']:.6f}/"
                        f"{train_metrics['edge']:.6f} "
                        f"rmse(all/missing)="
                        f"{train_metrics['rmse_mm']:.4f}/"
                        f"{train_metrics['missing_rmse_mm']:.4f}mm "
                        f"skips={train_metrics['skipped_steps']:.0f}"
                    )
                if args.architecture == "generative":
                    print(
                        "[Eval ] "
                        f"total_loss={val_metrics['total_loss']:.6f} "
                        f"loss(coarse/mid/fine)="
                        f"{val_metrics['loss_coarse']:.6f}/"
                        f"{val_metrics['loss_mid']:.6f}/"
                        f"{val_metrics['loss_fine']:.6f} "
                        f"loss(denoise/partial/repulsion)="
                        f"{val_metrics['loss_denoise']:.6f}/"
                        f"{val_metrics['loss_partial']:.6f}/"
                        f"{val_metrics['loss_repulsion']:.6f} "
                        f"chamfer_rmse(all/observed/missing)="
                        f"{val_metrics['rmse_mm']:.4f}/"
                        f"{val_metrics['observed_rmse_mm']:.4f}/"
                        f"{val_metrics['missing_rmse_mm']:.4f}mm "
                        f"lr={lr:.3e}"
                    )
                else:
                    print(
                        "[Eval ] "
                        f"loss={val_metrics['loss']:.6f} "
                        f"rmse(all/observed/missing)="
                        f"{val_metrics['rmse_mm']:.4f}/"
                        f"{val_metrics['observed_rmse_mm']:.4f}/"
                        f"{val_metrics['missing_rmse_mm']:.4f}mm "
                        f"lr={lr:.3e}"
                    )
                if val_metrics["rmse_mm"] < best_rmse_mm:
                    best_rmse_mm = val_metrics["rmse_mm"]
                    save_checkpoint(
                        save_dir / "best.pth",
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        epoch,
                        best_rmse_mm,
                        args,
                    )
                    print(f"[Info] New best completion RMSE: {best_rmse_mm:.4f} mm")
                save_checkpoint(
                    save_dir / "last.pth",
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch,
                    best_rmse_mm,
                    args,
                )
                record = {
                    "epoch": epoch,
                    "lr": lr,
                    "train": train_metrics,
                    "validation": val_metrics,
                }
                with (save_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if dist.is_initialized():
                dist.barrier()
    finally:
        distributed_cleanup()


if __name__ == "__main__":
    main()
