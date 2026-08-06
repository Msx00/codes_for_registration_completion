#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from MsDataset import LiverCompletionDataset, build_tokenizer, collate_fn
from loss import point_rmse
from completion.SPAQNet.utils.liver_losses import symmetric_chamfer_l1_fp32
from GIRNet_text_model import TextConditionedGIRNet


class CollateWrapper:
    def __init__(self):
        self.tokenizer = None

    def __call__(self, batch):
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer()
        return collate_fn(batch, self.tokenizer)


def load_model_checkpoint(model, checkpoint, checkpoint_path):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break

    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    print(f"[Info] Loaded checkpoint: {checkpoint_path}")
    print("[Info] Joint SPAQNet+GIRNet checkpoint matched strictly")


def write_ascii_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
):
    """Write one point cloud without requiring Open3D."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected PLY points shaped (N, 3), got {points.shape}")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != points.shape:
            raise ValueError(
                f"PLY points/colors shape mismatch: {points.shape} vs {colors.shape}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
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
            np.savetxt(handle, vertices, fmt="%.8f %.8f %.8f %d %d %d")


def write_colored_comparison(
    path: Path,
    point_clouds: list[tuple[np.ndarray, tuple[int, int, int]]],
):
    points = np.concatenate([cloud for cloud, _ in point_clouds], axis=0)
    colors = np.concatenate(
        [
            np.tile(np.asarray(color, dtype=np.uint8)[None], (cloud.shape[0], 1))
            for cloud, color in point_clouds
        ],
        axis=0,
    )
    write_ascii_ply(path, points, colors)


def save_case_ply_outputs(
    output_root: Path,
    case_name: str,
    source: torch.Tensor,
    partial: torch.Tensor,
    completed: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    pred_stages: list[torch.Tensor],
):
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "source.ply": source.detach().float().cpu().numpy(),
        "partial.ply": partial.detach().float().cpu().numpy(),
        "completed.ply": completed.detach().float().cpu().numpy(),
        "pred.ply": pred.detach().float().cpu().numpy(),
        "gt.ply": gt.detach().float().cpu().numpy(),
    }
    for filename, points in arrays.items():
        write_ascii_ply(case_dir / filename, points)

    for stage_index, stage in enumerate(pred_stages):
        stage_name = "coarse" if stage_index == 0 else f"refine_{stage_index}"
        write_ascii_ply(
            case_dir / f"pred_{stage_name}.ply",
            stage.detach().float().cpu().numpy(),
        )

    red = (255, 64, 64)
    green = (64, 255, 128)
    blue = (64, 128, 255)
    orange = (255, 180, 64)
    write_colored_comparison(
        case_dir / "completed_vs_gt.ply",
        [(arrays["completed.ply"], red), (arrays["gt.ply"], green)],
    )
    write_colored_comparison(
        case_dir / "pred_vs_gt.ply",
        [(arrays["pred.ply"], red), (arrays["gt.ply"], green)],
    )
    write_colored_comparison(
        case_dir / "source_vs_gt.ply",
        [(arrays["source.ply"], blue), (arrays["gt.ply"], green)],
    )
    write_colored_comparison(
        case_dir / "source_completed_pred_gt.ply",
        [
            (arrays["source.ply"], blue),
            (arrays["completed.ply"], orange),
            (arrays["pred.ply"], red),
            (arrays["gt.ply"], green),
        ],
    )


@torch.no_grad()
def test(
    model,
    loader,
    device,
    registration_target_mode,
    use_amp=True,
    ply_output_dir=None,
    case_paths=None,
    dataset_root=None,
):
    model.eval()

    meters = {
        "mse": 0.0,
        "rmse": 0.0,
        "pred_completed_cd": 0.0,
        "pred_gt_cd": 0.0,
        "mean_disp": 0.0,
        "num_batches": 0,
        "num_samples": 0,
    }

    start = time.time()
    sample_index = 0
    ply_manifest = []
    for batch in tqdm(loader, desc="Test", leave=True):
        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            out = model(
                batch["src_xyz"],
                batch["part_xyz"],
                batch["input_ids"],
                batch["attn_mask"],
                E_kPa=batch["E_kPa"],
                nu=batch["nu"],
                return_completion=True,
                registration_target_xyz=(
                    batch["gt_xyz"]
                    if registration_target_mode == "gt"
                    else None
                ),
                freeze_completion=True,
            )
            pred = out["pred_xyz"]
            completed = out["completed_xyz"]
            mse_value = F.mse_loss(pred, batch["gt_xyz"])
            rmse_value = point_rmse(pred, batch["gt_xyz"])
            pred_completed_cd = symmetric_chamfer_l1_fp32(pred, completed)
            pred_gt_cd = symmetric_chamfer_l1_fp32(pred, batch["gt_xyz"])
            mean_disp = torch.linalg.norm(pred - batch["src_xyz"], dim=-1).mean()

        batch_size = batch["src_xyz"].shape[0]
        meters["mse"] += mse_value.item()
        meters["rmse"] += rmse_value.item()
        meters["pred_completed_cd"] += pred_completed_cd.item()
        meters["pred_gt_cd"] += pred_gt_cd.item()
        meters["mean_disp"] += mean_disp.item()
        meters["num_batches"] += 1
        meters["num_samples"] += batch_size

        if ply_output_dir is not None:
            pred_stages = out.get("pred_stages_xyz", [pred])
            for batch_index in range(batch_size):
                case_path = (
                    Path(case_paths[sample_index])
                    if case_paths is not None and sample_index < len(case_paths)
                    else None
                )
                if case_path is not None and dataset_root is not None:
                    try:
                        relative_case = case_path.relative_to(
                            Path(dataset_root) / "test"
                        )
                        case_key = "__".join(relative_case.parts)
                    except ValueError:
                        case_key = case_path.name
                else:
                    case_key = "case"
                case_name = f"{sample_index:06d}_{case_key}"
                save_case_ply_outputs(
                    ply_output_dir,
                    case_name,
                    batch["src_xyz"][batch_index],
                    batch["part_xyz"][batch_index],
                    completed[batch_index],
                    pred[batch_index],
                    batch["gt_xyz"][batch_index],
                    [stage[batch_index] for stage in pred_stages],
                )
                ply_manifest.append(
                    {
                        "sample_index": sample_index,
                        "case_path": str(case_path) if case_path is not None else None,
                        "output_folder": str(ply_output_dir / case_name),
                    }
                )
                sample_index += 1
        else:
            sample_index += batch_size

    elapsed = time.time() - start
    n = max(meters["num_batches"], 1)
    metrics = {
        "mse": meters["mse"] / n,
        "rmse": meters["rmse"] / n,
        "pred_completed_cd": meters["pred_completed_cd"] / n,
        "pred_gt_cd": meters["pred_gt_cd"] / n,
        "mean_disp": meters["mean_disp"] / n,
        "num_batches": meters["num_batches"],
        "num_samples": meters["num_samples"],
        "elapsed_sec": elapsed,
        "sec_per_batch": elapsed / n,
    }
    return metrics, ply_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/home/ma_sx/Project/Dataset/MedShapeNet-Liver")
    parser.add_argument("--data_overlap", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/ma_sx/Project/Liver/test_results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_text", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bert_model_name", type=str, default="/home/ma_sx/Project/Liver/bert-base-uncased")
    parser.add_argument("--bert_local_files_only", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--GIRNet_checkpoint", type=str, default="/home/ma_sx/Project/Liver/GIRNet/checkpoints/GIRNet_v5/0/best_model.pth")
    parser.add_argument(
        "--completion_checkpoint",
        type=str,
        default="/home/ma_sx/Project/Liver2/completion/logs/full_aug_20260805_013524/best.pth",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--save_ply",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save per-case source/partial/completed/registered/GT PLY files.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = (
        checkpoint.get("config", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    GIRNet_arch = (
        checkpoint.get("GIRNet_arch")
        if isinstance(checkpoint, dict)
        else None
    ) or checkpoint_config.get("GIRNet_arch", "legacy")
    use_text = (
        bool(checkpoint_config.get("use_text", False))
        if args.use_text is None
        else args.use_text
    )
    registration_target_mode = checkpoint_config.get(
        "registration_target_mode", "completed"
    )
    if GIRNet_arch == "legacy" and not Path(args.GIRNet_checkpoint).is_file():
        parser.error(
            f"legacy GIRNet checkpoint not found: {args.GIRNet_checkpoint}"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LiverCompletionDataset(args.dataset_root, ["test"], args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=CollateWrapper(),
    )

    model = TextConditionedGIRNet(
        bert_model_name=args.bert_model_name,
        bert_local_files_only=args.bert_local_files_only,
        use_text=use_text,
        load_GIRNet_checkpoint=(
            args.GIRNet_checkpoint if GIRNet_arch == "legacy" else ""
        ),
        completion_checkpoint=args.completion_checkpoint,
        GIRNet_arch=GIRNet_arch,
        global_match_level=int(checkpoint_config.get("global_match_level", 2)),
        global_match_temperature=float(
            checkpoint_config.get("global_match_temperature", 0.1)
        ),
        global_match_dim=int(checkpoint_config.get("global_match_dim", 64)),
        global_spatial_sigma=float(
            checkpoint_config.get("global_spatial_sigma", 0.2)
        ),
        max_coarse_flow_normalized=float(
            checkpoint_config.get("max_coarse_flow_normalized", 0.25)
        ),
        num_refinement_steps=int(
            checkpoint_config.get("num_refinement_steps", 3)
        ),
        refinement_k=int(checkpoint_config.get("refinement_k", 35)),
        initialize_from_legacy_GIRNet=False,
    ).to(device)
    load_model_checkpoint(model, checkpoint, args.checkpoint)

    print(f"[Info] GIRNet_arch={GIRNet_arch}")
    print(f"[Info] registration_target_mode={registration_target_mode}")
    print(f"[Info] use_text={use_text}")
    ply_output_dir = output_dir / "ply" if args.save_ply else None
    print(f"[Info] save_ply={args.save_ply}")
    if ply_output_dir is not None:
        print(f"[Info] PLY output directory: {ply_output_dir}")

    metrics, ply_manifest = test(
        model,
        loader,
        device,
        registration_target_mode=registration_target_mode,
        use_amp=not args.no_amp,
        ply_output_dir=ply_output_dir,
        case_paths=dataset.cases["test"],
        dataset_root=args.dataset_root,
    )

    metrics_path = output_dir / "test_metrics.json"
    config_path = output_dir / "test_config.json"
    manifest_path = output_dir / "ply_manifest.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    test_config = {
        **vars(args),
        "GIRNet_arch": GIRNet_arch,
        "registration_target_mode": registration_target_mode,
        "resolved_use_text": use_text,
    }
    config_path.write_text(json.dumps(test_config, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_ply:
        manifest_path.write_text(
            json.dumps(ply_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("[Test]", json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[Info] Saved metrics to {metrics_path}")
    print(f"[Info] Saved config to {config_path}")
    if args.save_ply:
        print(f"[Info] Saved {len(ply_manifest)} case PLY folders to {ply_output_dir}")
        print(f"[Info] Saved PLY manifest to {manifest_path}")


if __name__ == "__main__":
    main()
