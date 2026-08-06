#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_completion_biomech_ddp.py
-------------------------------------------------------
多卡训练版本 - 使用 PyTorch DistributedDataParallel

Idea:
通过输入模型的旋转不变性特征
"""

import os, json, math, random, argparse
from pathlib import Path

import numpy as np
import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 新增: DDP相关导入
import torch.distributed as dist
import torch.multiprocessing as mp

from MsDataset import LiverCompletionDataset, build_tokenizer, collate_fn
from GIRNet_text_model import TextConditionedGIRNet
from completion.SPAQNet.utils.liver_losses import (
    GenerativeCompletionLoss,
    symmetric_chamfer_l1_fp32,
)
from loss import point_rmse, pointwise_huber_loss
from neohookean_loss import neohookean_loss
from evaluate import evaluate
from j_son import save_to_json


def setup_ddp(rank, world_size):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed, rank):
    """设置随机种子,每个进程使用不同的种子"""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_float_list(value):
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated floats, got {value!r}"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("float list cannot be empty")
    return values


# 将 collate_fn 包装为可序列化的函数
class CollateWrapper:
    """可序列化的 collate_fn 包装器"""
    def __init__(self):
        self.tokenizer = None
    
    def __call__(self, batch):
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer()
        return collate_fn(batch, self.tokenizer)


def save_ckpt(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def build_completion_criterion(config, device):
    return GenerativeCompletionLoss(
        set_loss_mode=config.get("set_loss_mode", "correntropy"),
        correntropy_sigma=float(config.get("correntropy_sigma", 1.0)),
        correntropy_trunc=float(config.get("correntropy_trunc", 0.2)),
        w_coarse=float(config.get("w_coarse_set", 0.25)),
        w_mid=float(config.get("w_mid_set", 0.50)),
        w_fine=float(config.get("w_fine_set", 1.0)),
        w_denoise=float(config.get("w_denoise", 0.50)),
        w_partial=float(config.get("w_partial", 0.50)),
        w_repulsion=float(config.get("w_repulsion", 0.01)),
        repulsion_k=int(config.get("repulsion_k", 5)),
        repulsion_radius=float(config.get("repulsion_radius", 0.02)),
    ).to(device)


def registration_chamfer(source, prediction, completed, eps=1e-6):
    """Symmetric CD between registered source and completed target."""
    centroid = source.float().mean(dim=1, keepdim=True)
    source_centered = source.float() - centroid
    scale = torch.linalg.vector_norm(source_centered, dim=-1).amax(
        dim=1, keepdim=True
    ).clamp_min(eps).unsqueeze(-1)
    prediction_normalized = (prediction.float() - centroid) / scale
    completed_normalized = (completed.float() - centroid) / scale
    return symmetric_chamfer_l1_fp32(
        prediction_normalized,
        completed_normalized,
    )


def _compute_grad_norm(parameters):
    """Compute total L2 gradient norm for a set of parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _gradients_are_finite(parameters):
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def stage_metric_names(GIRNet_arch, num_refinement_steps):
    if GIRNet_arch == "legacy":
        return ["final"]
    names = ["coarse"]
    for step in range(1, num_refinement_steps + 1):
        suffix = "_final" if step == num_refinement_steps else ""
        names.append(f"refine_{step}{suffix}")
    return names


def auxiliary_stage_huber_loss(pred_stages, gt, weights, beta_mm):
    """Huber supervision for every non-final full2full stage."""
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    completion_criterion,
    device,
    rank,
    args,
    completion_param_ids,
):
    model.train()
    if args.train_stage == "registration":
        # DDP.train() recurses into all children; restore the frozen completion
        # network to eval mode for deterministic, independent target creation.
        model.module.completion.eval()

    stage_names = stage_metric_names(
        args.GIRNet_arch, args.num_refinement_steps
    )
    loss_names = [
        "total_loss",
        "reg_huber",
        "aux_stage_loss",
        "reg_mse",
        "reg_cd",
        "physics",
        "completion",
        "completion_fine",
        "completion_partial",
        "completion_repulsion",
    ]
    # loss sums, gradient sums, source squared error, confidence sum,
    # sample/point/confidence/batch counts, non-finite gradient batch count,
    # then one squared-error sum per stage.
    base_meter_count = len(loss_names) + 9
    meters = torch.zeros(
        base_meter_count + len(stage_names),
        device=device,
        dtype=torch.float64,
    )

    iterator = tqdm(loader, desc="Train", leave=False) if rank == 0 else loader

    for batch_idx, batch in enumerate(iterator):
        try:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):

                # --- forward ---
                if args.registration_target_mode == "gt":
                    registration_target_xyz = batch["gt_xyz"]
                else:
                    registration_target_xyz = None

                out = model(
                    batch["src_xyz"], batch["part_xyz"],
                    batch["input_ids"], batch["attn_mask"],
                    E_kPa=batch["E_kPa"],
                    nu=batch["nu"],
                    return_completion=True,
                    registration_target_xyz=registration_target_xyz,
                    freeze_completion=(args.train_stage == "registration"),
                )
                pred = out["pred_xyz"]
                pred_stages = out["pred_stages_xyz"]
                completed = out["completed_xyz"]
                completion_outputs = out["completion_outputs"]
                if len(pred_stages) != len(stage_names):
                    raise RuntimeError(
                        f"Expected {len(stage_names)} prediction stages for "
                        f"{args.GIRNet_arch}, got {len(pred_stages)}"
                    )

                # --- registration losses ---
                L_reg_huber = pointwise_huber_loss(
                    pred, batch["gt_xyz"], beta_mm=args.huber_beta_mm,
                )
                L_aux_stages = auxiliary_stage_huber_loss(
                    pred_stages,
                    batch["gt_xyz"],
                    args.aux_stage_weights,
                    args.huber_beta_mm,
                )
                L_reg_mse = F.mse_loss(pred, batch["gt_xyz"])

                # CD target depends on mode
                if args.registration_target_mode == "gt":
                    cd_target = batch["gt_xyz"]
                else:
                    cd_target = completed.detach()
                L_reg_cd = registration_chamfer(
                    batch["src_xyz"], pred, cd_target,
                )

                # --- physics (off by default) ---
                if args.w_phys > 0:
                    L_phys = neohookean_loss(
                        batch["src_xyz"],
                        pred - batch["src_xyz"],
                        batch["E_kPa"],
                        batch["nu"],
                        k=args.phys_k,
                        reg=args.phys_reg,
                    )
                else:
                    L_phys = torch.tensor(0.0, device=device)

                # --- completion loss (logging only in registration mode) ---
                if completion_outputs is not None:
                    partial_mask = torch.ones(
                        batch["part_xyz"].shape[:2],
                        dtype=torch.bool,
                        device=device,
                    )
                    completion_losses = completion_criterion(
                        completion_outputs,
                        batch["gt_xyz"],
                        partial_mask,
                        torch.zeros_like(partial_mask),
                    )
                    L_completion = completion_losses["total_loss"]
                else:
                    L_completion = torch.tensor(0.0, device=device)
                    completion_losses = {
                        "loss_fine": L_completion,
                        "loss_partial": L_completion,
                        "loss_repulsion": L_completion,
                    }

                # --- total ---
                loss = (
                    args.w_reg_huber * L_reg_huber
                    + args.w_aux_stages * L_aux_stages
                    + args.w_reg_cd * L_reg_cd
                    + args.w_phys * L_phys
                    + args.w_completion * L_completion
                )

            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[Warning] skipped non-finite batch {batch_idx}")
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # --- gradient norms (after unscale, before clip) ---
            reg_params = [
                p for p in model.parameters()
                if id(p) not in completion_param_ids and p.requires_grad
            ]
            comp_params = [
                p for p in model.parameters()
                if id(p) in completion_param_ids and p.requires_grad
            ]
            local_nonfinite = not _gradients_are_finite(reg_params)
            nonfinite_flag = torch.tensor(
                int(local_nonfinite),
                device=device,
                dtype=torch.int32,
            )
            # All ranks must make the same optimizer-step decision.
            dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)
            nonfinite_registration_gradients = bool(nonfinite_flag.item())

            if nonfinite_registration_gradients:
                reg_grad_norm = 0.0
                comp_grad_norm = (
                    _compute_grad_norm(comp_params)
                    if _gradients_are_finite(comp_params)
                    else 0.0
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if rank == 0:
                    print(
                        "[Warning] Non-finite registration gradients; "
                        "skipped optimizer step"
                    )
            else:
                reg_grad_norm = _compute_grad_norm(reg_params)
                comp_grad_norm = (
                    _compute_grad_norm(comp_params)
                    if _gradients_are_finite(comp_params)
                    else 0.0
                )
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                batch_size, point_count_per_sample, _ = pred.shape
                sample_weight = float(batch_size)
                loss_values = [
                    loss,
                    L_reg_huber,
                    L_aux_stages,
                    L_reg_mse,
                    L_reg_cd,
                    L_phys,
                    L_completion,
                    completion_losses["loss_fine"],
                    completion_losses["loss_partial"],
                    completion_losses["loss_repulsion"],
                ]
                meters[:len(loss_names)] += torch.stack(
                    [value.double() * sample_weight for value in loss_values]
                )
                offset = len(loss_names)
                meters[offset] += reg_grad_norm
                meters[offset + 1] += comp_grad_norm
                source_squared = (
                    batch["src_xyz"].float() - batch["gt_xyz"].float()
                ).square().sum(dim=-1)
                meters[offset + 2] += source_squared.sum().double()

                confidence = out.get("global_match_confidence")
                confidence_count = 0
                if confidence is not None:
                    meters[offset + 3] += confidence.float().sum().double()
                    confidence_count = confidence.numel()
                meters[offset + 4] += batch_size
                meters[offset + 5] += batch_size * point_count_per_sample
                meters[offset + 6] += confidence_count
                meters[offset + 7] += 1
                meters[offset + 8] += int(nonfinite_registration_gradients)

                stage_offset = base_meter_count
                for stage_index, stage_pred in enumerate(pred_stages):
                    stage_squared = (
                        stage_pred.float() - batch["gt_xyz"].float()
                    ).square().sum(dim=-1)
                    meters[stage_offset + stage_index] += stage_squared.sum().double()

        except RuntimeError as e:
            if rank == 0:
                print(f"[Warning] Error in batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            continue

    dist.all_reduce(meters, op=dist.ReduceOp.SUM)
    offset = len(loss_names)
    sample_count = meters[offset + 4].clamp_min(1.0)
    point_count = meters[offset + 5].clamp_min(1.0)
    confidence_count = meters[offset + 6].clamp_min(1.0)
    batch_count = meters[offset + 7].clamp_min(1.0)
    metrics = {
        name: (meters[index] / sample_count).item()
        for index, name in enumerate(loss_names)
    }
    metrics["registration_grad_norm"] = (meters[offset] / batch_count).item()
    metrics["completion_grad_norm"] = (meters[offset + 1] / batch_count).item()
    metrics["source_point_rmse"] = torch.sqrt(
        meters[offset + 2] / point_count
    ).item()
    metrics["global_match_confidence"] = (
        meters[offset + 3] / confidence_count
    ).item()
    metrics["nonfinite_gradient_batches"] = meters[offset + 8].item()
    for stage_index, stage_name in enumerate(stage_names):
        metrics[f"{stage_name}_point_rmse"] = torch.sqrt(
            meters[base_meter_count + stage_index] / point_count
        ).item()
    metrics["final_point_rmse"] = metrics[f"{stage_names[-1]}_point_rmse"]
    metrics["reg_point_rmse"] = metrics["final_point_rmse"]
    return metrics

# def train_one_epoch(model, loader, optimizer, scaler, device, rank,
#                     w_sup, w_part, w_phys, alpha_dcd, phys_k, phys_reg):
#     model.train()
#     dcd = OneSidedDCD(alpha=alpha_dcd).to(device)

#     loss_meter = 0.0
#     mse_meter  = 0.0
#     sup_meter  = 0.0
#     part_meter = 0.0
#     phys_meter = 0.0
#     n_valid_batches = 0

#     # 只在rank 0显示进度条
#     iterator = tqdm(loader, desc="Train", leave=False) if rank == 0 else loader

#     for batch_idx, batch in enumerate(iterator):
#         # 初始化所有 Loss 为 0 (带 device)
#         L_sup = torch.tensor(0.0, device=device)
#         L_part = torch.tensor(0.0, device=device)
#         L_phys = torch.tensor(0.0, device=device)
        
#         # 标记当前 Batch 是否有效
#         batch_is_valid = True
        
#         try:
#             # 1. 数据搬运
#             for k in batch:
#                 batch[k] = batch[k].to(device, non_blocking=True)

#             # 2. 混合精度前向传播
#             with torch.cuda.amp.autocast(enabled=True):
#                 pred = model(
#                     batch["src_xyz"], batch["part_xyz"],
#                     batch["input_ids"], batch["attn_mask"],
#                     E_kPa=batch["E_kPa"],
#                     nu=batch["nu"]
#                 )

#                 # 3. 计算各个 Loss
#                 # (1) 监督损失
#                 L_sup = mse_loss(pred, batch["gt_xyz"])
#                 if torch.isnan(L_sup) or torch.isinf(L_sup):
#                     raise RuntimeError("L_sup is NaN/Inf")

#                 # (2) 局部损失
#                 L_part = dcd(pred, batch["part_xyz"])
#                 if torch.isnan(L_part) or torch.isinf(L_part):
#                     raise RuntimeError("L_part is NaN/Inf")

#                 # # (3) 物理损失 (如果不使用则不需要计算，避免报错)
#                 # if w_phys > 0:
#                 #     try:
#                 #         # 确保传入正确的参数 (注意这里的 neohookean_loss 必须是我们之前修正过的版本)
#                 #         L_phys = neohookean_loss(
#                 #             batch["src_xyz"], 
#                 #             (pred - batch["src_xyz"]), # disp
#                 #             batch["E_kPa"], 
#                 #             batch["nu"],
#                 #             k=phys_k,
#                 #             weight_energy=1.0
#                 #         )
#                 #         if torch.isnan(L_phys) or torch.isinf(L_phys):
#                 #             print(f"[Rank {rank}] Phys Loss is NaN, ignoring this term.")
#                 #             L_phys = 0.0 * pred.sum() # 伪造 0 Loss 保持图连接
#                 #     except Exception as e_phys:
#                 #         # 物理 Loss 计算失败不应该中断整个训练，降级处理
#                 #         print(f"[Rank {rank}] Phys Loss Warning: {e_phys}")
#                 #         L_phys = 0.0 * pred.sum()

#                 # 4. 总 Loss 加权
#                 loss = w_sup * L_sup + w_part * L_part #+ w_phys * L_phys

#         except RuntimeError as e:
#             # 捕获类似 "RuntimeError: cdist..." 或 NaN 错误
#             if rank == 0:
#                 print(f"[Warning] Error in batch {batch_idx} (Rank {rank}): {str(e)}")
            
#             # 🚨 关键修改：遇到错误时，不要 continue！
#             # 构造一个与模型参数有关联的 0 值 Loss
#             # 这样 backward() 产生的梯度全为 0，不会更新模型，但会参与 DDP 通信，防止死锁。
#             # 假设 batch 中有数据，我们用 input 计算一个 dummy graph
#             if 'pred' in locals():
#                 loss = 0.0 * pred.sum()
#             elif 'batch' in locals() and 'src_xyz' in batch:
#                  # 如果连 pred 都没算出来，就用输入过一遍模型 (极少数情况)
#                  dummy_pred = model(batch["src_xyz"], batch["part_xyz"], batch["input_ids"], batch["attn_mask"])
#                  loss = 0.0 * dummy_pred.sum()
#             else:
#                  # 极端情况：数据都没加载进来，此时只能抛出异常或自行处理，DDP 可能会挂
#                  # 但通常 RuntimeError 是发生在 forward 或 loss 阶段，上面的逻辑够用了
#                  loss = torch.tensor(0.0, device=device, requires_grad=True)

#             batch_is_valid = False
#             torch.cuda.empty_cache()

#         # 5. 反向传播 (无论 batch 是否有效，都要跑这一步以维持 DDP 同步)
#         optimizer.zero_grad(set_to_none=True)
#         if scaler is not None:
#             scaler.scale(loss).backward()
            
#             if batch_is_valid: # 只有有效 Batch 才更新参数 (无效 Batch 梯度为 0 也没事，但为了安全不 step)
#                 scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#         else:
#             loss.backward()
#             if batch_is_valid:
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()

#         # 6. 记录日志 (只记录有效 Batch)
#         if batch_is_valid:
#             loss_meter += loss.item()
#             mse_meter  += L_sup.item()
#             sup_meter  += L_sup.item()
#             part_meter += L_part.item()
#             phys_meter += L_phys.item()
#             n_valid_batches += 1

#     # ... 后续统计代码不变 ...
#     if n_valid_batches == 0:
#         return 0.0, 0.0, 0.0, 0.0, 0.0

#     # 同步所有进程的统计数据
#     metrics = torch.tensor([loss_meter, mse_meter, sup_meter, part_meter, phys_meter, n_valid_batches], 
#                            device=device)
#     dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
#     total_batches = metrics[5].item()
#     if total_batches > 0:
#         return tuple((metrics[i] / total_batches).item() for i in range(5))
#     return 0.0, 0.0, 0.0, 0.0, 0.0


def main_worker(rank, world_size, args_dict):
    """每个GPU进程的主函数"""
    
    try:
        # 从字典重建args对象
        class Args:
            pass
        args = Args()
        for k, v in args_dict.items():
            setattr(args, k, v)
        
        # 初始化DDP
        setup_ddp(rank, world_size)
        
        # 设置随机种子
        set_seed(42, rank)  # 使用固定种子42
        
        device = torch.device(f'cuda:{rank}')
        
        if rank == 0:
            print(f"[Info] Using {world_size} GPUs for training")
            print(f"[Info] Device: {device}")

        # 数据集 - 在每个进程中独立创建
        T_dataset = LiverCompletionDataset(args.dataset_root, ["train"], args)
        T_validation = LiverCompletionDataset(args.dataset_root, ["validation"], args)

        # 创建可序列化的 collate_fn
        collate_wrapper = CollateWrapper()

        # 使用DistributedSampler
        train_sampler = DistributedSampler(
            T_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=42  # 固定种子
        )
        
        eval_sampler = DistributedSampler(
            T_validation,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        train_loader = DataLoader(
            T_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            num_workers=0,  # 强制设为0，避免worker进程被kill
            pin_memory=False,  # 改为False减少内存使用
            collate_fn=collate_wrapper
        )

        eval_loader = DataLoader(
            T_validation, 
            batch_size=args.batch_size, 
            sampler=eval_sampler,
            num_workers=0,  # 强制设为0
            pin_memory=False,  # 改为False减少内存使用
            collate_fn=collate_wrapper
        )

        if rank == 0:
            print(f"[Info] Creating model...")

        # 创建模型 - 确保所有进程使用相同的初始化
        torch.manual_seed(42)  # 确保模型初始化一致
        model = TextConditionedGIRNet(
            bert_model_name=args.bert_model_name,
            bert_local_files_only=args.bert_local_files_only,
            train_bert=args.train_bert,
            use_text=args.use_text,
            load_GIRNet_checkpoint=args.GIRNet_checkpoint,
            strict_GIRNet_checkpoint=args.strict_GIRNet_checkpoint,
            completion_checkpoint=args.completion_checkpoint,
            GIRNet_arch=args.GIRNet_arch,
            global_match_level=args.global_match_level,
            global_match_temperature=args.global_match_temperature,
            global_match_dim=args.global_match_dim,
            global_spatial_sigma=args.global_spatial_sigma,
            max_coarse_flow_normalized=args.max_coarse_flow_normalized,
            num_refinement_steps=args.num_refinement_steps,
            refinement_k=args.refinement_k,
            initialize_from_legacy_GIRNet=args.initialize_from_legacy_GIRNet,
            debug_refinement=args.debug_refinement,
        ).to(device)
        if model.completion is None:
            raise RuntimeError("completion_checkpoint is required for training")

        # --- train_stage handling ---
        if args.train_stage == "registration":
            # Freeze all completion parameters
            for parameter in model.completion.parameters():
                parameter.requires_grad_(False)
            model.completion.eval()
            # Force completion loss weight to 0
            args.w_completion = 0.0
        else:  # joint
            for parameter in model.completion.parameters():
                parameter.requires_grad_(True)

        completion_criterion = build_completion_criterion(
            model.completion_config,
            device,
        )
        
        # 同步barrier确保所有进程都创建了模型
        dist.barrier()
        
        if rank == 0:
            print(f"[Info] Wrapping model with DDP...")
            print(f"[Debug] Checking model parameters:")
            for idx, (name, param) in enumerate(model.named_parameters()):
                print(f"  [{idx}] {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        
        # 包装为DDP模型
        model = DDP(
            model, 
            device_ids=[rank], 
            output_device=rank,
            find_unused_parameters=True,  # 必须设为True，因为有未使用的参数
            broadcast_buffers=False  # 禁用buffer广播以避免空tensor错误
        )

        # 只在rank 0初始化wandb和打印信息
        if rank == 0:
            if args.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=args.wandb_project,
                        name=args.wandb_run_name,
                        config=args_dict,
                        settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
                    )
                    wandb.watch(model, log="all", log_freq=100)
                except Exception as wandb_error:
                    print(f"[Warning] wandb init failed, continuing without wandb: {wandb_error}")
                    args.use_wandb = False

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            completion_total = sum(p.numel() for p in model.module.completion.parameters())
            completion_trainable = sum(
                p.numel()
                for p in model.module.completion.parameters()
                if p.requires_grad
            )
            registration_trainable = trainable_params - completion_trainable
            print(f"[Info] 模型已加载")
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            print(f"[Info] completion trainable parameters: {completion_trainable:,}")
            print(f"[Info] registration trainable parameters: {registration_trainable:,}")
            print(f"[Info] train_stage: {args.train_stage}")
            print(f"[Info] registration_target_mode: {args.registration_target_mode}")
            print(f"[Info] GIRNet_arch: {args.GIRNet_arch}")
            if args.GIRNet_arch == "full2full_v1":
                print(
                    "[Info] global match points: "
                    f"{model.module.backbone.global_match_points}"
                )
                print(f"[Info] global_match_level: {args.global_match_level}")
                print(
                    "[Info] global_match_temperature: "
                    f"{args.global_match_temperature}"
                )
                print(f"[Info] global_match_dim: {args.global_match_dim}")
                print(f"[Info] global_spatial_sigma: {args.global_spatial_sigma}")
                print(
                    "[Info] max_coarse_flow_normalized: "
                    f"{args.max_coarse_flow_normalized}"
                )
                print(
                    f"[Info] num_refinement_steps: {args.num_refinement_steps}"
                )
                print(f"[Info] refinement_k: {args.refinement_k}")
            print(f"[Info] completion frozen: {args.train_stage == 'registration'}")
            print(f"[Info] registration lr: {args.lr}")
            print(f"[Info] completion lr: {args.completion_lr}")
            print(f"[Info] w_reg_huber: {args.w_reg_huber}")
            print(f"[Info] w_aux_stages: {args.w_aux_stages}")
            print(f"[Info] aux_stage_weights: {args.aux_stage_weights}")
            print(f"[Info] w_reg_cd: {args.w_reg_cd}")
            print(f"[Info] w_phys: {args.w_phys}")
            print(f"[Info] w_completion: {args.w_completion}")
            print(f"[Info] use_text: {args.use_text}")

        completion_parameter_ids = {
            id(parameter) for parameter in model.module.completion.parameters()
        }
        if args.train_stage == "registration":
            # Only train GIRNet backbone and FiLM (if text enabled)
            registration_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in completion_parameter_ids and parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                registration_parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:  # joint
            completion_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) in completion_parameter_ids and parameter.requires_grad
            ]
            other_parameters = [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in completion_parameter_ids and parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_parameters, "lr": args.lr},
                    {"params": completion_parameters, "lr": args.completion_lr},
                ],
                weight_decay=args.weight_decay,
            )
        scaler = torch.cuda.amp.GradScaler()

        start_epoch = 0
        best_eval_point_rmse = float('inf')

        # 断点续训
        if args.resume and os.path.isfile(args.resume):
            if rank == 0:
                print(f"[Info] Loading checkpoint from {args.resume}")
            
            # 所有进程都需要加载checkpoint以保持同步
            map_location = f'cuda:{rank}'
            ckpt = torch.load(args.resume, map_location=map_location)
            checkpoint_arch = ckpt.get("GIRNet_arch")
            if checkpoint_arch is None:
                checkpoint_arch = ckpt.get("config", {}).get(
                    "GIRNet_arch", "legacy"
                )
            if checkpoint_arch != args.GIRNet_arch:
                raise RuntimeError(
                    "Cannot resume checkpoint with a different GIRNet "
                    f"architecture: checkpoint={checkpoint_arch}, "
                    f"requested={args.GIRNet_arch}"
                )
            
            model.module.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt and scaler is not None:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_eval_point_rmse = ckpt.get('best_eval_point_rmse', best_eval_point_rmse)

            if rank == 0:
                print(f"[Info] Resumed from epoch {start_epoch}, best_point_rmse={best_eval_point_rmse:.6f}")
            
            # 同步所有进程
            dist.barrier()

        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)

        # 同步确保目录创建完成
        dist.barrier()

        if rank == 0:
            print("[Info] Starting training...")

        for epoch in range(start_epoch, args.epochs):
            # 设置epoch以确保每个epoch的数据shuffle不同
            train_sampler.set_epoch(epoch)
            T_dataset.set_epoch(epoch)

            if rank == 0:
                print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                completion_criterion,
                device,
                rank,
                args,
                completion_parameter_ids,
            )

            if rank == 0:
                train_stage_text = " ".join(
                    f"{name}={train_metrics[f'{name}_point_rmse']:.4f}mm"
                    for name in stage_metric_names(
                        args.GIRNet_arch, args.num_refinement_steps
                    )
                )
                print(
                    f"[Train] total_loss={train_metrics['total_loss']:.6f} "
                    f"huber={train_metrics['reg_huber']:.6f} "
                    f"aux={train_metrics['aux_stage_loss']:.6f} "
                    f"mse={train_metrics['reg_mse']:.6f} "
                    f"cd={train_metrics['reg_cd']:.6f} "
                    f"{train_stage_text} "
                    f"final_point_rmse={train_metrics['final_point_rmse']:.4f}mm "
                    f"source_rmse={train_metrics['source_point_rmse']:.4f}mm "
                    f"global_conf={train_metrics['global_match_confidence']:.6f} "
                    f"phys={train_metrics['physics']:.6f} "
                    f"comp={train_metrics['completion']:.6f} "
                    f"reg_grad={train_metrics['registration_grad_norm']:.4f} "
                    f"comp_grad={train_metrics['completion_grad_norm']:.4f} "
                    f"nonfinite_grad_batches="
                    f"{int(train_metrics['nonfinite_gradient_batches'])}"
                )
                if train_metrics['registration_grad_norm'] < 1e-8:
                    print("[Warning] registration_grad_norm near zero — GIRNet may not be learning!")

            # 评估
            eval_metrics = evaluate(model, eval_loader, device, args)

            if rank == 0:
                eval_stage_text = " ".join(
                    f"{name}={eval_metrics[f'eval_{name}_point_rmse']:.4f}mm"
                    for name in stage_metric_names(
                        args.GIRNet_arch, args.num_refinement_steps
                    )
                )
                print(
                    f"[Eval ] huber={eval_metrics['eval_reg_huber']:.6f} "
                    f"aux={eval_metrics['eval_aux_stage_loss']:.6f} "
                    f"mse={eval_metrics['eval_reg_mse']:.6f} "
                    f"cd={eval_metrics['eval_reg_cd']:.6f} "
                    f"{eval_stage_text} "
                    f"final_point_rmse={eval_metrics['eval_final_point_rmse']:.4f}mm "
                    f"source_rmse={eval_metrics['eval_source_point_rmse']:.4f}mm "
                    f"global_conf={eval_metrics['eval_global_match_confidence']:.6f} "
                    f"comp_cd={eval_metrics['eval_completion_cd']:.6f}"
                )

                # 保存checkpoint (只在rank 0保存)
                save_ckpt({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_eval_point_rmse': best_eval_point_rmse,
                    'GIRNet_arch': args.GIRNet_arch,
                    'config': args_dict,
                }, Path(args.save_dir) / 'last.pth')

                if eval_metrics['eval_reg_point_rmse'] < best_eval_point_rmse:
                    best_eval_point_rmse = eval_metrics['eval_reg_point_rmse']
                    save_ckpt(
                        {
                            'epoch': epoch,
                            'model': model.module.state_dict(),
                            'best_eval_point_rmse': best_eval_point_rmse,
                            'GIRNet_arch': args.GIRNet_arch,
                            'config': args_dict,
                        },
                        Path(args.save_dir) / 'best.pth',
                    )
                    print(f"[Info] New best point_rmse: {best_eval_point_rmse:.6f}mm (saved best.pth)")

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **{
                            f"eval/{key.removeprefix('eval_')}": value
                            for key, value in eval_metrics.items()
                        },
                    }, step=epoch)

                train_data = {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **eval_metrics,
                }
                save_to_json(args.save_dir + "/log.json", train_data)

            # 同步所有进程
            dist.barrier()

        if rank == 0:
            print("\nTraining completed!")
    
    except Exception as e:
        if rank == 0:
            print(f"[Error] Exception in rank {rank}: {str(e)}")
        raise e
    finally:
        cleanup_ddp()

def main():
    """主入口函数"""
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default="/home/ma_sx/Project/Dataset/MedShapeNet-Liver")
    ap.add_argument('--data_overlap', type=float, default=0.8)
    ap.add_argument('--max_train_samples', type=int, default=-1)
    ap.add_argument('--max_val_samples', type=int, default=100)

    ap.add_argument('--save_dir', type=str, default=f'./logs/exp_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=2)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3' #选择显卡
    ap.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--completion_lr', type=float, default=0.0)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--num_workers', type=int, default=0)  # 改为0避免内存问题
    ap.add_argument('--use_text', default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument('--bert_model_name', type=str, default='/home/ma_sx/Project/Liver2/bert-base-uncased')
    ap.add_argument('--bert_local_files_only', default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument('--train_bert', action='store_true')
    ap.add_argument('--GIRNet_checkpoint', type=str, default='')
    ap.add_argument('--strict_GIRNet_checkpoint', default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument('--GIRNet_arch', type=str, default='full2full_v1',
                    choices=['legacy', 'full2full_v1'])
    ap.add_argument('--global_match_level', type=int, default=2)
    ap.add_argument('--global_match_temperature', type=float, default=0.1)
    ap.add_argument('--global_match_dim', type=int, default=64)
    ap.add_argument('--global_spatial_sigma', type=float, default=0.2)
    ap.add_argument('--max_coarse_flow_normalized', type=float, default=0.25)
    ap.add_argument('--num_refinement_steps', type=int, default=3)
    ap.add_argument('--refinement_k', type=int, default=35)
    ap.add_argument('--initialize_from_legacy_GIRNet', default=False,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument('--debug_refinement', default=False,
                    action=argparse.BooleanOptionalAction)
    ap.add_argument(
        '--completion_checkpoint',
        type=str,
        default='/home/ma_sx/Project/Liver2/completion/logs/full_aug_20260805_013524/best.pth',
    )
    
    # 训练模式
    ap.add_argument('--train_stage', type=str, default='registration',
                    choices=['registration', 'joint'])
    ap.add_argument('--registration_target_mode', type=str, default='gt',
                    choices=['completed', 'gt'])

    # 损失权重
    ap.add_argument('--w_reg_huber', type=float, default=1.0)
    ap.add_argument('--huber_beta_mm', type=float, default=5.0)
    ap.add_argument('--w_aux_stages', type=float, default=0.5)
    ap.add_argument('--aux_stage_weights', type=parse_float_list,
                    default=parse_float_list('0.1,0.2,0.5'))
    ap.add_argument('--w_reg_cd', type=float, default=0.05)
    ap.add_argument('--w_reg_mse', '--w_sup', dest='w_reg_mse', type=float, default=0.0)
    ap.add_argument('--w_completion', type=float, default=0.0)
    ap.add_argument('--w_phys', type=float, default=0.0)
    ap.add_argument('--phys_k', type=int, default=24)
    ap.add_argument('--phys_reg', type=float, default=1e-4)
    
    # 训练细节
    ap.add_argument('--eval_split_ratio', type=float, default=0.1)
    ap.add_argument('--resume', type=str, default='')
    
    # wandb 相关
    ap.add_argument('--use_wandb', default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument('--wandb_project', type=str, default='msn_completion_biomech')
    ap.add_argument('--wandb_run_name', type=str, default='exp1')
    ap.add_argument('--wandb_init_timeout', type=int, default=120)
    
    args = ap.parse_args()
    if args.global_match_level not in (0, 1, 2, 3, 4):
        ap.error('--global_match_level must be one of 0,1,2,3,4')
    if args.global_match_temperature <= 0:
        ap.error('--global_match_temperature must be positive')
    if args.global_match_dim < 1:
        ap.error('--global_match_dim must be positive')
    if args.global_spatial_sigma <= 0:
        ap.error('--global_spatial_sigma must be positive')
    if args.max_coarse_flow_normalized <= 0:
        ap.error('--max_coarse_flow_normalized must be positive')
    if args.num_refinement_steps < 1:
        ap.error('--num_refinement_steps must be at least 1')
    if args.refinement_k < 1:
        ap.error('--refinement_k must be at least 1')
    if (
        args.GIRNet_arch == 'full2full_v1'
        and len(args.aux_stage_weights) != args.num_refinement_steps
    ):
        ap.error(
            '--aux_stage_weights must contain exactly one value for coarse '
            'and every non-final refinement stage; expected '
            f'{args.num_refinement_steps}, got {len(args.aux_stage_weights)}'
        )
    if not os.path.isfile(args.completion_checkpoint):
        ap.error(
            f"completion checkpoint not found: {args.completion_checkpoint}"
        )
    if (
        args.GIRNet_arch == 'legacy'
        or args.initialize_from_legacy_GIRNet
    ) and not os.path.isfile(args.GIRNet_checkpoint):
        ap.error(f"GIRNet checkpoint not found: {args.GIRNet_checkpoint}")
    
    world_size = args.world_size
    
    # 检查GPU数量
    if world_size > torch.cuda.device_count():
        print(f"[Warning] Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
        world_size = torch.cuda.device_count()
    
    if world_size < 1:
        raise RuntimeError("No GPUs available for training")
    
    print(f"[Info] Starting distributed training with {world_size} GPUs")
    
    # 将args转换为字典，避免序列化问题
    args_dict = vars(args)
    args_dict['world_size'] = world_size
    
    # 使用spawn启动多进程
    mp.spawn(
        main_worker,
        args=(world_size, args_dict),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    # 设置multiprocessing的启动方法
    mp.set_start_method('spawn', force=True)
    main()
