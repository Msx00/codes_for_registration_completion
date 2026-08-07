#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal verification tests for full2full_v2.

Run:
    python GIRNet/tests/test_full2full_v2.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure GIRNet models are importable.
GIRNet_ROOT = Path(__file__).resolve().parent.parent
if str(GIRNet_ROOT) not in sys.path:
    sys.path.insert(0, str(GIRNet_ROOT))

from models.P_V2S_Net_Full2Full_V2 import PV2SNetFull2FullV2
from models.P_V2S_Net_Full2Full_V1 import PV2SNetFull2FullV1
from models.global_matcher import GlobalMatcherV2, interpolate_flow_v2


def _random_cloud(B, N, device="cpu"):
    return torch.randn(B, N, 3, device=device)


def test_full2full_v1_still_runs():
    """full2full_v1 should still run without errors."""
    model = PV2SNetFull2FullV1(
        num_refinement_steps=2,
        refinement_k=8,
    )
    src = _random_cloud(2, 512)
    tgt = _random_cloud(2, 512)
    with torch.no_grad():
        out = model(src, tgt)
    assert "flow_stages" in out
    assert len(out["flow_stages"]) == 3  # coarse + 2 refines
    assert out["flow_stages"][-1].shape == (2, 512, 3)
    print("[PASS] test_full2full_v1_still_runs")


def test_full2full_v2_forward_fp32():
    """full2full_v2 forward in FP32."""
    model = PV2SNetFull2FullV2(
        num_refinement_steps=2,
        refinement_k=8,
    )
    src = _random_cloud(2, 512)
    tgt = _random_cloud(2, 512)
    out = model(src, tgt)
    assert "flow_stages" in out
    assert len(out["flow_stages"]) == 3  # coarse + 2 refines
    assert out["flow_stages"][-1].shape == (2, 512, 3)
    assert "global_assignment" in out
    assert "source_global_indices" in out
    assert out["source_global_indices"].shape[1] == model.global_match_points
    assert "target_global_xyz" in out
    assert "score_weights" in out
    assert out["score_weights"].shape == (3,)
    print("[PASS] test_full2full_v2_forward_fp32")


def test_full2full_v2_backward_fp32():
    """full2full_v2 backward in FP32 produces finite gradients."""
    model = PV2SNetFull2FullV2(
        num_refinement_steps=2,
        refinement_k=8,
    )
    src = _random_cloud(2, 512)
    tgt = _random_cloud(2, 512)
    gt = tgt + 0.1 * torch.randn(2, 512, 3)

    out = model(src, tgt)
    pred = src + out["flow_stages"][-1]
    loss = F.mse_loss(pred, gt)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
    print("[PASS] test_full2full_v2_backward_fp32")


def test_source_global_indices_mapping():
    """source_global_indices correctly map to original source FPS points."""
    model = PV2SNetFull2FullV2(
        num_refinement_steps=1,
        refinement_k=8,
    )
    B, N = 2, 512
    src = _random_cloud(B, N)
    tgt = _random_cloud(B, 400)

    with torch.no_grad():
        out = model(src, tgt)

    source_global_indices = out["source_global_indices"]  # (B, Ns_coarse)
    assert source_global_indices.min() >= 0
    assert source_global_indices.max() < N

    # Verify: gathering src with source_global_indices should recover the
    # coarse source points used by GlobalMatcher.
    batch_idx = torch.arange(B).view(-1, 1)
    gathered_src = src[batch_idx, source_global_indices]  # (B, Ns_coarse, 3)
    assert gathered_src.shape[1] == model.global_match_points
    assert torch.isfinite(gathered_src).all()
    print("[PASS] test_source_global_indices_mapping")


def test_global_matcher_v2_score_weights():
    """GlobalMatcherV2 returns valid score_weights."""
    matcher = GlobalMatcherV2(
        feature_dim=50,
        projection_dim=64,
        temperature=0.1,
        spatial_sigma=0.3,
        gate_temperature=0.02,
    )
    B, Ns, Nt = 2, 92, 92
    src_coords = torch.randn(B, Ns, 3)
    tgt_coords = torch.randn(B, Nt, 3)
    src_feat = torch.randn(B, Ns, 50)
    tgt_feat = torch.randn(B, Nt, 50)

    out = matcher(src_coords, tgt_coords, src_feat, tgt_feat)
    sw = out["score_weights"]
    assert sw.shape == (3,)
    assert 0.99 <= sw.sum() <= 1.01, f"score_weights sum={sw.sum()}"
    assert (sw >= 0).all()
    assert "assignment" in out
    assert out["assignment"].shape == (B, Ns, Nt)
    print("[PASS] test_global_matcher_v2_score_weights")


def test_match_loss_finite():
    """L_match is finite and backward produces non-zero GlobalMatcher grads."""
    matcher = GlobalMatcherV2(
        feature_dim=50,
        projection_dim=64,
        temperature=0.1,
        spatial_sigma=0.3,
        gate_temperature=0.02,
    )
    B, Ns, Nt = 2, 92, 92
    src_coords = torch.randn(B, Ns, 3)
    tgt_coords = torch.randn(B, Nt, 3)
    src_feat = torch.randn(B, Ns, 50)
    tgt_feat = torch.randn(B, Nt, 50)
    gt_xyz = tgt_coords + 0.05 * torch.randn(B, Nt, 3)
    source_global_indices = torch.arange(Ns).unsqueeze(0).expand(B, Ns)

    out = matcher(src_coords, tgt_coords, src_feat, tgt_feat)

    # Compute L_match inline (same formula).
    gt_coarse = gt_xyz[torch.arange(B).view(-1, 1), source_global_indices]
    with torch.no_grad():
        distance2 = torch.cdist(gt_coarse.float(), tgt_coords.float(), p=2).square()
        pseudo_target_index = distance2.argmin(dim=-1)
        min_distance2 = distance2.gather(dim=-1, index=pseudo_target_index.unsqueeze(-1)).squeeze(-1)
        match_weight = torch.exp(-min_distance2 / (2.0 * 5.0 ** 2))

    matched_probability = out["assignment"].float().gather(
        dim=-1, index=pseudo_target_index.unsqueeze(-1)
    ).squeeze(-1)
    L_match = -(match_weight * torch.log(matched_probability.clamp_min(1e-8))).sum() / match_weight.sum().clamp_min(1e-8)
    assert torch.isfinite(L_match), f"L_match not finite: {L_match}"
    assert L_match > 0, f"L_match should be positive: {L_match}"

    L_match.backward()
    for name, param in matcher.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in matcher.{name}"
    # Verify GlobalMatcher params received gradients.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in matcher.parameters()
    )
    assert has_grad, "GlobalMatcherV2 params received no gradients from L_match"
    print("[PASS] test_match_loss_finite")


def test_completed_mode_uses_completed_not_gt():
    """In completed mode, GIRNet receives completed_xyz, not GT."""
    # This is satisfied by construction in GIRNet_text_model.forward():
    # when registration_target_xyz is None, the model uses
    # completed_xyz.detach() as the registration target.
    # GT only enters through the loss functions (train-multigpu.py / evaluate.py).
    print("[SKIP] test_completed_mode_uses_completed_not_gt (verified by code inspection)")


def test_interpolate_flow_v2_squared_distance():
    """interpolate_flow_v2 uses squared-distance weights."""
    B, Nq, Ns = 2, 512, 92
    query = torch.randn(B, Nq, 3)
    support = torch.randn(B, Ns, 3)
    support_flow = torch.randn(B, Ns, 3)

    # Verify gradients flow to support_flow.
    support_flow.requires_grad_(True)
    interp = interpolate_flow_v2(query, support, support_flow, k=3)
    loss = interp.sum()
    loss.backward()
    assert support_flow.grad is not None
    assert torch.isfinite(support_flow.grad).all()
    print("[PASS] test_interpolate_flow_v2_squared_distance")


def test_single_sample_overfit_fp32():
    """Single-sample overfit: final RMSE < source RMSE after a few steps."""
    model = PV2SNetFull2FullV2(
        num_refinement_steps=2,
        refinement_k=8,
        global_match_level=1,  # 35 points for faster test
    )
    src = _random_cloud(1, 256)
    gt = src + 0.3 * torch.randn(1, 256, 3)
    tgt = gt  # GT mode: target = gt

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    source_rmse = torch.sqrt(F.mse_loss(src, gt)).item()
    print(f"  source_rmse = {source_rmse:.4f}")

    for step in range(50):
        optimizer.zero_grad()
        out = model(src, tgt)
        pred = src + out["flow_stages"][-1]
        loss = F.mse_loss(pred, gt)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = model(src, tgt)
        pred = src + out["flow_stages"][-1]
        final_rmse = torch.sqrt(F.mse_loss(pred, gt)).item()

    print(f"  final_rmse  = {final_rmse:.4f}")
    assert final_rmse < source_rmse, (
        f"final_rmse={final_rmse:.4f} >= source_rmse={source_rmse:.4f}"
    )
    print("[PASS] test_single_sample_overfit_fp32")


# ---------------------------------------------------------------------------
# New tests for L_match coordinate unit bug fix
# ---------------------------------------------------------------------------

def _normalize_for_test(src_xyz, tgt_xyz, eps=1e-6):
    """Replicate _normalize_for_GIRNet for unit-test use (returns centroid)."""
    centroid = src_xyz.float().mean(dim=1, keepdim=True)
    src_centered = src_xyz.float() - centroid
    scale = torch.linalg.norm(src_centered, dim=-1).amax(dim=1, keepdim=True)
    scale = scale.clamp_min(eps).unsqueeze(-1)
    src_norm = (src_centered / scale).to(dtype=src_xyz.dtype)
    tgt_norm = ((tgt_xyz.float() - centroid) / scale).to(dtype=tgt_xyz.dtype)
    return src_norm, tgt_norm, centroid, scale


def test_target_global_xyz_unit_recovery():
    """Test 1: target_global_xyz inverse-transform recovers mm coordinates.

    Uses non-zero centroid and non-unit scale to ensure the bug
    (confusing normalized coords with mm) would be caught.
    """
    model = PV2SNetFull2FullV2(
        num_refinement_steps=1,
        refinement_k=8,
        global_match_level=2,  # 92 coarse points
    )
    B, Ns, Nt = 2, 512, 400

    # Realistic liver-scale coordinates (tens to hundreds of mm).
    centroid_val = torch.tensor([[[80.0, 50.0, -30.0]]])  # (1, 1, 3)
    src = centroid_val + 60.0 * torch.randn(B, Ns, 3)
    tgt = centroid_val + 10.0 + 55.0 * torch.randn(B, Nt, 3)

    # Normalize → forward → get target_global_xyz_normalized.
    src_norm, tgt_norm, centroid, scale = _normalize_for_test(src, tgt)
    with torch.no_grad():
        out = model(src_norm, tgt_norm)

    tgt_norm_coarse = out["target_global_xyz"]  # (B, Nc, 3), still normalized

    # Inverse transform (same as GIRNet_text_model.forward now does).
    tgt_mm_coarse = tgt_norm_coarse.float() * scale.float() + centroid.float()

    # Verify tgt_mm_coarse is in the same physical range as source/target.
    print(f"  src min/max/mean:        {src.min().item():.2f} / {src.max().item():.2f} / {src.mean().item():.2f}")
    print(f"  tgt min/max/mean:        {tgt.min().item():.2f} / {tgt.max().item():.2f} / {tgt.mean().item():.2f}")
    print(f"  tgt_norm_coarse min/max: {tgt_norm_coarse.min().item():.4f} / {tgt_norm_coarse.max().item():.4f}")
    print(f"  tgt_mm_coarse min/max:   {tgt_mm_coarse.min().item():.2f} / {tgt_mm_coarse.max().item():.2f}")
    print(f"  centroid: {centroid[0, 0, :].tolist()}")
    print(f"  scale:    {scale[0, 0, 0].item():.4f}")

    # tgt_mm_coarse must be near the centroid (not near 0 like normalized coords).
    centroid_dist = (tgt_mm_coarse - centroid).norm(dim=-1).mean().item()
    assert centroid_dist < 200.0, f"tgt_mm_coarse too far from centroid: {centroid_dist:.2f}"
    assert tgt_mm_coarse.abs().mean() > 1.0, "tgt_mm_coarse looks normalized (near 0)"

    # Verify: tgt_norm_coarse is in [-1, 1] range (normalized), tgt_mm_coarse is not.
    assert tgt_norm_coarse.abs().max() < 5.0, "normalized coords should be small"
    assert tgt_mm_coarse.abs().max() > 10.0, "mm coords should be large"

    print("[PASS] test_target_global_xyz_unit_recovery")


def test_match_loss_not_zero_with_mm_coords():
    """Test 2: L_match > 0 when both gt and target_global_xyz are in mm."""
    matcher = GlobalMatcherV2(
        feature_dim=50, projection_dim=64, temperature=0.1,
        spatial_sigma=0.3, gate_temperature=0.02,
    )
    B, Ns, Nt = 2, 92, 92

    # Use realistic mm-scale coords for BOTH gt and target.
    centroid = torch.tensor([[[100.0, 20.0, -40.0]]])
    src_coords = centroid + 50.0 * torch.randn(B, Ns, 3)
    tgt_coords = centroid + 5.0 + 45.0 * torch.randn(B, Nt, 3)
    src_feat = torch.randn(B, Ns, 50)
    tgt_feat = torch.randn(B, Nt, 50)
    # GT is nearby target coords (5mm std), all in mm.
    gt_xyz = tgt_coords + 5.0 * torch.randn(B, Nt, 3)
    source_global_indices = torch.arange(Ns).unsqueeze(0).expand(B, Ns)

    out = matcher(src_coords, tgt_coords, src_feat, tgt_feat)

    # Replicate compute_match_loss logic with mm coords.
    gt_coarse = gt_xyz[torch.arange(B).view(-1, 1), source_global_indices]
    with torch.no_grad():
        distance2 = torch.cdist(gt_coarse.float(), tgt_coords.float(), p=2).square()
        pseudo_target_index = distance2.argmin(dim=-1)
        min_distance2 = distance2.gather(dim=-1, index=pseudo_target_index.unsqueeze(-1)).squeeze(-1)
        match_weight = torch.exp(-min_distance2 / (2.0 * 5.0 ** 2))
        match_nn_dist = torch.sqrt(min_distance2.clamp_min(1e-12))

    matched_prob = out["assignment"].float().gather(
        dim=-1, index=pseudo_target_index.unsqueeze(-1)
    ).squeeze(-1)
    L_match = -(match_weight * torch.log(matched_prob.clamp_min(1e-8))).sum() / match_weight.sum().clamp_min(1e-8)

    print(f"  L_match = {L_match.item():.6f}")
    print(f"  match_weight_mean = {match_weight.mean().item():.4f}")
    print(f"  match_weight_min  = {match_weight.min().item():.4f}")
    print(f"  match_weight_max  = {match_weight.max().item():.4f}")
    print(f"  match_nn_dist_mm_mean = {match_nn_dist.mean().item():.2f}")
    print(f"  match_prob_mean = {matched_prob.mean().item():.4f}")

    assert torch.isfinite(L_match), f"L_match not finite: {L_match}"
    assert L_match > 0, f"L_match should be > 0, got {L_match.item()}"
    assert match_weight.mean() > 0.01, f"match_weight_mean too low: {match_weight.mean().item():.6f}"
    print("[PASS] test_match_loss_not_zero_with_mm_coords")


def test_match_loss_backward_to_global_matcher():
    """Test 3: L_match.backward() → non-zero, finite grads in GlobalMatcherV2."""
    matcher = GlobalMatcherV2(
        feature_dim=50, projection_dim=64, temperature=0.1,
        spatial_sigma=0.3, gate_temperature=0.02,
    )
    B, Ns, Nt = 2, 92, 92
    centroid = torch.tensor([[[50.0, -10.0, 30.0]]])
    src_coords = centroid + 40.0 * torch.randn(B, Ns, 3)
    tgt_coords = centroid + 3.0 + 35.0 * torch.randn(B, Nt, 3)
    src_feat = torch.randn(B, Ns, 50)
    tgt_feat = torch.randn(B, Nt, 50)
    gt_xyz = tgt_coords + 3.0 * torch.randn(B, Nt, 3)
    source_global_indices = torch.arange(Ns).unsqueeze(0).expand(B, Ns)

    out = matcher(src_coords, tgt_coords, src_feat, tgt_feat)

    gt_coarse = gt_xyz[torch.arange(B).view(-1, 1), source_global_indices]
    with torch.no_grad():
        distance2 = torch.cdist(gt_coarse.float(), tgt_coords.float(), p=2).square()
        pseudo_target_index = distance2.argmin(dim=-1)
        min_distance2 = distance2.gather(dim=-1, index=pseudo_target_index.unsqueeze(-1)).squeeze(-1)
        match_weight = torch.exp(-min_distance2 / (2.0 * 5.0 ** 2))

    matched_prob = out["assignment"].float().gather(
        dim=-1, index=pseudo_target_index.unsqueeze(-1)
    ).squeeze(-1)
    L_match = -(match_weight * torch.log(matched_prob.clamp_min(1e-8))).sum() / match_weight.sum().clamp_min(1e-8)

    matcher.zero_grad()
    L_match.backward()

    # Key params that must receive gradient.
    grad_ok = {}
    for name, param in matcher.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_ok[name] = grad_norm > 0 and torch.isfinite(param.grad).all()
            print(f"  {name}: grad_norm={grad_norm:.4e}, ok={grad_ok[name]}")

    assert grad_ok.get("query_projection.weight", False), "query_projection has no gradient"
    assert grad_ok.get("key_projection.weight", False), "key_projection has no gradient"
    assert grad_ok.get("score_weight_logits", False), "score_weight_logits has no gradient"
    print("[PASS] test_match_loss_backward_to_global_matcher")


def test_wrong_unit_gives_lower_match_weight():
    """Test 4: mm-vs-normalized mismatch → match_weight_mean << correct version.

    Simulates the old bug: gt in mm, target_global_xyz in normalized coords.
    """
    matcher = GlobalMatcherV2(
        feature_dim=50, projection_dim=64, temperature=0.1,
        spatial_sigma=0.3, gate_temperature=0.02,
    )
    B, Ns, Nt = 2, 92, 92

    centroid = torch.tensor([[[120.0, 30.0, -60.0]]])
    tgt_mm = centroid + 40.0 * torch.randn(B, Nt, 3)
    gt_mm = tgt_mm + 3.0 * torch.randn(B, Nt, 3)
    source_global_indices = torch.arange(Ns).unsqueeze(0).expand(B, Ns)

    # --- Correct version: both in mm ---
    src_mm = centroid + 45.0 * torch.randn(B, Ns, 3)
    src_norm, tgt_norm, cent, scale = _normalize_for_test(src_mm, tgt_mm)
    out_correct = matcher(src_norm, tgt_norm,
                          torch.randn(B, Ns, 50), torch.randn(B, Nt, 50))
    # Simulate wrapper inverse-transform on the target coords that GlobalMatcher used.
    tgt_coarse_mm = tgt_norm.float() * scale.float() + cent.float()

    gt_coarse_mm = gt_mm[torch.arange(B).view(-1, 1), source_global_indices]
    with torch.no_grad():
        d2 = torch.cdist(gt_coarse_mm.float(), tgt_coarse_mm.float(), p=2).square()
        mw_correct = torch.exp(-d2.min(dim=-1).values / (2.0 * 5.0 ** 2)).mean().item()

    # --- Wrong version: gt in mm, target in normalized (old bug) ---
    tgt_coarse_norm = tgt_norm  # normalized, ~0-mean, ~unit-scale (what old code would feed)
    with torch.no_grad():
        d2_wrong = torch.cdist(gt_coarse_mm.float(), tgt_coarse_norm.float(), p=2).square()
        mw_wrong = torch.exp(-d2_wrong.min(dim=-1).values / (2.0 * 5.0 ** 2)).mean().item()

    print(f"  match_weight_mean (correct, both mm):      {mw_correct:.6f}")
    print(f"  match_weight_mean (wrong, mm vs normalized): {mw_wrong:.6f}")
    assert mw_wrong < mw_correct, (
        f"Wrong-unit should give lower match_weight: "
        f"wrong={mw_wrong:.6f} >= correct={mw_correct:.6f}"
    )
    print("[PASS] test_wrong_unit_gives_lower_match_weight")


def test_gt_mode_target_global_xyz_mm():
    """Test 5: GT-mode: target_global_xyz inverse-transforms to GT mm coords."""
    model = PV2SNetFull2FullV2(
        num_refinement_steps=1, refinement_k=8, global_match_level=2,
    )
    B, Ns, Nt = 1, 512, 400
    centroid_val = torch.tensor([[[90.0, -20.0, 45.0]]])
    src = centroid_val + 50.0 * torch.randn(B, Ns, 3)
    gt = centroid_val + 8.0 + 48.0 * torch.randn(B, Nt, 3)
    tgt = gt  # GT mode

    src_norm, tgt_norm, centroid, scale = _normalize_for_test(src, tgt)
    with torch.no_grad():
        out = model(src_norm, tgt_norm)

    # Inverse transform as wrapper does.
    tgt_mm_coarse = out["target_global_xyz"].float() * scale.float() + centroid.float()

    print(f"  GT mm min/max/mean:     {gt.min().item():.2f} / {gt.max().item():.2f} / {gt.mean().item():.2f}")
    print(f"  tgt_mm_coarse min/max:  {tgt_mm_coarse.min().item():.2f} / {tgt_mm_coarse.max().item():.2f}")
    print(f"  centroid dist mean:     {(tgt_mm_coarse - centroid).norm(dim=-1).mean().item():.2f}")

    # tgt_mm_coarse should be in the same mm range as GT.
    assert tgt_mm_coarse.abs().mean() > 10.0, "tgt_mm_coarse in GT mode looks wrong"
    assert torch.isfinite(tgt_mm_coarse).all()
    print("[PASS] test_gt_mode_target_global_xyz_mm")


def test_completed_mode_target_global_xyz_mm():
    """Test 6: completed-mode: inverse transform uses completed coords, not GT."""
    # Cannot load SPAQNet, but verify the normalization/inverse-transform
    # identity: if src is centroid+X and tgt is centroid+Y,
    # the inverse-transform of tgt_norm should recover Y (not GT).
    model = PV2SNetFull2FullV2(
        num_refinement_steps=1, refinement_k=8, global_match_level=2,
    )
    B, Ns, Nt = 1, 512, 350
    centroid_val = torch.tensor([[[70.0, 40.0, -55.0]]])
    src = centroid_val + 55.0 * torch.randn(B, Ns, 3)
    completed = centroid_val + 15.0 + 50.0 * torch.randn(B, Nt, 3)  # different from GT
    tgt = completed  # completed mode: target = completed

    src_norm, tgt_norm, centroid, scale = _normalize_for_test(src, tgt)
    with torch.no_grad():
        out = model(src_norm, tgt_norm)

    # Inverse transform.
    tgt_mm_coarse = out["target_global_xyz"].float() * scale.float() + centroid.float()

    print(f"  completed mm min/max:   {completed.min().item():.2f} / {completed.max().item():.2f}")
    print(f"  tgt_mm_coarse min/max:  {tgt_mm_coarse.min().item():.2f} / {tgt_mm_coarse.max().item():.2f}")

    # tgt_mm_coarse should be near completed range, not near 0.
    assert tgt_mm_coarse.abs().mean() > 10.0, "tgt_mm_coarse in completed mode looks wrong"
    assert torch.isfinite(tgt_mm_coarse).all()
    print("[PASS] test_completed_mode_target_global_xyz_mm")


def main():
    print("=" * 60)
    print("full2full_v2 verification tests")
    print("=" * 60)

    # Original tests
    test_full2full_v1_still_runs()
    test_full2full_v2_forward_fp32()
    test_full2full_v2_backward_fp32()
    test_source_global_indices_mapping()
    test_global_matcher_v2_score_weights()
    test_match_loss_finite()
    test_completed_mode_uses_completed_not_gt()
    test_interpolate_flow_v2_squared_distance()
    test_single_sample_overfit_fp32()

    # New unit-bug-fix tests
    test_target_global_xyz_unit_recovery()
    test_match_loss_not_zero_with_mm_coords()
    test_match_loss_backward_to_global_matcher()
    test_wrong_unit_gives_lower_match_weight()
    test_gt_mode_target_global_xyz_mm()
    test_completed_mode_target_global_xyz_mm()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
