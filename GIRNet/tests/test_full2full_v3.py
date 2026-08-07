import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
GIRNET_ROOT = ROOT / "GIRNet"
for path in (ROOT, GIRNET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.P_V2S_Net_Full2Full_V3 import PV2SNetFull2FullV3
from models.global_matcher_v3 import GlobalMatcherV3
from models.iterative_refiner_v3 import build_fixed_source_knn
from loss_v3 import source_edge_consistency_loss, source_edge_error_mm


def test_v3_has_no_multiplicative_learned_gate():
    matcher = GlobalMatcherV3()
    assert not hasattr(matcher, "coarse_gate_logit")


def test_fixed_source_graph_is_flow_independent():
    torch.manual_seed(0)
    source = torch.randn(2, 32, 3)
    idx_a = build_fixed_source_knn(source, 8)
    # Current flow is intentionally irrelevant to the graph constructor.
    _flow = torch.randn_like(source) * 100.0
    idx_b = build_fixed_source_knn(source, 8)
    assert torch.equal(idx_a, idx_b)


def test_edge_loss_zero_for_exact_correspondence():
    torch.manual_seed(1)
    source = torch.randn(2, 32, 3) * 20.0
    gt = source + torch.randn_like(source) * 2.0
    knn = build_fixed_source_knn(source, 8)
    loss = source_edge_consistency_loss(gt, gt, knn, beta_mm=2.0)
    err = source_edge_error_mm(gt, gt, knn)
    assert loss.item() < 1e-8
    assert err.item() < 1e-8


def test_edge_loss_detects_index_sliding():
    # Same geometric set, different source-index assignment.
    n = 32
    theta = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    gt = torch.stack(
        [20.0 * torch.cos(theta), 20.0 * torch.sin(theta), torch.zeros_like(theta)],
        dim=-1,
    ).unsqueeze(0)
    source = gt.clone()
    pred = torch.roll(gt, shifts=5, dims=1)
    knn = build_fixed_source_knn(source, 4)
    loss = source_edge_consistency_loss(pred, gt, knn, beta_mm=2.0)
    assert loss.item() > 0.1


def test_source_indexed_output_shape_and_memory():
    torch.manual_seed(2)
    model = PV2SNetFull2FullV3(
        global_match_level=4,
        num_refinement_steps=1,
        refinement_k=8,
        source_graph_k=8,
    )
    source = torch.randn(1, 512, 3)
    target = torch.randn(1, 512, 3)
    with torch.no_grad():
        out = model(source, target)
    assert out["result"].shape == source.shape
    assert out["flow_stages"][0].shape == source.shape
    assert out["global_memory"].shape[:2] == source.shape[:2]
    assert out["source_knn_indices"].shape == (1, 512, 8)
    assert out["global_assignment"].shape[1] == model.global_match_points


def main():
    test_v3_has_no_multiplicative_learned_gate()
    test_fixed_source_graph_is_flow_independent()
    test_edge_loss_zero_for_exact_correspondence()
    test_edge_loss_detects_index_sliding()
    test_source_indexed_output_shape_and_memory()
    print("All full2full_v3 tests passed")


if __name__ == "__main__":
    main()
