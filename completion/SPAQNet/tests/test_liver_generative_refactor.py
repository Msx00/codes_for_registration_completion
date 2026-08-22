"""Regression coverage for the modular generative liver SPAQNet."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import unittest

import torch

from models.liver_generative_completion import LiverGenerativeCompletionSPAQNet
from models.point_ops import _batch_gather, fps_points, knn_indices


def _make_model(denoise_queries: int = 64) -> LiverGenerativeCompletionSPAQNet:
    return LiverGenerativeCompletionSPAQNet(
        feature_dim=24,
        num_heads=4,
        k_neighbors=4,
        context_points=64,
        num_output_points=128,
        coarse_points=16,
        encoder_depth=1,
        decoder_depth=1,
        denoise_queries=denoise_queries,
        denoise_jitter=0.005,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1729)
    source = torch.randn(1, 72, 3, generator=generator)
    partial = torch.randn(1, 70, 3, generator=generator)
    partial_mask = torch.ones(1, 70, dtype=torch.bool)
    return source, partial, partial_mask


def _legacy_monolithic_forward(
    model: LiverGenerativeCompletionSPAQNet,
    source: torch.Tensor,
    partial: torch.Tensor,
    partial_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The pre-refactor forward dataflow, expressed against migrated modules."""
    local_module = model.local_graph_encoding
    query_module = model.adaptive_query_generator
    memory_module = model.geometric_memory_encoder
    decoder_module = model.geometric_query_decoder
    expansion_module = model.coarse_to_fine_expansion

    source_norm, partial_norm, centroid, scale = model.normalize_by_source(
        source, partial
    )
    source_context, _ = fps_points(source_norm, model.context_points)
    partial_context = local_module._select_partial_context(partial_norm, partial_mask)
    source_features, _ = local_module.source_encoder(source_context)
    partial_features, _ = local_module.partial_encoder(partial_context)
    memory_positions = torch.cat([source_context, partial_context], dim=1)
    memory = torch.cat(
        [
            source_features + query_module.source_type,
            partial_features + query_module.partial_type,
        ],
        dim=1,
    ) + query_module.position_embedding(memory_positions)
    memory_neighbors = knn_indices(
        memory_positions, memory_module.memory_encoder[0].k_neighbors
    )
    for block in memory_module.memory_encoder:
        memory = block(memory, memory_positions, memory_neighbors)
    source_count = source_context.shape[1]
    source_features = memory[:, :source_count]
    partial_features = memory[:, source_count:]
    global_features = memory_module.global_fusion(
        torch.cat(
            [
                source_features.amax(dim=1),
                partial_features.amax(dim=1),
            ],
            dim=-1,
        )
    ).unsqueeze(1)

    source_xyz, source_anchor_features = fps_points(
        source_context,
        model.anchor_candidates,
        source_features,
    )
    partial_xyz, partial_anchor_features = fps_points(
        partial_context,
        model.anchor_candidates,
        partial_features,
    )
    predicted_xyz = torch.tanh(
        query_module.predicted_candidate_head(global_features.squeeze(1)).reshape(
            source_context.shape[0], model.predicted_candidates, 3
        )
    ) * 1.25
    predicted_features = query_module.predicted_feature_head(
        torch.cat(
            [
                global_features.expand(-1, model.predicted_candidates, -1),
                predicted_xyz,
            ],
            dim=-1,
        )
    )
    candidate_xyz = torch.cat([source_xyz, partial_xyz, predicted_xyz], dim=1)
    candidate_features = torch.cat(
        [
            source_anchor_features + query_module.source_type,
            partial_anchor_features + query_module.partial_type,
            predicted_features + query_module.predicted_type,
        ],
        dim=1,
    )
    ranking_scores = query_module.query_ranker(
        torch.cat([candidate_features, candidate_xyz], dim=-1)
    ).squeeze(-1)
    source_count = source_xyz.shape[1]
    partial_count = partial_xyz.shape[1]
    source_keep = min(model.anchor_queries, source_count)
    partial_keep = min(model.anchor_queries, partial_count)
    predicted_keep = model.coarse_points - source_keep - partial_keep
    source_indices = ranking_scores[:, :source_count].topk(
        source_keep, dim=1, largest=True
    ).indices
    partial_indices = ranking_scores[
        :, source_count : source_count + partial_count
    ].topk(partial_keep, dim=1, largest=True).indices + source_count
    predicted_indices = ranking_scores[:, source_count + partial_count :].topk(
        predicted_keep, dim=1, largest=True
    ).indices + source_count + partial_count
    selected_indices = torch.cat(
        [source_indices, partial_indices, predicted_indices], dim=1
    )
    query_xyz = _batch_gather(candidate_xyz, selected_indices)
    query_features = _batch_gather(candidate_features, selected_indices)
    selected_scores = _batch_gather(
        ranking_scores.unsqueeze(-1), selected_indices
    )
    query_features = query_features * (1.0 + selected_scores)
    query_features = (
        query_features
        + query_module.position_embedding(query_xyz)
        + global_features
    )
    regular_query_count = query_xyz.shape[1]
    denoise_target = None
    if model.training and model.denoise_queries > 0:
        denoise_target, denoise_features = fps_points(
            partial_context, model.denoise_queries, partial_features
        )
        denoise_xyz = denoise_target + torch.randn_like(
            denoise_target
        ) * model.denoise_jitter
        with torch.no_grad(), torch.autocast(
            device_type=denoise_xyz.device.type, enabled=False
        ):
            nearest_indices = torch.cdist(
                denoise_xyz.float(), partial_context.float()
            ).argmin(dim=-1)
        nearest_partial = _batch_gather(partial_context, nearest_indices)
        denoise_target = nearest_partial
        nearest_offset = nearest_partial - denoise_xyz
        nearest_distance = torch.linalg.vector_norm(
            nearest_offset.float(), dim=-1, keepdim=True
        ).to(denoise_xyz.dtype)
        denoise_geometry = query_module.denoise_geometry_projection(
            torch.cat(
                [denoise_xyz, nearest_offset, nearest_distance], dim=-1
            )
        )
        denoise_features = (
            denoise_features
            + query_module.position_embedding(denoise_xyz)
            + denoise_geometry
            + global_features
            + query_module.denoise_type
        )
        query_xyz = torch.cat([query_xyz, denoise_xyz], dim=1)
        query_features = torch.cat([query_features, denoise_features], dim=1)
        attention_mask = torch.zeros(
            query_xyz.shape[1],
            query_xyz.shape[1],
            dtype=torch.bool,
            device=query_xyz.device,
        )
        attention_mask[:regular_query_count, regular_query_count:] = True
    else:
        attention_mask = None

    query_k = min(
        decoder_module.query_decoder[0].k_neighbors,
        max(regular_query_count - 1, 1),
    )
    query_neighbors = knn_indices(query_xyz, query_k)
    if denoise_target is not None:
        regular_neighbors = knn_indices(
            query_xyz[:, :regular_query_count], query_k
        )
        query_neighbors = torch.cat(
            [regular_neighbors, query_neighbors[:, regular_query_count:]], dim=1
        )
    for block in decoder_module.query_decoder:
        query_features = block(
            query_features,
            query_xyz,
            memory,
            query_neighbors,
            attention_mask,
        )
    coarse = query_xyz[:, :regular_query_count] + torch.tanh(
        decoder_module.coarse_refine_head(
            query_features[:, :regular_query_count]
        )
    ) * 0.10
    denoised = (
        query_xyz[:, regular_query_count:]
        + torch.tanh(
            decoder_module.denoise_refine_head(
                query_features[:, regular_query_count:]
            )
        )
        * 0.10
        if denoise_target is not None
        else None
    )
    partial_seed_features = expansion_module.partial_seed_projection(
        partial_features + global_features
    )
    seed_xyz = torch.cat([coarse, partial_context], dim=1)
    seed_features = torch.cat(
        [query_features[:, :regular_query_count], partial_seed_features], dim=1
    )
    mid, mid_features = expansion_module.mid_expansion(
        seed_xyz, seed_features, global_features, model.mid_points
    )
    fine, _ = expansion_module.fine_expansion(
        mid, mid_features, global_features, model.num_output_points
    )
    return {
        "coarse_normalized": coarse,
        "mid_normalized": mid,
        "fine_normalized": fine,
        "completed_xyz": fine * scale + centroid,
        "partial_normalized": partial_norm,
        "denoised_normalized": denoised,
        "denoise_target_normalized": denoise_target,
        "ranking_scores": ranking_scores,
        "centroid": centroid,
        "scale": scale,
    }


def _as_legacy_state_dict(
    model: LiverGenerativeCompletionSPAQNet,
) -> OrderedDict[str, torch.Tensor]:
    reverse = {
        new_prefix: old_prefix
        for old_prefix, new_prefix in model.LEGACY_STATE_DICT_PREFIXES.items()
    }
    result = OrderedDict()
    for key, value in model.state_dict().items():
        matches = [prefix for prefix in reverse if key.startswith(prefix)]
        assert len(matches) == 1, key
        new_prefix = matches[0]
        result[reverse[new_prefix] + key[len(new_prefix) :]] = value.clone()
    return result


def _check_eval_outputs_match_pre_refactor_dataflow() -> None:
    torch.manual_seed(11)
    legacy_model = _make_model().eval()
    modular_model = _make_model().eval()
    modular_model.load_state_dict(
        _as_legacy_state_dict(legacy_model), strict=True
    )
    source, partial, partial_mask = _inputs()
    with torch.no_grad():
        legacy = _legacy_monolithic_forward(
            legacy_model, source, partial, partial_mask
        )
        modular = modular_model(source, partial, partial_mask)
    for key in (
        "coarse_normalized",
        "mid_normalized",
        "fine_normalized",
        "completed_xyz",
    ):
        torch.testing.assert_close(modular[key], legacy[key], rtol=0, atol=0)


def _check_training_uses_64_denoising_queries_and_backward_is_finite() -> None:
    torch.manual_seed(23)
    model = _make_model(denoise_queries=64).train()
    source, partial, partial_mask = _inputs()
    output = model(source, partial, partial_mask)
    assert output["denoised_normalized"].shape == (1, 64, 3)
    assert output["denoise_target_normalized"].shape == (1, 64, 3)
    loss = (
        output["coarse_normalized"].square().mean()
        + output["mid_normalized"].square().mean()
        + output["fine_normalized"].square().mean()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )


def _check_legacy_state_dict_loads_strictly() -> None:
    torch.manual_seed(31)
    original = _make_model()
    legacy_state = _as_legacy_state_dict(original)
    restored = _make_model()
    result = restored.load_state_dict(legacy_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)

    incomplete = OrderedDict(legacy_state)
    incomplete.pop(next(iter(incomplete)))
    try:
        _make_model().load_state_dict(incomplete, strict=True)
    except RuntimeError as error:
        assert "Missing key" in str(error)
    else:
        raise AssertionError("strict=True accepted an incomplete legacy state_dict")


def _check_saved_legacy_checkpoint_loads_strictly() -> None:
    checkpoint_path = (
        Path(__file__).parents[2]
        / "logs"
        / "full_aug_20260811_104608_overlap0.10"
        / "best.pth"
    )
    if not checkpoint_path.is_file():
        raise unittest.SkipTest("local training checkpoint is not available")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = LiverGenerativeCompletionSPAQNet(
        feature_dim=int(config["feature_dim"]),
        num_heads=int(config["num_heads"]),
        k_neighbors=int(config["k_neighbors"]),
        context_points=int(config["context_points"]),
        num_output_points=int(config["num_points"]),
        coarse_points=int(config["coarse_points"]),
        encoder_depth=int(config["encoder_depth"]),
        decoder_depth=int(config["decoder_depth"]),
        denoise_queries=int(config["denoise_queries"]),
        denoise_jitter=float(config["denoise_jitter"]),
    )
    result = model.load_state_dict(checkpoint["model"], strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


class LiverGenerativeRefactorTests(unittest.TestCase):
    def test_eval_outputs_match_pre_refactor_dataflow(self) -> None:
        _check_eval_outputs_match_pre_refactor_dataflow()

    def test_training_uses_64_denoising_queries_and_backward_is_finite(
        self,
    ) -> None:
        _check_training_uses_64_denoising_queries_and_backward_is_finite()

    def test_legacy_state_dict_loads_strictly(self) -> None:
        _check_legacy_state_dict_loads_strictly()

    def test_saved_legacy_checkpoint_loads_strictly(self) -> None:
        _check_saved_legacy_checkpoint_loads_strictly()


if __name__ == "__main__":
    unittest.main()
