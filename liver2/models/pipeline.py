import torch
import torch.nn as nn

from .girnet.backbone_v3 import PV2SNetFull2FullV3
from completion.SPAQNet.models.liver_generative_completion import (
    LiverGenerativeCompletionSPAQNet,
)


class LiverV3Model(nn.Module):
    def __init__(
        self,
        completion_checkpoint="",
        completion_from_scratch=False,
        end_to_end_completion=False,
        global_match_level=4,
        global_match_dim=64,
        num_refinement_steps=3,
        refinement_k=35,
        debug_refinement=False,
        v3_feature_temperature=1.0,
        v3_spatial_temperature=1.0,
        source_graph_k=16,
        init_registration_checkpoint="",
    ):
        super().__init__()
        self.end_to_end_completion = bool(end_to_end_completion)
        self.completion = None
        self.completion_config = {}
        if completion_checkpoint and completion_from_scratch:
            raise ValueError(
                "completion_checkpoint and completion_from_scratch are "
                "mutually exclusive"
            )
        if completion_from_scratch:
            self.completion = self._build_completion_model({})
            trainable = sum(
                parameter.numel()
                for parameter in self.completion.parameters()
                if parameter.requires_grad
            )
            print(
                "[Info] Initialized trainable SPAQNet completion model from "
                f"scratch; trainable={trainable:,}/{trainable:,}"
            )
        elif completion_checkpoint:
            self.completion = self._load_completion_checkpoint(
                completion_checkpoint
            )
        self.backbone = PV2SNetFull2FullV3(
            feature_dim=50,
            points_per_region=35,
            global_match_level=global_match_level,
            global_match_dim=global_match_dim,
            feature_temperature=v3_feature_temperature,
            spatial_temperature=v3_spatial_temperature,
            num_refinement_steps=num_refinement_steps,
            refinement_k=refinement_k,
            source_graph_k=source_graph_k,
            enc_freq=(2e-2, 2e-1, 2, 4, 8, 16, 32, 64),
            enc_freq_scale=1,
            debug_refinement=debug_refinement,
        )
        if init_registration_checkpoint:
            self._init_registration_from_checkpoint(init_registration_checkpoint)

    def _build_completion_model(self, config):
        config = dict(config)
        if config.get("architecture", "generative") != "generative":
            raise ValueError("Completion model architecture must be generative")
        completion = LiverGenerativeCompletionSPAQNet(
            feature_dim=int(config.get("feature_dim", 192)),
            num_heads=int(config.get("num_heads", 6)),
            k_neighbors=int(config.get("k_neighbors", 12)),
            context_points=int(config.get("context_points", 256)),
            num_output_points=int(config.get("num_points", 2048)),
            coarse_points=int(config.get("coarse_points", 256)),
            encoder_depth=int(config.get("encoder_depth", 3)),
            decoder_depth=int(config.get("decoder_depth", 4)),
            denoise_queries=int(config.get("denoise_queries", 64)),
            denoise_jitter=float(config.get("denoise_jitter", 0.005)),
        )
        self.completion_config = config
        return completion

    def _load_completion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                "Completion checkpoint must contain model and config: "
                f"{checkpoint_path}"
            )
        config = dict(checkpoint.get("config", {}))
        completion = self._build_completion_model(config)
        state = {
            key.removeprefix("module."): value
            for key, value in checkpoint["model"].items()
        }
        completion.load_state_dict(state, strict=True)
        for parameter in completion.parameters():
            parameter.requires_grad_(True)
        trainable = sum(
            parameter.numel()
            for parameter in completion.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in completion.parameters())
        print(
            "[Info] Loaded trainable SPAQNet completion checkpoint "
            f"{checkpoint_path}; epoch={checkpoint.get('epoch', -1) + 1}; "
            f"trainable={trainable:,}/{total:,}"
        )
        return completion

    @staticmethod
    def _normalize_for_GIRNet(src_xyz, part_xyz, eps=1e-6):
        centroid = src_xyz.float().mean(dim=1, keepdim=True)
        src_centered = src_xyz.float() - centroid
        scale = torch.linalg.norm(src_centered, dim=-1).amax(dim=1, keepdim=True)
        scale = scale.clamp_min(eps).unsqueeze(-1)
        return (src_centered / scale).to(dtype=src_xyz.dtype), (part_xyz.float() - centroid).to(dtype=part_xyz.dtype) / scale, centroid, scale

    def _init_registration_from_checkpoint(self, checkpoint_path):
        """Strictly load registration weights from a V3 checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_arch = None
        if isinstance(checkpoint, dict):
            checkpoint_arch = checkpoint.get("GIRNet_arch")
            if checkpoint_arch is None:
                checkpoint_arch = checkpoint.get("config", {}).get("GIRNet_arch")
        if checkpoint_arch != "full2full_v3":
            raise RuntimeError(
                "Registration initialization requires a full2full_v3 checkpoint; "
                f"got {checkpoint_arch!r}"
            )

        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break
        if not isinstance(state, dict):
            raise RuntimeError("Registration checkpoint has no state dict")

        current = self.backbone.state_dict()
        cleaned = {}
        shape_mismatch = []
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith("backbone."):
                key = key.removeprefix("backbone.")
            elif key not in current:
                # Wrapper checkpoints also contain completion/text parameters.
                continue
            if key in current:
                if current[key].shape != value.shape:
                    shape_mismatch.append(
                        f"{key}: checkpoint={tuple(value.shape)} "
                        f"current={tuple(current[key].shape)}"
                    )
                else:
                    cleaned[key] = value

        loaded_count = len(cleaned)
        if loaded_count == 0:
            raise RuntimeError(
                "init_registration_checkpoint matched zero backbone parameters; "
                "check checkpoint format and architecture"
            )
        missing = [key for key in current if key not in cleaned]
        if missing or shape_mismatch:
            raise RuntimeError(
                "Same-architecture registration initialization must be strict: "
                f"loaded={loaded_count}, missing={len(missing)}, "
                f"shape_mismatch={len(shape_mismatch)}"
            )

        self.backbone.load_state_dict(cleaned, strict=True)
        print(
            f"[Info] init_registration_checkpoint {checkpoint_path}: "
            f"strictly loaded {loaded_count} backbone tensors"
        )

    def forward(
        self,
        src_xyz,
        part_xyz,
        E_kPa=None,
        nu=None,
        return_completion=False,
        registration_target_xyz=None,
        freeze_completion=True,
    ):
        completion_outputs = None
        completed_xyz = part_xyz  # fallback when completion is absent
        if self.completion is not None:
            partial_mask = torch.ones(
                part_xyz.shape[:2],
                dtype=torch.bool,
                device=part_xyz.device,
            )
            completion_outputs = self.completion(
                src_xyz,
                part_xyz,
                partial_mask,
            )
            completed_xyz = completion_outputs["completed_xyz"]
            if freeze_completion or not self.end_to_end_completion:
                completed_xyz = completed_xyz.detach()

        # Decide what to use as the GIRNet registration target.
        if registration_target_xyz is not None:
            registration_target = registration_target_xyz
        else:
            registration_target = completed_xyz

        # Preserve the historical detached target unless end-to-end completion
        # is explicitly enabled in a non-frozen training stage.
        if registration_target_xyz is None:
            registration_target = completed_xyz
        elif freeze_completion:
            registration_target = registration_target.detach()

        src_norm, part_norm, centroid, scale = self._normalize_for_GIRNet(
            src_xyz, registration_target
        )
        results = self.backbone(src_norm, part_norm)

        normalized_flow_stages = results["flow_stages"]
        pred_stages_xyz = [
            src_xyz + normalized_flow * scale
            for normalized_flow in normalized_flow_stages
        ]
        pred_xyz = pred_stages_xyz[-1]
        global_match_confidence = results["global_match_confidence"]
        global_assignment = results.get("global_assignment")
        source_global_indices = results.get("source_global_indices")
        target_global_xyz_normalized = results.get("target_global_xyz")
        if target_global_xyz_normalized is not None:
            target_global_xyz = (
                target_global_xyz_normalized.float() * scale.float()
                + centroid.float()
            )
        else:
            target_global_xyz = None
        score_weights = results.get("score_weights")
        source_knn_indices = results.get("source_knn_indices")

        # Always return completion outputs for metric logging.
        return {
            "pred_xyz": pred_xyz,
            "pred_stages_xyz": pred_stages_xyz,
            "completed_xyz": completed_xyz,
            "completion_outputs": completion_outputs,
            "global_match_confidence": global_match_confidence,
            "global_assignment": global_assignment,
            "source_global_indices": source_global_indices,
            "target_global_xyz": target_global_xyz,
            "score_weights": score_weights,
            "source_knn_indices": source_knn_indices,
        }
