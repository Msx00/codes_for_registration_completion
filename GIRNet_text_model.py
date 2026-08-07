import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel


GIRNet_ROOT = Path(__file__).resolve().parent / "GIRNet"
if str(GIRNet_ROOT) not in sys.path:
    sys.path.insert(0, str(GIRNet_ROOT))

from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P_dgcnn import (  # noqa: E402
    PV2SNetV5DownsampledIntraopV2I2PDGCNN,
)
from models.P_V2S_Net_Full2Full_V1 import PV2SNetFull2FullV1  # noqa: E402
from models.P_V2S_Net_Full2Full_V2 import PV2SNetFull2FullV2  # noqa: E402
from completion.SPAQNet.models.liver_generative_completion import (  # noqa: E402
    LiverGenerativeCompletionSPAQNet,
)


class BertFiLM(nn.Module):
    def __init__(
        self,
        out_channels=50,
        model_name=None,
        local_files_only=True,
        train_bert=False,
    ):
        super().__init__()
        if model_name is None:
            model_name = str(Path(__file__).resolve().parent / "bert-base-uncased")

        self.bert = BertModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.train_bert = train_bert
        if not train_bert:
            self.bert.eval()
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden = self.bert.config.hidden_size
        self.to_film = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels * 2),
        )

        nn.init.zeros_(self.to_film[-1].weight)
        nn.init.zeros_(self.to_film[-1].bias)

    def forward(self, input_ids, attention_mask):
        if self.train_bert:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled = out.pooler_output
        gamma, beta = self.to_film(pooled).chunk(2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)


class TextConditionedGIRNet(nn.Module):
    def __init__(
        self,
        bert_model_name=None,
        bert_local_files_only=True,
        train_bert=False,
        use_text=True,
        load_GIRNet_checkpoint="",
        strict_GIRNet_checkpoint=False,
        completion_checkpoint="",
        GIRNet_arch="legacy",
        global_match_level=2,
        global_match_temperature=0.1,
        global_match_dim=64,
        global_spatial_sigma=0.2,
        max_coarse_flow_normalized=0.25,
        num_refinement_steps=3,
        refinement_k=35,
        initialize_from_legacy_GIRNet=True,
        debug_refinement=False,
        global_gate_temperature=0.02,
        init_registration_checkpoint="",
    ):
        super().__init__()
        self.use_text = use_text
        self.GIRNet_arch = GIRNet_arch
        self.completion = None
        self.completion_config = {}
        if completion_checkpoint:
            self.completion = self._load_completion_checkpoint(
                completion_checkpoint
            )
        if GIRNet_arch == "legacy":
            self.backbone = PV2SNetV5DownsampledIntraopV2I2PDGCNN(
                n_input_features=5,
                n_preprocess_features=50,
                n_intermediate_features=[200, 150, 110, 80, 60, 50],
                n_intermediate_points=[8, 35, 92, 144, 239, 321],
                n_output_features=3,
                embedding_size=29,
                points_per_region=35,
                enc_freq=[2e-2, 2e-1, 2, 4, 8, 16, 32, 64],
                enc_freq_scale=1,
                append_df_self=True,
                append_df_cross=True,
                append_positional_encoding=True,
                compact_return=True,
            )
        elif GIRNet_arch == "full2full_v1":
            self.backbone = PV2SNetFull2FullV1(
                feature_dim=50,
                points_per_region=35,
                global_match_level=global_match_level,
                global_match_temperature=global_match_temperature,
                global_match_dim=global_match_dim,
                global_spatial_sigma=global_spatial_sigma,
                max_coarse_flow_normalized=max_coarse_flow_normalized,
                num_refinement_steps=num_refinement_steps,
                refinement_k=refinement_k,
                enc_freq=(2e-2, 2e-1, 2, 4, 8, 16, 32, 64),
                enc_freq_scale=1,
                debug_refinement=debug_refinement,
            )
        elif GIRNet_arch == "full2full_v2":
            self.backbone = PV2SNetFull2FullV2(
                feature_dim=50,
                points_per_region=35,
                global_match_level=global_match_level,
                global_match_temperature=global_match_temperature,
                global_match_dim=global_match_dim,
                global_spatial_sigma=global_spatial_sigma,
                max_coarse_flow_normalized=max_coarse_flow_normalized,
                gate_temperature=global_gate_temperature,
                num_refinement_steps=num_refinement_steps,
                refinement_k=refinement_k,
                enc_freq=(2e-2, 2e-1, 2, 4, 8, 16, 32, 64),
                enc_freq_scale=1,
                debug_refinement=debug_refinement,
            )
        else:
            raise ValueError(
                f"Unsupported GIRNet_arch={GIRNet_arch!r}; expected legacy, full2full_v1, or full2full_v2"
            )
        self.text_film = BertFiLM(
            out_channels=50,
            model_name=bert_model_name,
            local_files_only=bert_local_files_only,
            train_bert=train_bert,
        )
        self._film = None
        self._hook_handle = self.backbone.reduce_channels.register_forward_hook(
            self._apply_text_film
        )

        if load_GIRNet_checkpoint:
            if GIRNet_arch == "legacy":
                self.load_GIRNet_checkpoint(
                    load_GIRNet_checkpoint,
                    strict=strict_GIRNet_checkpoint,
                )
            elif initialize_from_legacy_GIRNet:
                self.initialize_full2full_from_legacy(load_GIRNet_checkpoint)
            else:
                print(
                    "[Info] full2full_v1 legacy initialization disabled; "
                    "GIRNet backbone starts from its own initialization"
                )

        if init_registration_checkpoint:
            self._init_registration_from_checkpoint(init_registration_checkpoint)

    def _load_completion_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(
                "Completion checkpoint must contain model and config: "
                f"{checkpoint_path}"
            )
        config = dict(checkpoint.get("config", {}))
        if config.get("architecture", "generative") != "generative":
            raise ValueError("Joint training requires a generative completion checkpoint")
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
        state = {
            key.removeprefix("module."): value
            for key, value in checkpoint["model"].items()
        }
        completion.load_state_dict(state, strict=True)
        for parameter in completion.parameters():
            parameter.requires_grad_(True)
        self.completion_config = config
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

    def _apply_text_film(self, module, inputs, output):
        if self._film is None:
            return output
        gamma, beta = self._film
        return output * (1.0 + gamma.to(dtype=output.dtype)) + beta.to(dtype=output.dtype)

    @staticmethod
    def _as_GIRNet_preop(xyz):
        bsz, _, _ = xyz.shape
        zeros = torch.zeros(bsz, xyz.shape[1], 1, device=xyz.device, dtype=xyz.dtype)
        return torch.cat([xyz, zeros, xyz], dim=-1).transpose(1, 2).contiguous()

    @staticmethod
    def _as_GIRNet_intraop(xyz):
        bsz, _, _ = xyz.shape
        zeros = torch.zeros(bsz, xyz.shape[1], 3, device=xyz.device, dtype=xyz.dtype)
        return torch.cat([xyz, zeros], dim=-1).transpose(1, 2).contiguous()

    @staticmethod
    def _normalize_for_GIRNet(src_xyz, part_xyz, eps=1e-6):
        centroid = src_xyz.float().mean(dim=1, keepdim=True)
        src_centered = src_xyz.float() - centroid
        scale = torch.linalg.norm(src_centered, dim=-1).amax(dim=1, keepdim=True)
        scale = scale.clamp_min(eps).unsqueeze(-1)
        return (src_centered / scale).to(dtype=src_xyz.dtype), (part_xyz.float() - centroid).to(dtype=part_xyz.dtype) / scale, centroid, scale

    @staticmethod
    def _break_exact_duplicates(xyz, eps=1e-3):
        n_points = xyz.shape[1]
        idx = torch.arange(n_points, device=xyz.device, dtype=xyz.dtype).view(1, n_points, 1)
        offsets = torch.cat(
            [
                (idx % 997) / 997.0,
                (idx % 991) / 991.0,
                (idx % 983) / 983.0,
            ],
            dim=-1,
        )
        offsets = offsets - offsets.mean(dim=1, keepdim=True)
        return xyz + eps * offsets

    def load_GIRNet_checkpoint(self, checkpoint_path, strict=False):
        state = self._clean_GIRNet_state(
            torch.load(checkpoint_path, map_location="cpu")
        )
        cleaned = state
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)

        max_show = 20
        print(
            "[Info] Loaded GIRNet checkpoint "
            f"{checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print("[Info] GIRNet checkpoint missing keys (first {}):".format(
                min(len(missing), max_show)))
            for k in list(missing)[:max_show]:
                print(f"  - {k}")
        if unexpected:
            print("[Info] GIRNet checkpoint unexpected keys (first {}):".format(
                min(len(unexpected), max_show)))
            for k in list(unexpected)[:max_show]:
                print(f"  - {k}")

        if strict and (missing or unexpected):
            raise RuntimeError(
                "strict_GIRNet_checkpoint is enabled but keys mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}. "
                "See printed key lists above."
            )

    @staticmethod
    def _clean_GIRNet_state(state):
        if isinstance(state, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise ValueError("GIRNet checkpoint does not contain a state dict")
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("backbone.")
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _print_key_group(title, keys, max_show=20):
        print(f"[Info] {title}: {len(keys)}")
        for key in list(keys)[:max_show]:
            print(f"  - {key}")

    def initialize_full2full_from_legacy(self, checkpoint_path):
        legacy = self._clean_GIRNet_state(
            torch.load(checkpoint_path, map_location="cpu")
        )
        current = self.backbone.state_dict()
        compatible = {}
        shape_mismatched = []
        unexpected = []
        for key, value in legacy.items():
            if key not in current:
                unexpected.append(key)
            elif current[key].shape != value.shape:
                shape_mismatched.append(
                    f"{key}: legacy={tuple(value.shape)} new={tuple(current[key].shape)}"
                )
            else:
                compatible[key] = value

        self.backbone.load_state_dict(compatible, strict=False)
        missing = [key for key in current if key not in compatible]
        self._print_key_group("loaded compatible keys", sorted(compatible))
        self._print_key_group("skipped shape-mismatched keys", shape_mismatched)
        self._print_key_group("missing new architecture keys", missing)
        self._print_key_group("unexpected legacy keys", unexpected)
        critical_mismatch = [
            key for key in shape_mismatched
            if key.startswith(("reduce_channels", "dgcnn_"))
        ]
        if critical_mismatch:
            print(
                "[Warning] Critical shared encoder keys have incompatible shapes; "
                "these layers were not initialized from legacy GIRNet:"
            )
            for key in critical_mismatch[:20]:
                print(f"  - {key}")

    def _init_registration_from_checkpoint(self, checkpoint_path):
        """Load registration-only weights from a GT-pretrained full2full_v2 checkpoint.

        Only loads GIRNet backbone, GlobalMatcher, and iterative refiner parameters.
        Does NOT load optimizer, GradScaler, epoch, SPAQNet params, or best metric.
        """
        if self.GIRNet_arch != "full2full_v2":
            raise RuntimeError(
                "--init_registration_checkpoint is only supported for "
                f"full2full_v2, got arch={self.GIRNet_arch!r}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_arch = None
        if isinstance(checkpoint, dict):
            checkpoint_arch = checkpoint.get("GIRNet_arch")
            if checkpoint_arch is None:
                checkpoint_arch = checkpoint.get("config", {}).get("GIRNet_arch")
        if checkpoint_arch != "full2full_v2":
            raise RuntimeError(
                "Cannot initialize full2full_v2 from a checkpoint with "
                f"architecture={checkpoint_arch!r}; only full2full_v2 is allowed"
            )

        # Extract state dict, stripping "module." and "backbone." prefixes.
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break

        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            cleaned[key] = value

        # Filter to only registration-related keys (exclude completion, BertFiLM).
        reg_keys = {
            k for k in self.backbone.state_dict()
        }
        filtered = {k: v for k, v in cleaned.items() if k in reg_keys}
        missing = [k for k in reg_keys if k not in cleaned]
        unexpected = [k for k in cleaned if k not in reg_keys]

        self.backbone.load_state_dict(filtered, strict=False)
        loaded_count = len(filtered)
        print(
            f"[Info] init_registration_checkpoint {checkpoint_path}: "
            f"loaded={loaded_count}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
        if missing:
            print("[Info] Registration init missing keys (first 20):")
            for k in missing[:20]:
                print(f"  - {k}")
        if unexpected:
            print("[Info] Registration init unexpected keys (first 20):")
            for k in unexpected[:20]:
                print(f"  - {k}")

    def forward(
        self,
        src_xyz,
        part_xyz,
        input_ids,
        attention_mask,
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
            completed_xyz = completion_outputs["completed_xyz"].detach()

        # Decide what to use as the GIRNet registration target.
        if registration_target_xyz is not None:
            registration_target = registration_target_xyz
        else:
            registration_target = completed_xyz

        # Completed point sets are unordered and must remain independent from
        # registration supervision. Joint mode may still train SPAQNet through
        # its own completion loss, but registration gradients never enter it.
        if registration_target_xyz is None:
            registration_target = completed_xyz
        elif freeze_completion:
            registration_target = registration_target.detach()

        src_norm, part_norm, centroid, scale = self._normalize_for_GIRNet(
            src_xyz, registration_target
        )
        if self.use_text:
            self._film = self.text_film(input_ids, attention_mask)
        else:
            self._film = None

        try:
            if self.GIRNet_arch == "legacy":
                src_norm = self._break_exact_duplicates(src_norm)
                part_norm = self._break_exact_duplicates(part_norm)
                preop = self._as_GIRNet_preop(src_norm)
                intraop = self._as_GIRNet_intraop(part_norm)
                results = self.backbone(preop, intraop)
            else:
                results = self.backbone(src_norm, part_norm)
        finally:
            self._film = None

        if self.GIRNet_arch == "legacy":
            # Legacy backbone uses (B, 3, N); wrapper output is always (B, N, 3).
            normalized_flow = results[-1]["result"].transpose(1, 2).contiguous()
            pred_xyz = src_xyz + normalized_flow * scale
            pred_stages_xyz = [pred_xyz]
            global_match_confidence = None
            global_assignment = None
            source_global_indices = None
            target_global_xyz = None
            score_weights = None
            global_raw_coarse_flow_mm = None
            global_pre_tanh_coarse_flow_mm = None
            global_coarse_flow_mm = None
            global_confidence_gate = None
            global_learned_gate = None
            global_coarse_gate = None
        else:
            normalized_flow_stages = results["flow_stages"]
            pred_stages_xyz = [
                src_xyz + normalized_flow * scale
                for normalized_flow in normalized_flow_stages
            ]
            pred_xyz = pred_stages_xyz[-1]
            global_match_confidence = results["global_match_confidence"]
            global_assignment = results.get("global_assignment")
            source_global_indices = results.get("source_global_indices")
            # target_global_xyz from backbone is in GIRNet normalized
            # coordinates.  Inverse-transform to mm so that
            # compute_match_loss can compare it with gt_xyz (mm) using
            # match_sigma_mm (mm).
            target_global_xyz_normalized = results.get("target_global_xyz")
            if target_global_xyz_normalized is not None:
                target_global_xyz = (
                    target_global_xyz_normalized.float() * scale.float()
                    + centroid.float()
                )
            else:
                target_global_xyz = None
            score_weights = results.get("score_weights")

            # Coarse flow diagnostics: convert from normalized to mm.
            # Flow is displacement (not position), so multiply by scale only.
            def _flow_to_mm(key_norm):
                flow_norm = results.get(key_norm)
                if flow_norm is not None:
                    return flow_norm.float() * scale.float()
                return None

            global_raw_coarse_flow_mm = _flow_to_mm("global_raw_coarse_flow")
            global_pre_tanh_coarse_flow_mm = _flow_to_mm("global_pre_tanh_coarse_flow")
            global_coarse_flow_mm = _flow_to_mm("global_coarse_flow")

            global_confidence_gate = results.get("global_confidence_gate")
            global_learned_gate = results.get("global_learned_gate")
            global_coarse_gate = results.get("global_coarse_gate")

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
            # Coarse diagnostics in mm.
            "global_raw_coarse_flow_mm": global_raw_coarse_flow_mm,
            "global_pre_tanh_coarse_flow_mm": global_pre_tanh_coarse_flow_mm,
            "global_coarse_flow_mm": global_coarse_flow_mm,
            "global_confidence_gate": global_confidence_gate,
            "global_learned_gate": global_learned_gate,
            "global_coarse_gate": global_coarse_gate,
        }
