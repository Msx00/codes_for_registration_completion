import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoder


class SymmetricFullToFullPreprocessor(nn.Module):
    """Build identical full-to-full features for source and target.

    For a point x in cloud X and the other cloud Y, the five base features
    are [min_y ||x-y||_2, x, y, z, 1].  The raw coordinates and their
    positional encoding contribute another 51 channels, so the returned
    feature tensor has exactly 56 channels on both sides.
    """

    def __init__(
        self,
        enc_freq=(2e-2, 2e-1, 2, 4, 8, 16, 32, 64),
        enc_freq_scale=1.0,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(
            list(enc_freq),
            enc_freq_scale,
        )
        self.coordinate_feature_dim = 3 + self.positional_encoder.num_features
        self.base_feature_dim = 5
        self.output_feature_dim = (
            self.coordinate_feature_dim + self.base_feature_dim
        )
        if self.output_feature_dim != 56:
            raise ValueError(
                "Full-to-full preprocessing must produce 56 channels, got "
                f"{self.output_feature_dim}. Check positional frequencies."
            )

    @staticmethod
    def _cross_distance(a, b):
        # cdist/reduction in FP32 avoids half-precision overflow and keeps the
        # same source/target feature definition under AMP.
        distance = torch.cdist(a.float(), b.float(), p=2)
        return distance.amin(dim=-1, keepdim=True)

    def _features(self, xyz, cross_distance):
        coords_cf = xyz.transpose(1, 2).contiguous()
        positional = self.positional_encoder(coords_cf.float())
        coordinate_features = torch.cat([coords_cf.float(), positional], dim=1)
        ones = torch.ones_like(cross_distance)
        base_features = torch.cat(
            [cross_distance, xyz.float(), ones],
            dim=-1,
        ).transpose(1, 2).contiguous()
        features = torch.cat([coordinate_features, base_features], dim=1)
        if features.shape[1] != 56:
            raise RuntimeError(
                f"Expected 56 symmetric input channels, got {features.shape[1]}"
            )
        return features

    def forward(self, source_xyz, target_xyz):
        if source_xyz.ndim != 3 or source_xyz.shape[-1] != 3:
            raise ValueError("source_xyz must have shape (B, Nsource, 3)")
        if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3:
            raise ValueError("target_xyz must have shape (B, Ntarget, 3)")
        if source_xyz.shape[0] != target_xyz.shape[0]:
            raise ValueError("source_xyz and target_xyz batch sizes must match")

        pairwise = torch.cdist(source_xyz.float(), target_xyz.float(), p=2)
        source_cross = pairwise.amin(dim=-1, keepdim=True)
        target_cross = pairwise.amin(dim=-2).unsqueeze(-1)
        return (
            self._features(source_xyz, source_cross),
            self._features(target_xyz, target_cross),
        )
