"""MedShapeNet liver loading and GPU-side partial-view augmentation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CROP_TYPES = ("ball", "plane")
# CROP_TYPES = ("ball", "plane", "slab", "multi_ball")


class LiverCaseDataset(Dataset):
    """Load complete corresponding source/GT pairs exactly once per epoch."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        max_cases: int = -1,
        seed: int = 42,
    ):
        self.root = Path(dataset_root)
        self.split = split
        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {split_root}")
        self.case_folders = sorted(
            path.parent
            for path in split_root.rglob("gt.txt")
            if (path.parent / "source.txt").is_file()
        )
        if max_cases > 0 and len(self.case_folders) > max_cases:
            rng = np.random.default_rng(seed + (0 if split == "train" else 1))
            chosen = np.sort(
                rng.choice(len(self.case_folders), max_cases, replace=False)
            )
            self.case_folders = [self.case_folders[int(i)] for i in chosen]
        if not self.case_folders:
            raise RuntimeError(f"No source/gt cases found under {split_root}")

    def __len__(self) -> int:
        return len(self.case_folders)

    def __getitem__(self, index: int) -> dict[str, object]:
        case_folder = self.case_folders[index]
        source = np.loadtxt(case_folder / "source.txt", dtype=np.float32)
        gt = np.loadtxt(case_folder / "gt.txt", dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"Invalid source shape in {case_folder}: {source.shape}")
        if gt.shape != source.shape:
            raise ValueError(
                f"source/gt shapes differ in {case_folder}: "
                f"{source.shape} vs {gt.shape}"
            )
        return {
            "source_full": torch.from_numpy(source),
            "gt_full": torch.from_numpy(gt),
            "case_index": index,
            "case_path": str(case_folder),
        }


def collate_liver_cases(batch: list[dict[str, object]]) -> dict[str, object]:
    source_shapes = {tuple(item["source_full"].shape) for item in batch}
    gt_shapes = {tuple(item["gt_full"].shape) for item in batch}
    if len(source_shapes) != 1 or source_shapes != gt_shapes:
        raise ValueError(
            "All raw point clouds in a batch must have the same shape; "
            f"source={source_shapes}, gt={gt_shapes}"
        )
    return {
        "source_full": torch.stack(
            [item["source_full"] for item in batch],
            dim=0,
        ),
        "gt_full": torch.stack(
            [item["gt_full"] for item in batch],
            dim=0,
        ),
        "case_index": torch.tensor(
            [item["case_index"] for item in batch],
            dtype=torch.long,
        ),
        "case_path": [item["case_path"] for item in batch],
    }


@torch.no_grad()
def farthest_point_sample_indices(
    points: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Deterministic batched FPS implemented without custom CUDA extensions."""
    batch_size, num_points, _ = points.shape
    sample_count = min(int(sample_count), num_points)
    if sample_count == num_points:
        return torch.arange(num_points, device=points.device).expand(
            batch_size,
            -1,
        )
    points_float = points.float()
    centroid = points_float.mean(dim=1, keepdim=True)
    farthest = (points_float - centroid).square().sum(dim=-1).argmax(dim=1)
    batch = torch.arange(batch_size, device=points.device)
    distances = torch.full(
        (batch_size, num_points),
        float("inf"),
        device=points.device,
        dtype=torch.float32,
    )
    selected = torch.empty(
        batch_size,
        sample_count,
        dtype=torch.long,
        device=points.device,
    )
    for sample_index in range(sample_count):
        selected[:, sample_index] = farthest
        center = points_float[batch, farthest].unsqueeze(1)
        squared_distance = (points_float - center).square().sum(dim=-1)
        distances = torch.minimum(distances, squared_distance)
        farthest = distances.argmax(dim=1)
    return selected


@torch.no_grad()
def sample_corresponding_points(
    source_full: torch.Tensor,
    gt_full: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = farthest_point_sample_indices(source_full, sample_count)
    batch = torch.arange(source_full.shape[0], device=source_full.device)[:, None]
    return source_full[batch, indices], gt_full[batch, indices]


def _seed_for_view(
    base_seed: int,
    epoch: int,
    case_index: int,
    view_index: int,
) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(epoch) * 97_409
        + int(case_index) * 9_973
        + int(view_index) * 991
    )
    return value % (2**63 - 1)


@torch.no_grad()
def _spread_centers(
    points: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    count = max(1, int(count))
    num_points = points.shape[0]
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    first = torch.randint(
        num_points,
        (1,),
        generator=generator,
        device=points.device,
    )
    selected[0] = first
    minimum_distance = (points - points[first]).square().sum(dim=-1)
    for index in range(1, count):
        selected[index] = minimum_distance.argmax()
        squared_distance = (
            points - points[selected[index]]
        ).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, squared_distance)
    return selected


@torch.no_grad()
def _crop_indices(
    points: torch.Tensor,
    crop_type: str,
    target_count: int,
    center_index: int,
    second_center_index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    target_count = max(2, min(int(target_count), points.shape[0]))
    center = points[center_index]
    if crop_type == "ball":
        score = (points - center).square().sum(dim=-1)
        return score.topk(target_count, largest=False).indices

    if crop_type == "multi_ball":
        second_center = points[second_center_index]
        first_score = (points - center).square().sum(dim=-1)
        second_score = (points - second_center).square().sum(dim=-1)
        score = torch.minimum(first_score, second_score)
        return score.topk(target_count, largest=False).indices

    normal = torch.randn(
        3,
        generator=generator,
        device=points.device,
        dtype=points.dtype,
    )
    normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-6)
    projection = (points - center) @ normal
    if crop_type == "plane":
        largest = bool(
            torch.randint(
                2,
                (1,),
                generator=generator,
                device=points.device,
            ).item()
        )
        return projection.topk(target_count, largest=largest).indices
    if crop_type == "slab":
        return projection.abs().topk(target_count, largest=False).indices
    raise ValueError(f"Unknown crop type: {crop_type}")


def _rotation_matrix(
    max_degrees: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radians = math.radians(float(max_degrees))
    angles = (
        torch.rand(3, generator=generator, device=device, dtype=dtype) * 2 - 1
    ) * radians
    x_angle, y_angle, z_angle = angles.unbind()
    one = torch.ones((), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    cx, sx = torch.cos(x_angle), torch.sin(x_angle)
    cy, sy = torch.cos(y_angle), torch.sin(y_angle)
    cz, sz = torch.cos(z_angle), torch.sin(z_angle)
    rotate_x = torch.stack(
        [one, zero, zero, zero, cx, -sx, zero, sx, cx]
    ).reshape(3, 3)
    rotate_y = torch.stack(
        [cy, zero, sy, zero, one, zero, -sy, zero, cy]
    ).reshape(3, 3)
    rotate_z = torch.stack(
        [cz, -sz, zero, sz, cz, zero, zero, zero, one]
    ).reshape(3, 3)
    return rotate_z @ rotate_y @ rotate_x


@torch.no_grad()
def build_partial_views(
    source: torch.Tensor,
    gt: torch.Tensor,
    case_indices: torch.Tensor,
    epoch: int,
    crops_per_gt: int,
    overlap_min: float,
    overlap_max: float,
    anchor_overlap: float,
    anchor_probability: float,
    seed: int,
    training: bool,
    rotation_degrees: float = 10.0,
    translation_mm: float = 2.0,
    scale_min: float = 0.98,
    scale_max: float = 1.02,
    partial_jitter_mm: float = 0.0,
    crop_types: tuple[str, ...] = CROP_TYPES,
) -> dict[str, object]:
    """Create multi-location, multi-overlap and multi-shape partial views."""
    if not crop_types:
        raise ValueError("At least one crop type is required")
    views_per_case = int(crops_per_gt) if training else 1
    source_views = []
    gt_views = []
    partial_views = []
    dense_partial_views = []
    observed_masks = []
    overlaps = []
    crop_names = []

    for batch_index in range(source.shape[0]):
        case_index = int(case_indices[batch_index].item())
        center_generator = torch.Generator(device=source.device)
        center_generator.manual_seed(
            _seed_for_view(seed, epoch if training else 0, case_index, 0)
        )
        centers = _spread_centers(
            gt[batch_index],
            max(views_per_case, 2),
            center_generator,
        )
        type_offset = int(
            torch.randint(
                len(crop_types),
                (1,),
                generator=center_generator,
                device=source.device,
            ).item()
        )

        for view_index in range(views_per_case):
            generator = torch.Generator(device=source.device)
            generator.manual_seed(
                _seed_for_view(
                    seed,
                    epoch if training else 0,
                    case_index,
                    view_index,
                )
            )
            if training and float(
                torch.rand((), generator=generator, device=source.device)
            ) >= anchor_probability:
                overlap = overlap_min + float(
                    torch.rand((), generator=generator, device=source.device)
                ) * (overlap_max - overlap_min)
            else:
                overlap = anchor_overlap
            target_count = int(round(overlap * gt.shape[1]))
            crop_type = crop_types[
                (type_offset + view_index) % len(crop_types)
            ]
            center_index = int(centers[view_index].item())
            second_center_index = int(
                centers[(view_index + 1) % len(centers)].item()
            )
            partial_indices = _crop_indices(
                gt[batch_index],
                crop_type,
                target_count,
                center_index,
                second_center_index,
                generator,
            )
            source_view = source[batch_index].clone()
            gt_view = gt[batch_index].clone()
            partial_view = gt[batch_index, partial_indices].clone()

            if training:
                rotation = _rotation_matrix(
                    rotation_degrees,
                    generator,
                    source.device,
                    source.dtype,
                )
                scale = scale_min + float(
                    torch.rand((), generator=generator, device=source.device)
                ) * (scale_max - scale_min)
                translation = (
                    torch.rand(
                        3,
                        generator=generator,
                        device=source.device,
                        dtype=source.dtype,
                    )
                    * 2
                    - 1
                ) * translation_mm
                source_view = (source_view @ rotation.T) * scale + translation
                gt_view = (gt_view @ rotation.T) * scale + translation
                partial_view = (partial_view @ rotation.T) * scale + translation
                if partial_jitter_mm > 0:
                    partial_view = partial_view + torch.randn(
                        partial_view.shape,
                        generator=generator,
                        device=source.device,
                        dtype=source.dtype,
                    ) * partial_jitter_mm

            # Preserve the exact source/GT index correspondence. Missing
            # entries stay zero and are ignored according to observed_mask.
            dense_partial_view = torch.zeros_like(gt_view)
            dense_partial_view[partial_indices] = partial_view

            observed_mask = torch.zeros(
                gt.shape[1],
                dtype=torch.bool,
                device=source.device,
            )
            observed_mask[partial_indices] = True
            source_views.append(source_view)
            gt_views.append(gt_view)
            partial_views.append(partial_view)
            dense_partial_views.append(dense_partial_view)
            observed_masks.append(observed_mask)
            overlaps.append(overlap)
            crop_names.append(crop_type)

    max_partial_points = max(partial.shape[0] for partial in partial_views)
    padded_partial = source.new_zeros(
        len(partial_views),
        max_partial_points,
        3,
    )
    partial_mask = torch.zeros(
        len(partial_views),
        max_partial_points,
        dtype=torch.bool,
        device=source.device,
    )
    for index, partial in enumerate(partial_views):
        padded_partial[index, : partial.shape[0]] = partial
        partial_mask[index, : partial.shape[0]] = True

    return {
        "source_xyz": torch.stack(source_views),
        "gt_xyz": torch.stack(gt_views),
        "partial_xyz": padded_partial,
        "partial_mask": partial_mask,
        "partial_dense_xyz": torch.stack(dense_partial_views),
        "observed_mask": torch.stack(observed_masks),
        "overlap": torch.tensor(
            overlaps,
            device=source.device,
            dtype=torch.float32,
        ),
        "crop_type": crop_names,
    }
