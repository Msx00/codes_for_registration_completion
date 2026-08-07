# ----------------------------- Collate（分词 + 张量化） -----------------------------

import math
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm
from scipy.spatial import KDTree
from collections import deque
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm

# --- 确保这些导入存在 ---
from scipy.spatial import KDTree
from collections import deque
# ------------------------

def normalize_pointcloud(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    scale = np.max(distances)

    if scale < 1e-6:
        normalized_points = centered_points
        scale = 1.0
    else:
        normalized_points = centered_points / scale

    return normalized_points, centroid, scale

def normalize_pointcloud_withscale(points, centroid, scale):
    centered_points = points - centroid
    normalized_points = centered_points / scale

    return normalized_points

def generate_transform_matrix(
    tx=None, ty=None, tz=None,  # 平移量(mm), None则随机0-20
    theta_deg=None,             # 旋转角(度), None则随机0-30
    axis='z'                    # 旋转轴: 'x'/'y'/'z'
):
    # 1. 随机生成平移量(0-20mm)或使用指定值
    tx = np.random.uniform(0, 20) if tx is None else tx
    ty = np.random.uniform(0, 20) if ty is None else ty
    tz = np.random.uniform(0, 20) if tz is None else tz

    # 2. 随机生成旋转角(0-30°)并转换为弧度
    theta_deg = np.random.uniform(0, 20) if theta_deg is None else theta_deg
    theta = math.radians(theta_deg)
    cosθ, sinθ = math.cos(theta), math.sin(theta)

    # 3. 构建旋转子矩阵(3×3)
    if axis == 'x':
        rotate_mat = np.array([
            [1, 0, 0],
            [0, cosθ, -sinθ],
            [0, sinθ, cosθ]
        ])
    elif axis == 'y':
        rotate_mat = np.array([
            [cosθ, 0, sinθ],
            [0, 1, 0],
            [-sinθ, 0, cosθ]
        ])
    elif axis == 'z':  # 默认绕Z轴
        rotate_mat = np.array([
            [cosθ, -sinθ, 0],
            [sinθ, cosθ, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("旋转轴仅支持 'x'/'y'/'z'")

    # # 4. 构建4×4齐次变换矩阵(先旋转后平移)
    # transform_mat = np.eye(4)  # 初始化单位矩阵
    # transform_mat[:3, :3] = rotate_mat  # 上左3×3为旋转矩阵
    # transform_mat[:3, 3] = [tx, ty, tz]  # 上右3列为平移量

    return rotate_mat, theta_deg, (tx, ty, tz)

# 定义数据集
class LiverCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, root, subfolders, args):
        super().__init__()
        self.root = Path(root)
        self.subfolders = subfolders
        self.cases = {"train": [], "validation": [], "test": []}
        self.data_overlap = args.data_overlap  # 定义数据重叠比例
        self.seed = int(getattr(args, "seed", 42))
        self.train_crops_per_gt = int(
            getattr(args, "train_crops_per_gt", 1)
        )
        if self.train_crops_per_gt < 1:
            raise ValueError("train_crops_per_gt must be at least 1")
        self.epoch = 0
        self.num_template_points = int(
            getattr(args, "num_template_points", 2048)
        )
        # Cache deterministic FPS indices so each case is sampled only once
        # per dataset process instead of recomputing FPS every epoch.
        self._fps_index_cache = {}
        self._train_crop_start_cache = {}

        # 遍历SUBROOT和SUBSUBROOT中的每个文件夹
        for subfolder in self.subfolders:
            case_folder = self.root / subfolder
            if case_folder.is_dir():
                self._find_case_folders(case_folder, subfolder)

        max_train_samples = getattr(args, "max_train_samples", -1)
        dataset_seed = self.seed
        if (
            max_train_samples is not None
            and max_train_samples > 0
            and len(self.cases["train"]) > max_train_samples
        ):
            old_count = len(self.cases["train"])
            # 每个 DDP 进程都使用相同 seed，确保选择相同的训练子集
            subset_rng = np.random.default_rng(dataset_seed)
            selected_indices = subset_rng.choice(
                old_count,
                size=max_train_samples,
                replace=False,
            )
            self.cases["train"] = [
                self.cases["train"][int(i)]
                for i in selected_indices
            ]
            print(
                f"Randomly limited train cases: "
                f"{old_count} -> {len(self.cases['train'])}"
            )
            if old_count != len(self.cases["train"]):
                print(f"Limited train cases: {old_count} -> {len(self.cases['train'])}")

        max_val_samples = getattr(args, "max_val_samples", -1)

        if (
            max_val_samples is not None
            and max_val_samples > 0
            and len(self.cases["validation"]) > max_val_samples
        ):
            old_count = len(self.cases["validation"])

            subset_rng = np.random.default_rng(dataset_seed + 1)

            selected_indices = subset_rng.choice(
                old_count,
                size=max_val_samples,
                replace=False,
            )

            self.cases["validation"] = [
                self.cases["validation"][int(i)]
                for i in selected_indices
            ]

            print(
                f"Randomly limited validation cases: "
                f"{old_count} -> {len(self.cases['validation'])}"
            )
        max_test_samples = getattr(args, "max_test_samples", -1)

        if (
            max_test_samples is not None
            and max_test_samples > 0
            and len(self.cases["test"]) > max_test_samples
        ):
            old_count = len(self.cases["test"])

            subset_rng = np.random.default_rng(dataset_seed + 2)

            selected_indices = subset_rng.choice(
                old_count,
                size=max_test_samples,
                replace=False,
            )

            self.cases["test"] = [
                self.cases["test"][int(i)]
                for i in selected_indices
            ]

            print(
                f"Randomly limited test cases: "
                f"{old_count} -> {len(self.cases['test'])}"
            )

        assert len(self.cases["train"]) + len(self.cases["validation"]) + len(self.cases["test"]) > 0, f"No cases found under {root}"
        print(f"Total cases found in train: {len(self.cases['train'])}")
        print(f"Total cases found in validation: {len(self.cases['validation'])}")
        print(f"Total cases found in test: {len(self.cases['test'])}")
        if self.cases["train"]:
            print(
                "Effective train samples per epoch: "
                f"{len(self.cases['train'])} cases x "
                f"{self.train_crops_per_gt} crops = "
                f"{len(self.cases['train']) * self.train_crops_per_gt}"
            )

    def __len__(self):
        return sum(
            len(cases) * (
                self.train_crops_per_gt if split == "train" else 1
            )
            for split, cases in self.cases.items()
        )

    def set_epoch(self, epoch):
        """Select a new reproducible set of training crop positions."""
        epoch = int(epoch)
        if epoch != self.epoch:
            self._train_crop_start_cache.clear()
        self.epoch = epoch

    @staticmethod
    def _spread_random_start_indices(point_cloud, count, random_seed):
        """Choose random-but-spatially-spread crop centers.

        The first center is random. Later centers use farthest-point sampling
        relative to the already selected centers, so multiple views of one GT
        are unlikely to crop the same anatomical region.
        """
        points = np.asarray(point_cloud, dtype=np.float32)
        num_points = len(points)
        count = int(count)
        if num_points == 0 or count < 1:
            return np.empty((0,), dtype=np.int64)

        rng = np.random.default_rng(random_seed)
        unique_count = min(count, num_points)
        selected = np.empty(unique_count, dtype=np.int64)
        selected_mask = np.zeros(num_points, dtype=bool)
        min_squared_distance = np.full(
            num_points,
            np.inf,
            dtype=np.float32,
        )
        farthest = int(rng.integers(0, num_points))

        for i in range(unique_count):
            selected[i] = farthest
            selected_mask[farthest] = True
            squared_distance = np.sum(
                (points - points[farthest]) ** 2,
                axis=1,
            )
            np.minimum(
                min_squared_distance,
                squared_distance,
                out=min_squared_distance,
            )
            min_squared_distance[selected_mask] = -1.0
            farthest = int(np.argmax(min_squared_distance))

        if count == unique_count:
            return selected

        repeated = rng.choice(
            selected,
            size=count - unique_count,
            replace=True,
        )
        return np.concatenate([selected, repeated.astype(np.int64)])

    @staticmethod
    def _farthest_point_sample_indices(point_cloud, sample_count):
        """Return deterministic FPS indices for an (N, 3) point cloud.

        The first point is chosen as the point farthest from the cloud
        centroid. All later points maximize their minimum squared distance to
        the already selected set. No random number generator is used.
        """
        points = np.asarray(point_cloud, dtype=np.float32)
        num_points = points.shape[0]
        sample_count = min(int(sample_count), num_points)

        if sample_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if sample_count == num_points:
            return np.arange(num_points, dtype=np.int64)

        selected = np.empty(sample_count, dtype=np.int64)
        selected_mask = np.zeros(num_points, dtype=bool)
        min_squared_distance = np.full(
            num_points,
            np.inf,
            dtype=np.float32,
        )

        centroid = points.mean(axis=0, keepdims=True)
        squared_to_centroid = np.sum((points - centroid) ** 2, axis=1)
        farthest = int(np.argmax(squared_to_centroid))

        for i in range(sample_count):
            selected[i] = farthest
            selected_mask[farthest] = True

            squared_distance = np.sum(
                (points - points[farthest]) ** 2,
                axis=1,
            )
            np.minimum(
                min_squared_distance,
                squared_distance,
                out=min_squared_distance,
            )
            min_squared_distance[selected_mask] = -1.0
            farthest = int(np.argmax(min_squared_distance))

        return selected

    # --- 1. 新增的 _extract_contiguous_points 函数 ---
    @staticmethod
    def _extract_contiguous_points(
        point_cloud,
        target_size=1024,
        k_neighbors=16,
        random_seed=0,
        start_index=None,
    ):
        """Extract an exact-size contiguous subset without duplicate points."""
        random_state = np.random.RandomState(seed=random_seed)
        num_points = len(point_cloud)
        if num_points == 0:
            return np.empty((0, point_cloud.shape[1]), dtype=point_cloud.dtype)

        target_size = max(1, min(int(target_size), num_points))
        if target_size == num_points:
            return point_cloud.copy()

        try:
            kdtree = KDTree(point_cloud)
        except ValueError as exc:
            print(
                "Warning: KDTree creation failed "
                f"(shape: {point_cloud.shape}): {exc}. "
                "Returning a unique random subset."
            )
            indices = random_state.choice(
                num_points,
                size=target_size,
                replace=False,
            )
            return point_cloud[indices]

        visited = np.zeros(num_points, dtype=bool)
        selected_indices = []
        if start_index is None:
            seed_idx = int(random_state.randint(0, num_points))
        else:
            seed_idx = int(start_index)
            if not 0 <= seed_idx < num_points:
                raise ValueError(
                    f"start_index {seed_idx} is outside [0, {num_points})"
                )
        queue = deque([seed_idx])
        visited[seed_idx] = True
        query_k = min(max(1, int(k_neighbors)), num_points)

        while queue and len(selected_indices) < target_size:
            current_idx = queue.popleft()
            selected_indices.append(current_idx)

            try:
                _, neighbor_indices = kdtree.query(
                    point_cloud[current_idx],
                    k=query_k,
                )
            except ValueError as exc:
                print(
                    f"Warning: KDTree query failed for index {current_idx}: "
                    f"{exc}"
                )
                continue

            neighbor_indices = np.atleast_1d(neighbor_indices)
            for neighbor_idx in neighbor_indices:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx >= num_points:
                    continue
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

        # A k-NN graph can occasionally be disconnected. Fill the remainder
        # from unvisited points, still without replacement, to keep batch
        # shapes exact and avoid duplicated partial points.
        if len(selected_indices) < target_size:
            remaining = np.flatnonzero(~visited)
            needed = target_size - len(selected_indices)
            fill = random_state.choice(remaining, size=needed, replace=False)
            selected_indices.extend(int(i) for i in fill)

        return point_cloud[np.asarray(selected_indices, dtype=np.int64)]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        for split, cases in self.cases.items():
            crops_per_case = (
                self.train_crops_per_gt if split == "train" else 1
            )
            split_sample_count = len(cases) * crops_per_case
            if idx < split_sample_count:
                case_idx, crop_view_idx = divmod(idx, crops_per_case)
                case_folder = cases[case_idx]
                
                # --- 2. 您的 __getitem__ 逻辑 ---
                gt_full = np.loadtxt(case_folder / "gt.txt")
                source_full = np.loadtxt(case_folder / "source.txt")
                
                # source 与 gt 必须使用相同索引，保持逐点对应关系。
                if len(source_full) != len(gt_full):
                    raise ValueError(
                        f"source/gt point counts differ in {case_folder}: "
                        f"{len(source_full)} vs {len(gt_full)}"
                    )

                sample_count = min(
                    self.num_template_points,
                    len(source_full),
                )
                if len(source_full) > sample_count:
                    # Train/validation/test all use the same deterministic FPS
                    # rule on source_full. Apply exactly the same indices to
                    # gt_full to preserve source_i <-> gt_i correspondence.
                    cache_key = (str(case_folder), sample_count)
                    indices = self._fps_index_cache.get(cache_key)
                    if indices is None:
                        indices = self._farthest_point_sample_indices(
                            source_full,
                            sample_count,
                        )
                        self._fps_index_cache[cache_key] = indices

                    source = source_full[indices]
                    gt = gt_full[indices]
                else:
                    source = source_full.copy()
                    gt = gt_full.copy()

                # (加载其余文件)
                abdominal_f0 = np.loadtxt(case_folder / "abdominal_f0.txt")
                abdominal_f1 = np.loadtxt(case_folder / "abdominal_f1.txt")
                tumor_f0 = np.loadtxt(case_folder / "tumor_f0.txt")
                tumor_f1 = np.loadtxt(case_folder / "tumor_f1.txt")

                statistics = yaml.safe_load((case_folder / "statistics.yaml").read_text(encoding="utf-8"))
                E_kPa = 1e-3 * float(statistics.get("SofaSimulationBlock_young_modulus", 12.0))
                nu = float(statistics.get("SofaSimulationBlock_poisson_ratio", 0.49))
                text = f"organ: liver; Young's modulus {E_kPa:.2f} kPa; poisson ratio {nu:.2f}"
                
                # --- 3. 调用新函数从 gt 裁剪 target ---
                # (不再加载 target.txt)
                target_size = max(
                    1,
                    int(round(self.data_overlap * len(gt))),
                )

                if split == "train":
                    # Each virtual view uses a different seed point. The set
                    # of spatially spread positions is reproducible and
                    # changes every epoch.
                    crop_start_indices = self._train_crop_start_cache.get(
                        case_idx
                    )
                    if crop_start_indices is None:
                        position_seed = int(
                            np.random.SeedSequence(
                                [self.seed, self.epoch, case_idx]
                            ).generate_state(1, dtype=np.uint32)[0]
                        )
                        crop_start_indices = (
                            self._spread_random_start_indices(
                                gt,
                                self.train_crops_per_gt,
                                position_seed,
                            )
                        )
                        self._train_crop_start_cache[case_idx] = (
                            crop_start_indices
                        )
                    crop_start_index = int(
                        crop_start_indices[crop_view_idx]
                    )
                    partial_seed = int(
                        np.random.SeedSequence(
                            [
                                self.seed,
                                self.epoch,
                                case_idx,
                                crop_view_idx,
                            ]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                else:
                    # 验证集必须固定，否则每个 epoch 的验证对象都不一样
                    partial_seed = 100000 + case_idx
                    crop_start_index = None

                target = self._extract_contiguous_points(
                    gt,
                    target_size=target_size,
                    k_neighbors=16,
                    random_seed=partial_seed,
                    start_index=crop_start_index,
                )
                # print(f"Extracted target shape: {target.shape} from gt shape: {gt.shape}")
                # -------------------------------------

                save_target_ply = False
                if save_target_ply:
                    # 定义输出文件名，例如 "target_cut_s0_idx0.ply"
                    # 使用 self.seed 和 idx 确保文件名唯一且可复现
                    ply_filename = f"target_cut_idx{idx}.ply"
                    output_ply_path = str(ply_filename) 
                    
                    import open3d as o3d
                    
                    try:
                        # 1. 创建 Open3D 点云对象
                        pcd = o3d.geometry.PointCloud()
                        # 2. 从 numpy 数组设置点
                        pcd.points = o3d.utility.Vector3dVector(target)
                        # 3. 写入 PLY 文件
                        o3d.io.write_point_cloud(output_ply_path, pcd, write_ascii=True)
                    except Exception as e:
                        print(f"Error saving PLY file {output_ply_path}: {e}")
                    
                    break
                
                # source_n, _, _ = normalize_pointcloud(source)
                # gt_n, gt_centroid, gt_scale = normalize_pointcloud(gt)
                # target_n = normalize_pointcloud_withscale(target, gt_centroid, gt_scale)

                # rot, degree, trans = generate_transform_matrix()
                # source = source @ rot.T + trans  # (M,3)

                return {
                    "src_xyz": source, "part_xyz": target, "gt_xyz": gt,
                    "E_kPa": E_kPa, "nu": nu, "text": text,
                    "split": split,
                    "abdominal_f0":abdominal_f0, "abdominal_f1":abdominal_f1,
                    "tumor_f0":tumor_f0, "tumor_f1":tumor_f1
                }
            
            idx -= split_sample_count

        raise IndexError(idx)

    # --- 4. 您的 _find_case_folders 函数 ---
    def _find_case_folders(self, root_folder, split):
        # (这是一个递归实现，确保 'split' 被正确传递)
        for subdir in sorted(root_folder.iterdir()):
            if subdir.is_dir():
                # 检查当前目录是否是一个 "case" 目录
                if (subdir / "gt.txt").exists() and (subdir / "source.txt").exists():
                    self.cases[split].append(subdir)
                else:
                    # 如果不是 case 目录，则递归进入子目录
                    self._find_case_folders(subdir, split)

# Collate function
def collate_fn(batch, tokenizer, src_max_n=None):
    B = len(batch)

    def to_tensor(pc):
        return torch.from_numpy(pc)

    src_list, part_list, gt_list = [], [], []
    texts, E_list, nu_list = [], [], []

    for item in batch:
        src = to_tensor(item["src_xyz"])
        part = to_tensor(item["part_xyz"])
        gt = to_tensor(item["gt_xyz"])

        if src_max_n is not None:
            n = src.shape[0]
            if n >= src_max_n:
                idx = torch.randperm(n)[:src_max_n]
            else:
                pad_idx = torch.randint(0, n, (src_max_n - n,))
                idx = torch.cat([torch.arange(n), pad_idx])
                idx = idx[torch.randperm(src_max_n)]
            src = src[idx]; gt = gt[idx]
        if part.shape[0] == 0:
            raise ValueError("part_xyz contains zero points")

        src_list.append(src)
        part_list.append(part)
        gt_list.append(gt)

        texts.append(item["text"])
        E_list.append([item["E_kPa"]])
        nu_list.append([item["nu"]])
        
    # Do not repeat the observed partial cloud to 1024/2048 points.
    # Normally every sample already has round(data_overlap * 2048) points
    # (410 points for overlap=0.2). If a rare batch contains different source
    # sizes, crop all partials to the smallest real size without replacement.
    part_n = min(part.shape[0] for part in part_list)
    for i, part in enumerate(part_list):
        if part.shape[0] > part_n:
            # Evenly spaced deterministic indices avoid introducing new
            # validation randomness while preserving unique points.
            idx = torch.linspace(
                0,
                part.shape[0] - 1,
                steps=part_n,
            ).round().long()
            part_list[i] = part[idx]

    src_tensor  = torch.stack(src_list, dim=0).float()
    part_tensor = torch.stack(part_list, dim=0).float()
    gt_tensor  = torch.stack(gt_list, dim=0).float()

    encoded = tokenizer(
        texts, add_special_tokens=True, max_length=64,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    attn_mask = encoded['attention_mask']

    E = torch.tensor(E_list, dtype=torch.float32)
    nu = torch.tensor(nu_list, dtype=torch.float32)

    return {
        "src_xyz": src_tensor, "part_xyz": part_tensor, "gt_xyz": gt_tensor,
        "input_ids": input_ids, "attn_mask": attn_mask,
        "E_kPa": E, "nu": nu
    }


# 创建DataLoader
def build_tokenizer(local_files=True):
    from transformers import BertTokenizer
    bert_dir = Path(__file__).resolve().parent / 'bert-base-uncased'
    tok = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=local_files)
    return tok


def option():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default="/home/ma_sx/Project/Dataset/MedShapeNet-Liver")
    ap.add_argument('--save_dir', type=str, default='./logs/exp')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-7)
    ap.add_argument('--weight_decay', type=float, default=1e-7)
    ap.add_argument('--num_workers', type=int, default=4)
    # ap.add_argument('--src_max_n', type=int, default=2048, help='源点云统一采样数')
    ap.add_argument('--use_text', action='store_true')
    ap.add_argument('--use_numeric_biomech', default=False, action='store_true')
    ap.add_argument('--bert_local_files_only', action='store_true')
    ap.add_argument('--device', type=str, default='cuda:0')
    # 损失权重
    ap.add_argument('--w_sup', type=float, default=1e-2)
    ap.add_argument('--w_part', type=float, default=1.0)
    ap.add_argument('--w_phys', type=float, default=0.2)
    ap.add_argument('--alpha_dcd', type=float, default=1.0)
    ap.add_argument('--phys_k', type=int, default=16)
    ap.add_argument('--phys_reg', type=float, default=1e-4)
    # 训练细节
    ap.add_argument('--eval_split_ratio', type=float, default=0.1)
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    # 实例化数据集和DataLoader
    args = option()
    dataset_root = args.dataset_root
    T_dataset = LiverCompletionDataset(dataset_root, ["train"], args)
    T_validation = LiverCompletionDataset(dataset_root, ["validation"], args)
    T_test = LiverCompletionDataset(dataset_root, ["test"], args)

    tokenizer = build_tokenizer()
    text = "Hello, how are you?"
    encoded_input = tokenizer(text)

    print(f"Encoded input: {encoded_input}")
    print("Tokenizer loaded.")

    train_loader = DataLoader(
        T_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    validation_loader = DataLoader(
        T_validation,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    test_loader = DataLoader(
        T_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    # 示例
    for data in train_loader:
        print(f"Batch data (Train): {data['src_xyz'].shape}, {data['gt_xyz'].shape}, {data['part_xyz'].shape}")
        break

    # for data in validation_loader:
    #     print(f"Batch data (Validation): {data['src_xyz'].shape}, {data['gt_xyz'].shape}, {data['part_xyz'].shape}")
    #     break

    # for data in test_loader:
    #     print(f"Batch data (Test): {data['src_xyz'].shape}, {data['gt_xyz'].shape}, {data['part_xyz'].shape}")
    #     break
