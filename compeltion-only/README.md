# Liver Completion Only

该目录只训练点云补全网络，不创建或运行配准网络、PIVOTS、BERT 和物理损失。

## 默认训练方案

- 启动脚本当前默认使用 100 个 train case、10 个 validation case，便于先检查训练；设置 `MAX_TRAIN_CASES=-1 MAX_VAL_CASES=-1` 才是完整数据集。
- 每个 case 的 10,000 点 source/GT 只加载一次，在 GPU 上做一次对应 FPS，随后展开 4 个 partial 视图。
- 每个 GT 每个 epoch 覆盖 4 种裁剪形状：局部球形、平面端部、平面薄层和双区域裁剪。
- 50% partial 固定 overlap=0.25，其余在 0.15～0.40 随机采样。
- source、GT、partial 接受一致的旋转、平移和尺度增强。默认关闭 partial 点噪声；可用 `PARTIAL_JITTER_MM` 显式开启。
- 验证 partial 固定，不随 epoch 改变。
- 前 10 个 epoch 使用增强 curriculum：从接近 overlap=0.25 的球形裁剪和弱几何扰动，逐步扩展到完整 overlap、4 种裁剪形状和完整扰动强度。

## 网络

默认 `ARCHITECTURE=generative` 使用 `SplAttN/models/liver_generative_completion.py`：

1. source 局部图编码；
2. partial 上下文编码；
3. source/partial 特征和全局特征融合；
4. learned queries 直接生成 256 个绝对 coarse 坐标；
5. coarse 与 partial seed 融合，生成 1024 个 mid 点；
6. coarse-to-fine expansion 生成 2048 个完成点。

source 只提供 case-specific 形状特征和归一化坐标系，不作为输出模板；最终输出不是 `source + DDF`，也不保证与 source/GT 同索引。

旧的稠密位移版本仍可用 `ARCHITECTURE=displacement` 启动。两种结构的 checkpoint 不可互相续训。

## 损失

生成式网络默认总损失为：

```text
0.25 * coarse FP32 symmetric Chamfer
+ 0.50 * mid FP32 symmetric Chamfer
+ 1.00 * fine FP32 symmetric Chamfer
+ 0.10 * multi-stage OAReg-style FP32 correntropy Chamfer
+ 0.10 * FP32 partial coverage
```

生成点没有 GT 对应索引，因此不使用对应点 MSE/Huber。训练和 checkpoint 选择使用 permutation-invariant Chamfer；日志中的 `chamfer_rmse` 定义为 symmetric Chamfer squared distance 的平方根。

OA correntropy 项按 `L1 KNN → truncation=0.2 → exp(-distance/sigma²) → negative bidirectional correlation` 计算，默认 `sigma²=1.0`。训练中加 2 将其平移为非负 penalty；这只改变显示数值，不改变梯度。可用 `W_OA_CORRENTROPY`、`OA_SIGMA2`、`OA_TRUNCATION` 和 `OA_NORM` 调整。标准 Chamfer 始终保留，因为被 OA 阈值截断的大误差点没有梯度，而它们通常正是补全初期最困难的 missing 点。

所有 Chamfer/cdist 均强制 FP32，不受 AMP FP16 影响。

## 启动

```bash
bash /home/ma_sx/Project/Liver2/compeltion-only/run_train_multigpu.sh
```

完整训练集：

```bash
MAX_TRAIN_CASES=-1 MAX_VAL_CASES=-1 \
  bash /home/ma_sx/Project/Liver2/compeltion-only/run_train_multigpu.sh
```

快速小数据检查：

```bash
MAX_TRAIN_CASES=8 MAX_VAL_CASES=4 EPOCHS=2 WORLD_SIZE=1 GPU_IDS=0 \
  bash /home/ma_sx/Project/Liver2/compeltion-only/run_train_multigpu.sh
```

## Chamfer 与 OAReg 风格 correntropy 消融

推荐先使用默认标准 Chamfer：

```bash
SET_LOSS_MODE=chamfer bash run_train_multigpu.sh
```

OAReg 风格的鲁棒 correntropy 集合项：

```bash
SET_LOSS_MODE=correntropy bash run_train_multigpu.sh
```

二者各占一半：

```bash
SET_LOSS_MODE=hybrid bash run_train_multigpu.sh
```

生成式分支三组实验都不使用对应点 Huber，模型按 validation Chamfer RMSE（毫米）选择 `best.pth`。

OAReg 面向无监督遮挡配准，会降低或截断大残差的梯度。补全任务的大残差通常正是需要学习的缺失区域，因此默认不启用 correntropy 截断；只有明确做消融时才建议设置 `--correntropy_truncation`。

## 输出

每次训练目录包含：

- `config.json`：完整配置；
- `metrics.jsonl`：每个 epoch 的训练/验证指标；
- `best.pth`：最低 validation completion RMSE；
- `last.pth`：最近 epoch checkpoint。

## 推理

指定 checkpoint 的默认推理入口：

```bash
bash /home/ma_sx/Project/Liver2/compeltion-only/run_inference.sh
```

显式指定推理输出目录：

```bash
OUTPUT_DIR=/path/to/inference_result bash run_inference.sh
```

脚本会自动识别 legacy、aligned-observed 和 generative checkpoint。默认读取 checkpoint 保存的 validation 数量限制、使用 overlap=0.25，并以 ASCII PLY 保存 `completed/source/partial/gt` 点云，同时生成带颜色的 `completed_vs_gt.ply`（completed 红色、GT 绿色），以及逐 case 指标和汇总指标。`cases.jsonl` 对每个 case/view 同时记录 `pointwise_rmse_mm`、`symmetric_chamfer_rmse_mm` 和 `mae_mm`；`summary.json` 同时记录两种 pooled RMSE、两种 case 平均 RMSE、pooled MAE 和 case 平均 MAE。生成式 checkpoint 的主指标 `rmse_mm` 仍指 symmetric Chamfer RMSE，旧对应点模型的主指标指逐点 RMSE。

对完整 validation split 推理：

```bash
MAX_CASES=-1 bash run_inference.sh
```

旧版 checkpoint 可在推理后处理中固定 observed 点：

```bash
ANCHOR_OBSERVED=1 bash run_inference.sh
```

该后处理会自然降低 overall RMSE；比较网络真实补全能力时应主要查看 `missing_rmse_mm`。
