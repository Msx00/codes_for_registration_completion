#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

CHECKPOINT="${CHECKPOINT:-/home/ma_sx/Project/Liver2/compeltion-only/logs/full_aug_20260805_013524/best.pth}"
DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/inference_outputs/inference_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-validation}"
MAX_CASES="${MAX_CASES:-0}"
OVERLAP="${OVERLAP:-0.25}"
CROPS_PER_CASE="${CROPS_PER_CASE:-1}"
GPU_ID="${GPU_ID:-1}"
AMP="${AMP:-fp16}"
SAVE_POINTS="${SAVE_POINTS:-1}"
ANCHOR_OBSERVED="${ANCHOR_OBSERVED:-0}"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SCRIPT_DIR}/SplAttN${PYTHONPATH:+:${PYTHONPATH}}"

INFER_ARGS=(
  infer_completion.py
  --checkpoint "${CHECKPOINT}"
  --dataset_root "${DATASET_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --max_cases "${MAX_CASES}"
  --overlap "${OVERLAP}"
  --crops_per_case "${CROPS_PER_CASE}"
  --crop_types ball
  --amp "${AMP}"
)
if [[ "${SAVE_POINTS}" == "1" ]]; then
  INFER_ARGS+=(--save_points)
fi
if [[ "${ANCHOR_OBSERVED}" == "1" ]]; then
  INFER_ARGS+=(--anchor_observed)
fi
INFER_ARGS+=("$@")

python "${INFER_ARGS[@]}"
