#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname -- "${SCRIPT_DIR}")}"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

GPU_ID="${GPU_ID:-1}"
DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATA_OVERLAP="${DATA_OVERLAP:-0.25}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:--1}"

COMPLETION_CKPT="${COMPLETION_CKPT:-${PROJECT_DIR}/completion/logs/full_aug_20260805_013524/best.pth}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[Error] set CHECKPOINT to an existing V3 checkpoint" >&2
  echo "Set CHECKPOINT=/path/to/best.pth and rerun." >&2
  exit 1
fi
if [[ ! -f "${COMPLETION_CKPT}" ]]; then
  echo "[Error] completion checkpoint not found: ${COMPLETION_CKPT}" >&2
  exit 1
fi
RUN_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/test_results/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[Info] CHECKPOINT=${CHECKPOINT}"
echo "[Info] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Info] COMPLETION_CKPT=${COMPLETION_CKPT}"

python -m liver2.evaluation.test_pipeline \
  --dataset_root "${DATASET_ROOT}" \
  --data_overlap "${DATA_OVERLAP}" \
  --max_test_samples "${MAX_TEST_SAMPLES}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --device "cuda:0" \
  --completion_checkpoint "${COMPLETION_CKPT}" \
  "$@"
