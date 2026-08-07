#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ma_sx/Project/Liver2"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

GPU_ID="${GPU_ID:-1}"
DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATA_OVERLAP="${DATA_OVERLAP:-0.25}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-200}"

BERT_MODEL="${BERT_MODEL:-${PROJECT_DIR}/bert-base-uncased}"
LEGACY_PIVOTS_CKPT="${LEGACY_PIVOTS_CKPT:-${PROJECT_DIR}/PIVOTS/checkpoints/pivots_v5/0/best_model.pth}"
COMPLETION_CKPT="${COMPLETION_CKPT:-${PROJECT_DIR}/completion/logs/full_aug_20260805_013524/best.pth}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/spaq_GIRNet_20260807_113931/best.pth}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[Error] checkpoint not found: ${CHECKPOINT}" >&2
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
export TOKENIZERS_PARALLELISM=false

echo "[Info] CHECKPOINT=${CHECKPOINT}"
echo "[Info] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Info] COMPLETION_CKPT=${COMPLETION_CKPT}"

python test_pivots_text.py \
  --dataset_root "${DATASET_ROOT}" \
  --data_overlap "${DATA_OVERLAP}" \
  --max_test_samples "${MAX_TEST_SAMPLES}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --device "cuda:0" \
  --bert_model_name "${BERT_MODEL}" \
  --bert_local_files_only \
  --completion_checkpoint "${COMPLETION_CKPT}" \
  "$@"
