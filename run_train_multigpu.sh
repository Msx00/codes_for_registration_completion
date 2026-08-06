#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ma_sx/Project/Liver2"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

GPU_IDS="${GPU_IDS:-0,1,3}"
WORLD_SIZE="${WORLD_SIZE:-3}"

DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/logs/spaq_pivots_$(date +%Y%m%d_%H%M%S)}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-600}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# --- Training stage ---
TRAIN_STAGE="${TRAIN_STAGE:-registration}"
REGISTRATION_TARGET_MODE="${REGISTRATION_TARGET_MODE:-gt}"

# --- PIVOTS architecture ---
PIVOTS_ARCH="${PIVOTS_ARCH:-full2full_v1}"
GLOBAL_MATCH_LEVEL="${GLOBAL_MATCH_LEVEL:-2}"
GLOBAL_MATCH_TEMPERATURE="${GLOBAL_MATCH_TEMPERATURE:-0.1}"
GLOBAL_MATCH_DIM="${GLOBAL_MATCH_DIM:-64}"
GLOBAL_SPATIAL_SIGMA="${GLOBAL_SPATIAL_SIGMA:-0.2}"
MAX_COARSE_FLOW_NORMALIZED="${MAX_COARSE_FLOW_NORMALIZED:-0.25}"
NUM_REFINEMENT_STEPS="${NUM_REFINEMENT_STEPS:-3}"
REFINEMENT_K="${REFINEMENT_K:-35}"
INITIALIZE_FROM_LEGACY_PIVOTS="${INITIALIZE_FROM_LEGACY_PIVOTS:-0}"
STRICT_PIVOTS_CHECKPOINT="${STRICT_PIVOTS_CHECKPOINT:-1}"
DEBUG_REFINEMENT="${DEBUG_REFINEMENT:-0}"

# --- Learning rates ---
LR="${LR:-1e-5}"
COMPLETION_LR="${COMPLETION_LR:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

# --- Loss weights ---
W_REG_HUBER="${W_REG_HUBER:-1.0}"
HUBER_BETA_MM="${HUBER_BETA_MM:-5.0}"
W_AUX_STAGES="${W_AUX_STAGES:-0.5}"
AUX_STAGE_WEIGHTS="${AUX_STAGE_WEIGHTS:-0.1,0.2,0.5}"
W_REG_CD="${W_REG_CD:-0.05}"
W_PHYS="${W_PHYS:-0.0}"
W_COMPLETION="${W_COMPLETION:-0.0}"

# --- Physics (off by default) ---
PHYS_K="${PHYS_K:-24}"
PHYS_REG="${PHYS_REG:-1e-4}"

DATA_OVERLAP="${DATA_OVERLAP:-0.25}"

BERT_MODEL="${BERT_MODEL:-${PROJECT_DIR}/bert-base-uncased}"
LEGACY_PIVOTS_CKPT="${LEGACY_PIVOTS_CKPT:-${PROJECT_DIR}/PIVOTS/checkpoints/pivots_v5/0/best_model.pth}"
PIVOTS_CKPT="${PIVOTS_CKPT:-}"
COMPLETION_CKPT="${COMPLETION_CKPT:-${PROJECT_DIR}/completion/logs/full_aug_20260805_013524/best.pth}"

# The new full2full_v1 model starts from scratch by default.  Resolve an old
# checkpoint only for the legacy architecture or an explicitly requested
# compatible encoder warm-start.
if [[ "${PIVOTS_ARCH}" == "legacy" || "${INITIALIZE_FROM_LEGACY_PIVOTS}" == "1" ]]; then
  PIVOTS_CKPT="${PIVOTS_CKPT:-${LEGACY_PIVOTS_CKPT}}"
fi

if [[ ! -f "${COMPLETION_CKPT}" ]]; then
  echo "[Error] completion checkpoint not found: ${COMPLETION_CKPT}" >&2
  exit 1
fi
if [[ "${PIVOTS_ARCH}" == "legacy" || "${INITIALIZE_FROM_LEGACY_PIVOTS}" == "1" ]] && [[ ! -f "${PIVOTS_CKPT}" ]]; then
  echo "[Error] PIVOTS checkpoint not found: ${PIVOTS_CKPT}" >&2
  exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-msn_completion_biomech}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-spaq_pivots_diag}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MASTER_PORT
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE}"

echo "[Info] MASTER_PORT=${MASTER_PORT}"
echo "[Info] TRAIN_DIR=${DATASET_ROOT}/train MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
echo "[Info] VALIDATION_DIR=${DATASET_ROOT}/validation MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES}"
echo "[Info] COMPLETION_CKPT=${COMPLETION_CKPT}"
if [[ -n "${PIVOTS_CKPT}" ]]; then
  echo "[Info] PIVOTS_CKPT=${PIVOTS_CKPT}"
else
  echo "[Info] PIVOTS_CKPT=none (full2full_v1 initialized from scratch)"
fi
echo "[Info] train_stage=${TRAIN_STAGE}"
echo "[Info] registration_target_mode=${REGISTRATION_TARGET_MODE}"
echo "[Info] PIVOTS_ARCH=${PIVOTS_ARCH}"
echo "[Info] GLOBAL_MATCH_LEVEL=${GLOBAL_MATCH_LEVEL} (default points=92)"
echo "[Info] GLOBAL_MATCH_TEMPERATURE=${GLOBAL_MATCH_TEMPERATURE}"
echo "[Info] GLOBAL_MATCH_DIM=${GLOBAL_MATCH_DIM}"
echo "[Info] GLOBAL_SPATIAL_SIGMA=${GLOBAL_SPATIAL_SIGMA}"
echo "[Info] MAX_COARSE_FLOW_NORMALIZED=${MAX_COARSE_FLOW_NORMALIZED}"
echo "[Info] NUM_REFINEMENT_STEPS=${NUM_REFINEMENT_STEPS}"
echo "[Info] REFINEMENT_K=${REFINEMENT_K}"
echo "[Info] INITIALIZE_FROM_LEGACY_PIVOTS=${INITIALIZE_FROM_LEGACY_PIVOTS}"
echo "[Info] LR=${LR} COMPLETION_LR=${COMPLETION_LR}"
echo "[Info] W_REG_HUBER=${W_REG_HUBER} W_AUX_STAGES=${W_AUX_STAGES} W_REG_CD=${W_REG_CD} W_PHYS=${W_PHYS} W_COMPLETION=${W_COMPLETION}"
echo "[Info] AUX_STAGE_WEIGHTS=${AUX_STAGE_WEIGHTS}"
echo "[Info] HUBER_BETA_MM=${HUBER_BETA_MM}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb)
fi

# Build --use_text / --no-use_text flag.
# Default diagnostic experiment: no text.
USE_TEXT_FLAG=""
if [[ "${USE_TEXT:-0}" == "1" ]]; then
  USE_TEXT_FLAG="--use_text"
else
  USE_TEXT_FLAG="--no-use_text"
fi

PIVOTS_BOOL_ARGS=()
if [[ "${INITIALIZE_FROM_LEGACY_PIVOTS}" == "1" ]]; then
  PIVOTS_BOOL_ARGS+=(--initialize_from_legacy_pivots)
else
  PIVOTS_BOOL_ARGS+=(--no-initialize_from_legacy_pivots)
fi
if [[ "${STRICT_PIVOTS_CHECKPOINT}" == "1" ]]; then
  PIVOTS_BOOL_ARGS+=(--strict_pivots_checkpoint)
else
  PIVOTS_BOOL_ARGS+=(--no-strict_pivots_checkpoint)
fi
if [[ "${DEBUG_REFINEMENT}" == "1" ]]; then
  PIVOTS_BOOL_ARGS+=(--debug_refinement)
fi

python train-multigpu.py \
  --dataset_root "${DATASET_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --world_size "${WORLD_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --completion_lr "${COMPLETION_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --data_overlap "${DATA_OVERLAP}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --train_stage "${TRAIN_STAGE}" \
  --registration_target_mode "${REGISTRATION_TARGET_MODE}" \
  --pivots_arch "${PIVOTS_ARCH}" \
  --global_match_level "${GLOBAL_MATCH_LEVEL}" \
  --global_match_temperature "${GLOBAL_MATCH_TEMPERATURE}" \
  --global_match_dim "${GLOBAL_MATCH_DIM}" \
  --global_spatial_sigma "${GLOBAL_SPATIAL_SIGMA}" \
  --max_coarse_flow_normalized "${MAX_COARSE_FLOW_NORMALIZED}" \
  --num_refinement_steps "${NUM_REFINEMENT_STEPS}" \
  --refinement_k "${REFINEMENT_K}" \
  --w_reg_huber "${W_REG_HUBER}" \
  --huber_beta_mm "${HUBER_BETA_MM}" \
  --w_aux_stages "${W_AUX_STAGES}" \
  --aux_stage_weights "${AUX_STAGE_WEIGHTS}" \
  --w_reg_cd "${W_REG_CD}" \
  --w_phys "${W_PHYS}" \
  --phys_k "${PHYS_K}" \
  --phys_reg "${PHYS_REG}" \
  --w_completion "${W_COMPLETION}" \
  --bert_model_name "${BERT_MODEL}" \
  --bert_local_files_only \
  --pivots_checkpoint "${PIVOTS_CKPT}" \
  "${PIVOTS_BOOL_ARGS[@]}" \
  --completion_checkpoint "${COMPLETION_CKPT}" \
  ${USE_TEXT_FLAG} \
  "${WANDB_ARGS[@]}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  "$@"


# Diagnostic experiments:
#
# Experiment A1 — 1-sample overfit (oracle full-to-full):
#   GPU_IDS=0 WORLD_SIZE=1 MAX_TRAIN_SAMPLES=1 MAX_VAL_SAMPLES=1 EPOCHS=300 BATCH_SIZE=1 \
#   TRAIN_STAGE=registration REGISTRATION_TARGET_MODE=gt \
#   PIVOTS_ARCH=full2full_v1 LR=1e-5 W_REG_HUBER=1.0 W_AUX_STAGES=0.5 \
#   W_REG_CD=0.05 W_PHYS=0 W_COMPLETION=0 USE_TEXT=0 \
#   ./run_train_multigpu.sh
#
# Experiment A2 — 5-sample overfit (oracle full-to-full):
#   GPU_IDS=0 WORLD_SIZE=1 MAX_TRAIN_SAMPLES=5 MAX_VAL_SAMPLES=5 EPOCHS=300 BATCH_SIZE=1 \
#   TRAIN_STAGE=registration REGISTRATION_TARGET_MODE=gt \
#   PIVOTS_ARCH=full2full_v1 LR=1e-5 W_REG_HUBER=1.0 W_AUX_STAGES=0.5 \
#   W_REG_CD=0.05 W_PHYS=0 W_COMPLETION=0 USE_TEXT=0 \
#   ./run_train_multigpu.sh
#
# Experiment B — completed full-to-full:
#   TRAIN_STAGE=registration REGISTRATION_TARGET_MODE=completed \
#   PIVOTS_ARCH=full2full_v1 DATA_OVERLAP=0.2 LR=1e-5 \
#   W_REG_HUBER=1.0 W_AUX_STAGES=0.5 W_REG_CD=0.05 \
#   W_PHYS=0 W_COMPLETION=0 USE_TEXT=0 \
#   ./run_train_multigpu.sh
