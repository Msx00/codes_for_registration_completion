#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

GPU_IDS="${GPU_IDS:-0,1}"
WORLD_SIZE="${WORLD_SIZE:-2}"
DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"

MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-600}"
MAX_VAL_CASES="${MAX_VAL_CASES:-100}"
NUM_POINTS="${NUM_POINTS:-2048}"
CROPS_PER_GT="${CROPS_PER_GT:-4}"
OVERLAP_MIN="${OVERLAP_MIN:-0.07}"
OVERLAP_MAX="${OVERLAP_MAX:-0.07}"
ANCHOR_OVERLAP="${ANCHOR_OVERLAP:-0.07}"
ANCHOR_PROBABILITY="${ANCHOR_PROBABILITY:-1}"
PARTIAL_JITTER_MM="${PARTIAL_JITTER_MM:-0.0}"
# Use the same augmentation distribution in every epoch.  A changing
# curriculum makes losses from adjacent epochs incomparable and caused the
# systematic rebound seen around epochs 5-10 in the previous run.
AUGMENTATION_CURRICULUM_EPOCHS="${AUGMENTATION_CURRICULUM_EPOCHS:-0}"
ARCHITECTURE="${ARCHITECTURE:-generative}"
ENCODER_DEPTH="${ENCODER_DEPTH:-3}"
DECODER_DEPTH="${DECODER_DEPTH:-4}"
DENOISE_QUERIES="${DENOISE_QUERIES:-64}"
DENOISE_JITTER="${DENOISE_JITTER:-0.005}"

SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/logs/full_aug_$(date +%Y%m%d_%H%M%S)_overlap${ANCHOR_OVERLAP}}"

EPOCHS="${EPOCHS:-200}"
BATCH_CASES="${BATCH_CASES:-1}"
if [[ -z "${GRAD_ACCUM_STEPS+x}" ]]; then
  GRAD_ACCUM_STEPS="$(((4 + WORLD_SIZE - 1) / WORLD_SIZE))"
fi
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
AMP="${AMP:-fp16}"
AMP_INIT_SCALE="${AMP_INIT_SCALE:-4}"
MASTER_PORT="${MASTER_PORT:-29517}"

SET_LOSS_MODE="${SET_LOSS_MODE:-correntropy}"
CORRENTROPY_SIGMA="${CORRENTROPY_SIGMA:-1.0}"
CORRENTROPY_TRUNC="${CORRENTROPY_TRUNC:-0.2}"
CORRESPONDENCE_LOSS="${CORRESPONDENCE_LOSS:-hybrid}"
HUBER_BETA_MM="${HUBER_BETA_MM:-5.0}"
MISSING_WEIGHT="${MISSING_WEIGHT:-1.5}"
W_HUBER="${W_HUBER:-1.0}"
W_SET="${W_SET:-0.20}"
W_PARTIAL="${W_PARTIAL:-0.5}"
W_SMOOTH="${W_SMOOTH:-0.05}"
W_EDGE="${W_EDGE:-0.05}"
W_COARSE_SET="${W_COARSE_SET:-0.25}"
W_MID_SET="${W_MID_SET:-0.50}"
W_FINE_SET="${W_FINE_SET:-1.0}"
W_DENOISE="${W_DENOISE:-0.5}"
W_REPULSION="${W_REPULSION:-0.01}"
REPULSION_K="${REPULSION_K:-5}"
REPULSION_RADIUS="${REPULSION_RADIUS:-0.02}"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SCRIPT_DIR}/SplAttN${PYTHONPATH:+:${PYTHONPATH}}"

echo "[Info] Pure completion training"
echo "[Info] GPUs=${GPU_IDS} WORLD_SIZE=${WORLD_SIZE}"
echo "[Info] train/validation limits=${MAX_TRAIN_CASES}/${MAX_VAL_CASES}"
echo "[Info] crops/GT=${CROPS_PER_GT} overlap=${OVERLAP_MIN}-${OVERLAP_MAX} anchor=${ANCHOR_OVERLAP}"
echo "[Info] crop type=ball; training locations change every epoch"
echo "[Info] architecture=${ARCHITECTURE}; partial_jitter=${PARTIAL_JITTER_MM}mm"
echo "[Info] augmentation curriculum=${AUGMENTATION_CURRICULUM_EPOCHS} epochs"
echo "[Info] grad_accum_steps=${GRAD_ACCUM_STEPS}; effective global views/step≈$((BATCH_CASES * CROPS_PER_GT * WORLD_SIZE * GRAD_ACCUM_STEPS))"
if [[ "${ARCHITECTURE}" == "generative" ]]; then
  echo "[Info] AdaPoinTr-style adaptive queries: encoder/decoder=${ENCODER_DEPTH}/${DECODER_DEPTH}, denoise_queries=${DENOISE_QUERIES}"
  echo "[Info] set loss mode=${SET_LOSS_MODE}"
  echo "[Info] stage weights coarse/mid/fine=${W_COARSE_SET}/${W_MID_SET}/${W_FINE_SET}"
  echo "[Info] Correntropy sigma=${CORRENTROPY_SIGMA}"
  echo "[Info] Correntropy trunc=${CORRENTROPY_TRUNC}"
  echo "[Info] W_PARTIAL=${W_PARTIAL}"
  echo "[Info] W_REPULSION=${W_REPULSION}"
  echo "[Info] REPULSION_K=${REPULSION_K}"
  echo "[Info] REPULSION_RADIUS=${REPULSION_RADIUS} (source-normalized coordinates, approximately unit sphere)"
else
  echo "[Info] correspondence=${CORRESPONDENCE_LOSS} set_loss=${SET_LOSS_MODE} Huber beta=${HUBER_BETA_MM}mm missing_weight=${MISSING_WEIGHT}"
fi
echo "[Info] save_dir=${SAVE_DIR}"

TRAIN_ARGS=(
  train_completion.py
  --dataset_root "${DATASET_ROOT}"
  --save_dir "${SAVE_DIR}"
  --max_train_cases "${MAX_TRAIN_CASES}"
  --max_val_cases "${MAX_VAL_CASES}"
  --num_points "${NUM_POINTS}"
  --crops_per_gt "${CROPS_PER_GT}"
  --overlap_min "${OVERLAP_MIN}"
  --overlap_max "${OVERLAP_MAX}"
  --anchor_overlap "${ANCHOR_OVERLAP}"
  --anchor_probability "${ANCHOR_PROBABILITY}"
  --partial_jitter_mm "${PARTIAL_JITTER_MM}"
  --augmentation_curriculum_epochs "${AUGMENTATION_CURRICULUM_EPOCHS}"
  --architecture "${ARCHITECTURE}"
  --encoder_depth "${ENCODER_DEPTH}"
  --decoder_depth "${DECODER_DEPTH}"
  --denoise_queries "${DENOISE_QUERIES}"
  --denoise_jitter "${DENOISE_JITTER}"
  --epochs "${EPOCHS}"
  --batch_cases "${BATCH_CASES}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --amp "${AMP}"
  --amp_init_scale "${AMP_INIT_SCALE}"
  --correspondence_loss "${CORRESPONDENCE_LOSS}"
  --set_loss_mode "${SET_LOSS_MODE}"
  --correntropy_sigma "${CORRENTROPY_SIGMA}"
  --correntropy_trunc "${CORRENTROPY_TRUNC}"
  --huber_beta_mm "${HUBER_BETA_MM}"
  --missing_weight "${MISSING_WEIGHT}"
  --w_huber "${W_HUBER}"
  --w_set "${W_SET}"
  --w_partial "${W_PARTIAL}"
  --w_smooth "${W_SMOOTH}"
  --w_edge "${W_EDGE}"
  --w_coarse_set "${W_COARSE_SET}"
  --w_mid_set "${W_MID_SET}"
  --w_fine_set "${W_FINE_SET}"
  --w_denoise "${W_DENOISE}"
  --w_repulsion "${W_REPULSION}"
  --repulsion_k "${REPULSION_K}"
  --repulsion_radius "${REPULSION_RADIUS}"
)
TRAIN_ARGS+=("$@")

if [[ "${WORLD_SIZE}" == "1" ]]; then
  python "${TRAIN_ARGS[@]}"
else
  torchrun \
    --standalone \
    --nproc_per_node "${WORLD_SIZE}" \
    "${TRAIN_ARGS[@]}"
fi
