#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname -- "${SCRIPT_DIR}")}"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

GPU_IDS="${GPU_IDS:-0}"
GPU_COUNT="$(awk -F, '{print NF}' <<<"${GPU_IDS}")"
WORLD_SIZE="${WORLD_SIZE:-${GPU_COUNT}}"
DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/logs/v3_$(date +%Y%m%d_%H%M%S)}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

TRAIN_STAGE="${TRAIN_STAGE:-joint}"
REGISTRATION_TARGET_MODE="${REGISTRATION_TARGET_MODE:-completed}"
COMPLETION_FROM_SCRATCH="${COMPLETION_FROM_SCRATCH:-0}"
END_TO_END_COMPLETION="${END_TO_END_COMPLETION:-1}"

GLOBAL_MATCH_LEVEL="${GLOBAL_MATCH_LEVEL:-4}"
GLOBAL_MATCH_DIM="${GLOBAL_MATCH_DIM:-64}"
NUM_REFINEMENT_STEPS="${NUM_REFINEMENT_STEPS:-3}"
REFINEMENT_K="${REFINEMENT_K:-35}"
DEBUG_REFINEMENT="${DEBUG_REFINEMENT:-0}"
V3_FEATURE_TEMPERATURE="${V3_FEATURE_TEMPERATURE:-1.0}"
V3_SPATIAL_TEMPERATURE="${V3_SPATIAL_TEMPERATURE:-1.0}"
SOURCE_GRAPH_K="${SOURCE_GRAPH_K:-16}"

LR="${LR:-1e-5}"
COMPLETION_LR="${COMPLETION_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

W_REG_HUBER="${W_REG_HUBER:-1.0}"
HUBER_BETA_MM="${HUBER_BETA_MM:-5.0}"
W_REG_MSE="${W_REG_MSE:-0.02}"
W_REG_CD_GT="${W_REG_CD_GT:-0.05}"
W_REG_CD_COMPLETED="${W_REG_CD_COMPLETED:-0.0}"
W_AUX_STAGES="${W_AUX_STAGES:-0.5}"
AUX_STAGE_WEIGHTS="${AUX_STAGE_WEIGHTS:-0.1,0.2,0.5}"
W_MATCH="${W_MATCH:-0.1}"
MATCH_SIGMA_MM="${MATCH_SIGMA_MM:-5.0}"
W_EDGE="${W_EDGE:-0.1}"
EDGE_K="${EDGE_K:-8}"
EDGE_BETA_MM="${EDGE_BETA_MM:-2.0}"
W_PHYS="${W_PHYS:-0.0}"
W_COMPLETION="${W_COMPLETION:-0.1}"
PHYS_K="${PHYS_K:-24}"
PHYS_REG="${PHYS_REG:-1e-4}"
DATA_OVERLAP="${DATA_OVERLAP:-0.25}"
DATA_OVERLAPS="${DATA_OVERLAPS:-0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30}"
AMP_DTYPE="${AMP_DTYPE:-fp32}"

INIT_REGISTRATION_CHECKPOINT="${INIT_REGISTRATION_CHECKPOINT:-}"
# Set COMPLETION_CKPT to use one legacy checkpoint for every sample. By
# default, the ten specialized checkpoints below are routed by sample overlap.
COMPLETION_CKPT="${COMPLETION_CKPT:-}"
COMPLETION_RUN_SPECS=(
  "0.05=full_aug_20260812_095734_overlap0.05"
  "0.06=full_aug_20260812_214024_overlap0.06"
  "0.07=full_aug_20260813_093558_overlap0.07"
  "0.08=full_aug_20260813_093538_overlap0.08"
  "0.09=full_aug_20260812_095539_overlap0.09"
  "0.10=full_aug_20260811_104608_overlap0.10"
  "0.15=full_aug_20260811_104624_overlap0.15"
  "0.20=full_aug_20260811_105240_overlap0.20"
  "0.25=full_aug_20260811_105310_overlap0.25"
  "0.30=full_aug_20260811_105337_overlap0.30"
)
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-msn_completion_biomech}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-v3_joint_routed}"

die() {
  echo "[Error] $*" >&2
  exit 1
}

[[ -f "${PROJECT_DIR}/liver2/training/train_multigpu.py" ]] || die "invalid PROJECT_DIR: ${PROJECT_DIR}"
[[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]] || die "conda initialization script not found"
[[ -d "${DATASET_ROOT}" ]] || die "dataset root not found: ${DATASET_ROOT}"
[[ "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "WORLD_SIZE must be a positive integer"
(( WORLD_SIZE <= GPU_COUNT )) || die "WORLD_SIZE=${WORLD_SIZE} exceeds GPU_IDS count ${GPU_COUNT}"
completion_checkpoint_args=()
if [[ "${COMPLETION_FROM_SCRATCH}" == "1" ]]; then
  COMPLETION_CKPT=""
  completion_checkpoint_args+=(--completion_checkpoint "")
elif [[ -n "${COMPLETION_CKPT}" ]]; then
  [[ -f "${COMPLETION_CKPT}" ]] || die "completion checkpoint not found: ${COMPLETION_CKPT}"
  completion_checkpoint_args+=(--completion_checkpoint "${COMPLETION_CKPT}")
else
  completion_checkpoint_args+=(--completion_checkpoint "")
  for spec in "${COMPLETION_RUN_SPECS[@]}"; do
    overlap="${spec%%=*}"
    run_name="${spec#*=}"
    checkpoint="${PROJECT_DIR}/completion/logs/${run_name}/best.pth"
    [[ -f "${checkpoint}" ]] || die "completion checkpoint not found: ${checkpoint}"
    completion_checkpoint_args+=(
      --completion_checkpoint_map "${overlap}=${checkpoint}"
    )
  done
fi
if [[ -n "${INIT_REGISTRATION_CHECKPOINT}" ]]; then
  [[ -f "${INIT_REGISTRATION_CHECKPOINT}" ]] || die "registration checkpoint not found: ${INIT_REGISTRATION_CHECKPOINT}"
fi

# shellcheck source=/dev/null
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MASTER_PORT
export PYTHONUNBUFFERED=1
export WANDB_MODE

args=(
  --dataset_root "${DATASET_ROOT}"
  --save_dir "${SAVE_DIR}"
  --world_size "${WORLD_SIZE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --completion_lr "${COMPLETION_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_clip "${GRAD_CLIP}"
  --data_overlap "${DATA_OVERLAP}"
  --data_overlaps "${DATA_OVERLAPS}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_val_samples "${MAX_VAL_SAMPLES}"
  --train_stage "${TRAIN_STAGE}"
  --registration_target_mode "${REGISTRATION_TARGET_MODE}"
  --global_match_level "${GLOBAL_MATCH_LEVEL}"
  --global_match_dim "${GLOBAL_MATCH_DIM}"
  --num_refinement_steps "${NUM_REFINEMENT_STEPS}"
  --refinement_k "${REFINEMENT_K}"
  --v3_feature_temperature "${V3_FEATURE_TEMPERATURE}"
  --v3_spatial_temperature "${V3_SPATIAL_TEMPERATURE}"
  --source_graph_k "${SOURCE_GRAPH_K}"
  --w_reg_huber "${W_REG_HUBER}"
  --huber_beta_mm "${HUBER_BETA_MM}"
  --w_reg_mse "${W_REG_MSE}"
  --w_reg_cd_gt "${W_REG_CD_GT}"
  --w_reg_cd_completed "${W_REG_CD_COMPLETED}"
  --w_aux_stages "${W_AUX_STAGES}"
  --aux_stage_weights "${AUX_STAGE_WEIGHTS}"
  --w_match "${W_MATCH}"
  --match_sigma_mm "${MATCH_SIGMA_MM}"
  --w_edge "${W_EDGE}"
  --edge_k "${EDGE_K}"
  --edge_beta_mm "${EDGE_BETA_MM}"
  --w_phys "${W_PHYS}"
  --phys_k "${PHYS_K}"
  --phys_reg "${PHYS_REG}"
  --w_completion "${W_COMPLETION}"
  --amp_dtype "${AMP_DTYPE}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
)
args+=("${completion_checkpoint_args[@]}")

[[ "${DEBUG_REFINEMENT}" == "1" ]] && args+=(--debug_refinement)
[[ "${COMPLETION_FROM_SCRATCH}" == "1" ]] && args+=(--completion_from_scratch) || args+=(--no-completion_from_scratch)
[[ "${END_TO_END_COMPLETION}" == "1" ]] && args+=(--end_to_end_completion) || args+=(--no-end_to_end_completion)
[[ "${USE_WANDB}" == "1" ]] && args+=(--use_wandb) || args+=(--no-use_wandb)
[[ -n "${INIT_REGISTRATION_CHECKPOINT}" ]] && args+=(--init_registration_checkpoint "${INIT_REGISTRATION_CHECKPOINT}")

echo "[Info] project=${PROJECT_DIR} GPUs=${GPU_IDS} world_size=${WORLD_SIZE} port=${MASTER_PORT}"
echo "[Info] stage=${TRAIN_STAGE} target=${REGISTRATION_TARGET_MODE} architecture=full2full_v3"
echo "[Info] overlaps=${DATA_OVERLAPS} completion_routing=$([[ -z "${COMPLETION_CKPT}" ]] && echo specialized || echo single)"
echo "[Info] dataset=${DATASET_ROOT} save_dir=${SAVE_DIR}"

exec python -m liver2.training.train_multigpu "${args[@]}" "$@"
