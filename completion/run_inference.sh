#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-/home/ma_sx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-liver}"

DATASET_ROOT="${DATASET_ROOT:-/home/ma_sx/Project/Dataset/MedShapeNet-Liver}"


CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/inference_outputs/comparison_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-validation}"
MAX_CASES="${MAX_CASES:-0}"
CROPS_PER_CASE="${CROPS_PER_CASE:-1}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-0}"
AMP="${AMP:-fp16}"
# Error comparison does not need the large per-case PLY files. Set to 1 if needed.
SAVE_POINTS="${SAVE_POINTS:-1}"
ANCHOR_OBSERVED="${ANCHOR_OBSERVED:-0}"

# Each checkpoint is evaluated at the overlap used to train its model.
OVERLAPS=(0.05 0.06 0.07 0.08 0.09 0.10 0.15 0.20 0.25 0.30)
RUN_NAMES=(
  full_aug_20260812_095734_overlap0.05
  full_aug_20260812_214024_overlap0.06
  full_aug_20260813_093558_overlap0.07
  full_aug_20260813_093538_overlap0.08
  full_aug_20260812_095539_overlap0.09
  full_aug_20260811_104608_overlap0.10
  full_aug_20260811_104624_overlap0.15
  full_aug_20260811_105240_overlap0.20
  full_aug_20260811_105310_overlap0.25
  full_aug_20260811_105337_overlap0.30
)

if [[ ${#OVERLAPS[@]} -ne ${#RUN_NAMES[@]} ]]; then
  echo "[Error] OVERLAPS and RUN_NAMES must have the same length" >&2
  exit 1
fi

# Fail before starting a long comparison if any model is unavailable.
for run_name in "${RUN_NAMES[@]}"; do
  checkpoint="${CHECKPOINT_ROOT}/${run_name}/best.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[Error] checkpoint not found: ${checkpoint}" >&2
    exit 1
  fi
done

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${SCRIPT_DIR}/SplAttN${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
echo "[Info] comparing ${#RUN_NAMES[@]} checkpoints; output_dir=${OUTPUT_DIR}"

for index in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[index]}"
  overlap="${OVERLAPS[index]}"
  checkpoint="${CHECKPOINT_ROOT}/${run_name}/best.pth"
  model_output_dir="${OUTPUT_DIR}/${run_name}"

  echo "[Info] model=$((index + 1))/${#RUN_NAMES[@]} run=${run_name} overlap=${overlap}"
  INFER_ARGS=(
    infer_completion.py
    --checkpoint "${checkpoint}"
    --dataset_root "${DATASET_ROOT}"
    --output_dir "${model_output_dir}"
    --split "${SPLIT}"
    --max_cases "${MAX_CASES}"
    --overlap "${overlap}"
    --crops_per_case "${CROPS_PER_CASE}"
    --seed "${SEED}"
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
done

# Build machine-readable results plus a compact table sorted by completion RMSE.
python - "${OUTPUT_DIR}" "${RUN_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = []
for run_name in sys.argv[2:]:
    summary_path = output_dir / run_name / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    rows.append({
        "run_name": run_name,
        "overlap": summary["overlap"],
        "rmse_mm": summary["rmse_mm"],
        "mae_mm": summary["mae_mm"],
        "observed_rmse_mm": summary["observed_rmse_mm"],
        "missing_rmse_mm": summary["missing_rmse_mm"],
        "source_baseline_rmse_mm": summary["source_baseline_rmse_mm"],
        "case_count": summary["case_count"],
        "view_count": summary["view_count"],
        "checkpoint": summary["checkpoint"],
    })

rows.sort(key=lambda row: row["rmse_mm"])
for rank, row in enumerate(rows, 1):
    row["rank"] = rank

comparison = {
    "sort_metric": "rmse_mm",
    "best_run": rows[0]["run_name"] if rows else None,
    "results": rows,
}
(output_dir / "comparison.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

columns = [
    "rank", "run_name", "overlap", "rmse_mm", "mae_mm",
    "observed_rmse_mm", "missing_rmse_mm", "source_baseline_rmse_mm",
    "case_count", "view_count", "checkpoint",
]
with (output_dir / "comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print("\n[Comparison] sorted by rmse_mm (lower is better)")
print(f"{'rank':>4}  {'overlap':>7}  {'rmse_mm':>12}  {'mae_mm':>12}  {'missing_rmse_mm':>17}  run")
for row in rows:
    print(
        f"{row['rank']:>4}  {row['overlap']:>7.2f}  "
        f"{row['rmse_mm']:>12.6f}  {row['mae_mm']:>12.6f}  "
        f"{row['missing_rmse_mm']:>17.6f}  {row['run_name']}"
    )
print(f"[Result] comparison_json={output_dir / 'comparison.json'}")
print(f"[Result] comparison_tsv={output_dir / 'comparison.tsv'}")
PY
