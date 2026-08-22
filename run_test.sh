#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# V3 checkpoint to evaluate. Defaults to the model produced by run_train_multigpu.sh.
# Override on the command line: CHECKPOINT=/path/to/checkpoint.pth ./run_test.sh
export CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/v3_20260813_220238/last.pth}"

exec "${PROJECT_DIR}/scripts/test_v3.sh" "$@"
