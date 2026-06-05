#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

COMMON_ENV=(
  GPU="${GPU:-0}"
  CPU="${CPU:-8}"
  MEMORY="${MEMORY:-48G}"
  INSTALL_DEPS="${INSTALL_DEPS:-0}"
  SKIP_FINALIZE=1
)

for TREE in animal mammal group worker; do
  env "${COMMON_ENV[@]}" \
    JOB_NAME="hypopt-full-${TREE}" \
    CONFIG="configs/full_${TREE}.yaml" \
    LOG_PATH="${PROJECT_ROOT:-/home/sjiang/hyperbolic_workspace}/logs/$(date +%m%d)_hypopt-full-${TREE}.txt" \
    "${SCRIPT_DIR}/run_full.sh"
done

echo "Submitted split full jobs: animal, mammal, group, worker."
echo "After all finish, run:"
echo "  CONFIG=configs/full_experiment.yaml INSTALL_DEPS=0 scripts/plot_all.sh"
