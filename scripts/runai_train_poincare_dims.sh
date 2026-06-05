#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/home/sjiang/hyperbolic_workspace}
POINCARE_ROOT=${POINCARE_ROOT:-${PROJECT_ROOT}/poincare-embeddings}
DIMS=${DIMS:-"2 3 5 10"}

for DIM in ${DIMS}; do
  CHECKPOINT=${CHECKPOINT_PREFIX:-nouns_gpu_d}${DIM}.bin \
  JOB_NAME=${JOB_NAME_PREFIX:-poincare-nouns-gpu-d}${DIM} \
  DIM=${DIM} \
  FRESH=${FRESH:-0} \
  CONVERT_AFTER=${CONVERT_AFTER:-0} \
  EMBEDDINGS_OUT="${PROJECT_ROOT}/data/embeddings/wordnet_embeddings_d${DIM}.csv" \
  "${POINCARE_ROOT}/runai_train_nouns.sh"
done

echo "Submitted Poincare noun embedding jobs for dims: ${DIMS}"
echo "After checkpoints are ready, convert with:"
echo "  scripts/convert_poincare_dims.sh"
