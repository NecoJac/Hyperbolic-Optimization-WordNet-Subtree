#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/sjiang/hyperbolic_workspace}
POINCARE_ROOT=${POINCARE_ROOT:-${PROJECT_ROOT}/poincare-embeddings}
PYTHON=${PYTHON:-/home/sjiang/miniconda3/envs/poincare-modern/bin/python}
DIMS=${DIMS:-"2 3 5 10"}
CHECKPOINT_PREFIX=${CHECKPOINT_PREFIX:-nouns_gpu_d}
CHECKPOINT_SUFFIX=${CHECKPOINT_SUFFIX:-.bin}

cd "${PROJECT_ROOT}"
for DIM in ${DIMS}; do
  CHECKPOINT="${POINCARE_ROOT}/${CHECKPOINT_PREFIX}${DIM}${CHECKPOINT_SUFFIX}"
  OUT="${PROJECT_ROOT}/data/embeddings/wordnet_embeddings_d${DIM}.csv"
  echo "Converting dim=${DIM}: ${CHECKPOINT} -> ${OUT}"
  "${PYTHON}" scripts/convert_poincare_wordnet.py \
    --closure "${POINCARE_ROOT}/wordnet/noun_closure.csv" \
    --checkpoint "${CHECKPOINT}" \
    --edges-out "${PROJECT_ROOT}/data/processed/wordnet_edges.csv" \
    --embeddings-out "${OUT}"
done
