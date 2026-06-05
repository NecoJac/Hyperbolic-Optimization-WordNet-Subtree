#!/usr/bin/env bash
set -euo pipefail

JOB_NAME=${JOB_NAME:-hypopt-full}
NODE_POOL=${NODE_POOL:-default}
GPU=${GPU:-0}
CPU=${CPU:-8}
MEMORY=${MEMORY:-48G}
CONFIG=${CONFIG:-configs/full_experiment.yaml}
INSTALL_DEPS=${INSTALL_DEPS:-1}
SKIP_FINALIZE=${SKIP_FINALIZE:-0}
FINALIZE_ONLY=${FINALIZE_ONLY:-0}

PROJECT_ROOT=${PROJECT_ROOT:-/home/sjiang/hyperbolic_workspace}
CONDA_ENV=${CONDA_ENV:-hypopt}
RUNAI_PROJECT=${RUNAI_PROJECT:-vita-sjiang}
RUNAI_USER=${RUNAI_USER:-sjiang}
RUNAI_UID=${RUNAI_UID:-$(id -u)}
RUNAI_IMAGE=${RUNAI_IMAGE:-wymancv/opencv-v2:cuda-12}
LOG_PATH=${LOG_PATH:-${PROJECT_ROOT}/logs/$(date +%m%d)_${JOB_NAME}.txt}

RUN_SCRIPT_B64=$(base64 -w0 <<'RUN_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")" results/logs results/tables results/figures results/checkpoints

if [[ -f /home/sjiang/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/sjiang/miniconda3/etc/profile.d/conda.sh
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
  fi
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export USER="${RUNAI_USER:-sjiang}"
export LOGNAME="${RUNAI_USER:-sjiang}"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache/runai"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torch_inductor"
mkdir -p "${TORCH_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements.txt
fi

RUN_ARGS=(--config "${CONFIG}")
if [[ "${SKIP_FINALIZE}" == "1" ]]; then
  RUN_ARGS+=(--skip-finalize)
fi
if [[ "${FINALIZE_ONLY}" == "1" ]]; then
  RUN_ARGS+=(--finalize-only)
fi
python run.py "${RUN_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"
RUN_SCRIPT
)

runai submit \
  --project "${RUNAI_PROJECT}" \
  --name "${JOB_NAME}" \
  --run-as-uid "${RUNAI_UID}" \
  --run-as-user "${RUNAI_USER}" \
  --image "${RUNAI_IMAGE}" \
  --gpu "${GPU}" \
  --cpu "${CPU}" \
  --memory "${MEMORY}" \
  --large-shm \
  --pvc vita-scratch:/mnt/vita/scratch \
  --pvc home:/home/sjiang \
  --node-pool "${NODE_POOL}" \
  --working-dir /tmp \
  --environment "RUN_SCRIPT_B64=${RUN_SCRIPT_B64}" \
  --environment "PROJECT_ROOT=${PROJECT_ROOT}" \
  --environment "CONDA_ENV=${CONDA_ENV}" \
  --environment "CONFIG=${CONFIG}" \
  --environment "INSTALL_DEPS=${INSTALL_DEPS}" \
  --environment "SKIP_FINALIZE=${SKIP_FINALIZE}" \
  --environment "FINALIZE_ONLY=${FINALIZE_ONLY}" \
  --environment "LOG_PATH=${LOG_PATH}" \
  --environment "RUNAI_USER=${RUNAI_USER}" \
  --environment "RUNAI_UID=${RUNAI_UID}" \
  --command -- bash -c 'printf "%s" "$RUN_SCRIPT_B64" | base64 -d | bash'
