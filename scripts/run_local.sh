#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/small_experiment.yaml}
python run.py --config "${CONFIG}"
