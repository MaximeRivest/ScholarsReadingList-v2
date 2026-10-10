#!/usr/bin/env bash
# Capacity-only 80M-entry experiment. Always unfreeze the previous sweep on exit.
set -euo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
trap 'systemctl --user thaw srl-compression-speed-fixed 2>/dev/null || true' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false
# The previously active query worker was allowed to finish before starting this.
taskset -c 16-31 .venv/bin/python -m bench.disk80m build
.venv/bin/python -m bench.disk80m queries
.venv/bin/python -m bench.disk80m sweep
printf '80M-entry capacity experiment complete; no 80M-corpus accuracy claim.\n'
