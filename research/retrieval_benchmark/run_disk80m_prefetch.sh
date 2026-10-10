#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
trap 'systemctl --user thaw srl-compression-speed-fixed 2>/dev/null || true' EXIT
systemctl --user freeze srl-compression-speed-fixed 2>/dev/null || true
# Let an already-running isolated query worker finish; never benchmark against it.
deadline=$((SECONDS+180))
while pgrep -f 'bench[.]serve_bench worker' >/dev/null; do
  if ((SECONDS > deadline)); then echo 'Other timing worker still active; not taking contaminated measurements'; exit 1; fi
  sleep 2
done
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
.venv/bin/python -m bench.disk80m sweep --prefetch
.venv/bin/python summarize_disk80m.py
