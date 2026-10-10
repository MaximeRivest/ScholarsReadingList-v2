#!/usr/bin/env bash
# Compression test chain (quality, builds, then serving speed). Each heavy step waits
# until the machine has memory to spare, so it does not disturb or get disturbed by
# other work on lambda.
set -euo pipefail
trap 'echo "FAILED: compression pipeline stopped at line $LINENO (exit $?)" >&2' ERR
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"
log() { echo "[$(date +%H:%M)] $*"; }

wait_for() {  # wait_for <min available GB> <max memory-pressure %>
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    psi=$(awk 'NR==1 {split($2,a,"="); print int(a[2])}' /proc/pressure/memory)
    [ "$avail" -ge "$1" ] && [ "$psi" -le "$2" ] && return
    log "waiting: ${avail} GB available (need $1), memory pressure ${psi}% (max $2)"
    sleep 120
  done
}

while systemctl --user is-active -q srl-bench-pool; do sleep 60; done
log "quality: exact scoring of every variant"
wait_for 40 20
uv run python -m bench.run_beir exact --models lateon_pool2 lateon_pool4
uv run python -m bench.run_beir exact --binary --models lateon lateon_pool2 lateon_pool4 denseon
uv run python -m bench.run_scale exact --models lateon_pool2 lateon_pool4
uv run python -m bench.run_scale exact --binary --models lateon lateon_pool2 lateon_pool4 denseon

log "builds: WARP 1/2/4-bit x pool 1/2/4 (GPU 1), then faiss (CPU)"
wait_for 110 10
uv run python -m bench.build_compressed warp
wait_for 40 10
taskset -c 16-47 uv run python -m bench.build_compressed faiss --threads 32

log "serving speed on 4 cores (8-11), RAM and disk regimes"
wait_for 60 2
uv run python -m bench.serve_bench run
log "done"
