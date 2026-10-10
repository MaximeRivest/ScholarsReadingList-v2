#!/usr/bin/env bash
# Index builds on GPU 1 (shared with the family services; ~7 GB free), then CPU-only
# latency tests restricted to 4 cores to emulate a small server.
set -uo pipefail
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib   # NixOS: no /sbin/ldconfig for Triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"
uv run python -m bench.run_beir build --models lateon mxbai32m || exit 1
uv run python -m bench.run_scale build --models lateon mxbai32m || exit 1
export BENCH_THREADS=4 CUDA_VISIBLE_DEVICES=""
taskset -c 0-3 uv run python -m bench.run_scale search --models lateon mxbai32m
taskset -c 0-3 uv run python -m bench.run_beir search --models lateon mxbai32m
