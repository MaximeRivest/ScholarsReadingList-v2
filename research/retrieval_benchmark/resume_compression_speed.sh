#!/usr/bin/env bash
# Repaired speed experiment. Uses a separate, validated WARP loader overlay.
# No GPU work and no service interruptions. Failures are retained, never called done.
set -uo pipefail
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export PYTHONPATH=/mnt/fast/scholarsreadinglist/retrieval-bench/warp-headerfix
export BENCH_ENGINE_VARIANT='xtr-warp-rs b86dd21d + header-only loader patch; torch ABI 0.24.0'
export BENCH_MEMORY_MIB=2048 BENCH_CORES=8-11
export BENCH_RESULTS=/mnt/fast/scholarsreadinglist/retrieval-bench/results/compress-v3-headerfix-nvme
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
failed=0
run() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; if ! "$@"; then failed=1; echo 'STAGE FAILED; details retained in results'; fi; }
run .venv/bin/python -m bench.serve_bench run --only lateon-pool1-4bit-np32 lateon-pool4-2bit-np32 denseon-hnsw32-ef128 denseon-ivfsq8-np64

# Same two serving indexes on the spinning disk, separate labels/results. Use 25
# queries per dataset to bound HDD runtime; saved query IDs allow matched comparison.
run .venv/bin/python prepare_hdd_benchmark.py
run env BENCH_INDEX_ROOT=/mnt/bulk/scholarsreadinglist-retrieval-bench/index2 \
BENCH_RESULTS=/mnt/fast/scholarsreadinglist/retrieval-bench/results/compress-v3-headerfix-hdd \
BENCH_STORAGE_LABEL='Seagate ST12000VN0008 12TB spinning HDD /mnt/bulk' \
BENCH_QUERIES_PER_DATASET=25 \
.venv/bin/python -m bench.serve_bench run --only lateon-pool4-2bit-np32 denseon-ivfsq8-np64 --regimes disk

# Remaining quality/storage/speed points (1-bit WARP configs are explicitly marked
# unsupported). Do not repeat the four completed representatives above.
mapfile -t remaining < <(.venv/bin/python -c 'from bench.serve_bench import configs; done={"lateon-pool1-4bit-np32","lateon-pool4-2bit-np32","denseon-hnsw32-ef128","denseon-ivfsq8-np64"}; print("\n".join(k for k in configs() if k not in done))')
run .venv/bin/python -m bench.serve_bench run --only "${remaining[@]}"
if ((failed)); then echo 'PARTIAL: at least one stage failed; inspect results before reporting.'; exit 1; fi
echo 'All supported speed experiments completed; 1-bit WARP remains unsupported.'
