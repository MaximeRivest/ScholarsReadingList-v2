#!/usr/bin/env bash
# Finish only the cells missing from the October 4 run. Do not rebuild embeddings.
set -euo pipefail
cd "$(dirname "$0")"
export BENCH_MAXSIM_DOC_CHUNK=512 BENCH_MAXSIM_BUDGET_MIB=64
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -m bench.run_scale exact --binary --models denseon --datasets scidocs trec-covid
.venv/bin/python -m bench.run_scale exact --binary --models lateon lateon_pool2 lateon_pool4 --datasets trec-covid
printf 'Missing compression-quality cells completed successfully.\n'
