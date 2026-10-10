#!/usr/bin/env bash
# Build a header-only loader fix against the same torch ABI as the pinned wheel.
# Source checkout and patch provenance are documented in patches/README.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=/mnt/fast/scholarsreadinglist/retrieval-bench/sources/xtr-warp-rs
export VIRTUAL_ENV="$HERE/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHON="$VIRTUAL_ENV/bin/python" PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH=$("$PYTHON" -c 'import torch,os; print(os.path.dirname(torch.__file__))')
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cd "$SRC"
python scripts/release_prep.py --version 2.1.0.2110 --torch-req 'torch>=2.11,<2.12' --tch-version 0.24.0
rcargo build --release -j 8
# Do not overwrite the installed package. Select this overlay via PYTHONPATH.
OVERLAY=/mnt/fast/scholarsreadinglist/retrieval-bench/warp-headerfix
mkdir -p "$OVERLAY"
cp -r python/xtr_warp "$OVERLAY/"
SUFFIX=$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')
cp target/release/libxtr_warp_rs.so "$OVERLAY/xtr_warp/xtr_warp_rs$SUFFIX"
printf 'Overlay built at %s\n' "$OVERLAY"
