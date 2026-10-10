# WARP header-only mmap startup fix

Upstream: https://github.com/pau-mensa/xtr-warp-rs
Source commit: `b86dd21dba3f9c1ee51976f4916032ecd96eb755`
Local change: `warp-read-header-only.patch`, in `rust/search/loader.rs`.

The released loader's `parse_npy_header()` calls `std::fs::read(path)` before
mapping the file. This reads and temporarily allocates the entire tensor, not
just the header. On a 15 GB search index this prevents startup on a low-RAM
machine despite mmap being enabled.

The patch reads a 10/12-byte prefix and then only the header. It validates the
header length (at most 1 MiB) and rejects truncated headers. No scoring, index,
quantization, or search parameters are changed. Raw 1-bit WARP scoring remains
unsupported; the patch does not implement it.

## Build and use

The source checkout is at
`/mnt/fast/scholarsreadinglist/retrieval-bench/sources/xtr-warp-rs`.
`build_warp_patch.sh` runs the upstream release-preparation script to select
`tch/torch-sys/pyo3-tch 0.24.0` for PyTorch 2.11, then `rcargo build --release -j 8`.
This mirrors the upstream torch-2.11 wheel build procedure.

The patched package is an **overlay**, not a modification of the installed wheel:

```
PYTHONPATH=/mnt/fast/scholarsreadinglist/retrieval-bench/warp-headerfix
```

The source tree's `Cargo.lock` pins the locally resolved build dependencies.
All patch benchmarks set `BENCH_ENGINE_VARIANT` and use separate result folders.

## Validation performed, October 5, 2026

- `probes/check_warp_headerfix.py --record` captures the released engine's output
  on 20 queries (5 from each scientific dataset), 100 results each.
- Running the same probe with the overlay produced exactly the same 2,000 ranked
  document IDs, and scores matched within 1e-5.
- `estimate_index_memory()` on the 15.5 GB unpooled 4-bit index succeeded in a
  fresh systemd scope with **1 GiB MemoryMax and no swap**, in about 1 ms after
  imports. This exercises the header parsing without reading the tensor payload.

The original wheel remains available for reproducing the startup failures.
The patch has not been submitted upstream or represented as an official release.
