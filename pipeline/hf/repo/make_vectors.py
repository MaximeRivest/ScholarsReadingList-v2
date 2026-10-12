# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2", "pyarrow>=21"]
# ///
"""Copy every embedding into one memory-mappable array: vectors.npy, row r = the dataset's `row` r.

    uv run make_vectors.py                 # from inside the download folder (needs the whole dataset)
    uv run make_vectors.py --data ~/openalex-denseon

What it is for: random access by row number, which Parquet is bad at. search.py uses it to re-score
the index's candidates exactly. Size: rows x 768 x 2 bytes (~313 GB for the full dataset), float16,
row order. Put it on an SSD: re-scoring reads ~300 scattered rows per question.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DIM = 768


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, help="default: <data>/vectors.npy")
    args = ap.parse_args()
    out = args.out or args.data / "vectors.npy"
    files = sorted(args.data.glob("data/**/part-*.parquet"))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} files, {total:,} rows -> {out} ({total * DIM * 2 / 1e9:.0f} GB)")

    tmp = out.with_suffix(".npy.partial")
    vectors = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(total, DIM))
    seen = np.zeros(total, dtype=bool)
    for i, f in enumerate(files, 1):
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["row", "embedding"])
            rows = t["row"].to_numpy()
            if rows.max() >= total:
                raise SystemExit(f"{f}: row {rows.max()} beyond {total:,} rows: is the download incomplete?")
            vectors[rows] = t["embedding"].combine_chunks().values.to_numpy().reshape(-1, DIM)
            seen[rows] = True
        print(f"  {i}/{len(files)} {f.relative_to(args.data)}", flush=True)
    if not seen.all():
        raise SystemExit(f"{(~seen).sum():,} rows missing: download every file under data/ first")
    vectors.flush()
    del vectors
    tmp.rename(out)
    print("done")


if __name__ == "__main__":
    main()
