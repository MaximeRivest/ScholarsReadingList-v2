"""Random sample of OpenAlex works (title + abstract) to use as distractors.

Draws from completed Parquet files only. Files are chosen at random, then rows are
sampled at the same rate in every file, so the sample is spread over many update dates but is
not a perfectly uniform sample of the whole snapshot (the download is incomplete).
Kept: main-collection works (not is_xpac) with a title and a non-empty abstract.
"""
from __future__ import annotations

import argparse
import glob
import random

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .common import DATA, OPENALEX_WORKS, abstract_from_inverted_index, normalize_title

OUT = DATA / "openalex"


def sample(n: int, n_files: int, seed: int = 13) -> None:
    files = sorted(glob.glob(str(OPENALEX_WORKS / "*" / "*.parquet")))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(files, min(n_files, len(files))))
    con = duckdb.connect()
    con.execute("SET threads=8")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in chosen)
    # About a quarter of all rows are main-collection works with an abstract; oversample
    # so the per-row sampling rate is the same in every file (proportional to its size).
    pct = min(100.0, 100.0 * 2.0 * n / (0.2 * total))
    rows = []
    for i, f in enumerate(chosen):
        part = con.execute(f"""
            SELECT id, title, abstract_inverted_index, publication_year, type, language
            FROM read_parquet('{f}')
            WHERE NOT coalesce(is_xpac, false) AND title IS NOT NULL
              AND abstract_inverted_index IS NOT NULL AND length(abstract_inverted_index) > 2
            USING SAMPLE {pct} PERCENT (bernoulli, {seed})""").fetchall()
        rows.extend(part)
        if i % 20 == 0:
            print(f"{i + 1}/{len(chosen)} files, {len(rows):,} candidates", flush=True)
    rng.shuffle(rows)
    seen, out = set(), []
    for wid, title, aii, year, typ, lang in rows:
        abstract = abstract_from_inverted_index(aii)
        key = normalize_title(title)
        if len(abstract.split()) < 20 or not key or key in seen:
            continue
        seen.add(key)
        out.append({"id": wid, "title": title, "text": f"{title} {abstract}", "year": year,
                    "type": typ, "language": lang})
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out):,} usable works; raise --files")
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), OUT / f"sample_{n}.parquet")
    print(f"wrote {len(out):,} works from {len(chosen)} of {len(files)} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--files", type=int, default=250)
    a = p.parse_args()
    sample(a.n, a.files)
