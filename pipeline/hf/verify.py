# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.3", "pyarrow==26.0.0", "polars==1.44.2"]
# ///
"""Check the finished export against the sources, on random rows of every file.

    uv run pipeline/hf/verify.py --out /mnt/bulk/hf-export/openalex-denseon [--per-file 200]

For each sampled row: the vector equals the shard's vector bit for bit, and the title + abstract of
its OpenAlex record hash to the row's text_hash (so title, metadata and vector belong together).
Also: rows 0..N-1 each appear exactly once, and every vector has length ~1.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")
BASE_CONTENT = Path("/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts")
RELEASES = Path("/mnt/fast/scholarsreadinglist/release")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-file", type=int, default=200)
    args = ap.parse_args()
    ex = json.loads((args.out / "export.json").read_text())
    rel = json.loads((RELEASES / "releases" / ex["release"] / "manifest.json").read_text())
    shards = json.loads((Path(rel["paths"]["rows"]) / "shards.json").read_text())
    firsts = np.array([s["first"] for s in shards])
    updates_content = sorted((Path(rel["paths"]["updates"]).parents[1] / "content").glob("*.parquet"))
    rng = np.random.default_rng(0)

    seen = np.zeros(ex["rows"], dtype=bool)
    by_source: dict[tuple, list] = defaultdict(list)  # content files -> sampled (wid, text_hash)
    checked_vectors = 0
    for f in sorted((args.out / "data").rglob("part-*.parquet")):
        t = pq.read_table(f, columns=["row", "openalex_id", "text_hash", "embedding"])
        rows = t["row"].to_numpy()
        assert not seen[rows].any(), f"{f}: rows already seen"
        seen[rows] = True
        E = t["embedding"].combine_chunks().values.to_numpy().reshape(-1, 768)
        norms = np.linalg.norm(E[:: max(1, len(E) // 2000)].astype(np.float32), axis=1)
        assert np.abs(norms - 1).max() < 0.01, (f, norms.min(), norms.max())
        for i in rng.choice(len(rows), min(args.per_file, len(rows)), replace=False):
            r = int(rows[i])
            s = shards[int(np.searchsorted(firsts, r, side="right")) - 1]
            j = r - s["first"]
            v = np.load(SHARDS / s["dir"] / "vectors.npy", mmap_mode="r")[j]
            assert (v == E[i]).all(), (f, r)
            wid = int(np.load(SHARDS / s["dir"] / "work_ids.npy", mmap_mode="r")[j])
            assert f"W{wid}" == t["openalex_id"][i].as_py(), (f, r)
            checked_vectors += 1
            src = tuple(updates_content) if "/rg-u" in s["dir"] else (BASE_CONTENT / s["dir"].split("/")[0] / "data.parquet",)
            by_source[src].append((wid, t["text_hash"][i].as_py()))
    assert seen.all(), f"{(~seen).sum()} rows missing"
    print(f"rows 0..{ex['rows'] - 1:,} each exactly once; {checked_vectors:,} vectors identical to the shards", flush=True)

    ok = bad = 0
    for src, pairs in by_source.items():
        want = dict(pairs)
        recs = (pl.scan_parquet(list(src)).select("id", "title", "abstract")
                .with_columns(_wid=pl.col("id").str.strip_prefix("https://openalex.org/W").cast(pl.UInt64))
                .join(pl.LazyFrame({"_wid": np.array(list(want), dtype=np.uint64)}), on="_wid", how="semi")
                .collect(engine="streaming"))
        for w, ti, ab in zip(recs["_wid"], recs["title"], recs["abstract"]):
            h = hashlib.blake2b(f"{ti or ''} {ab}".strip().encode(), digest_size=16).digest()
            ok += h == want[w]
            bad += h != want[w]
        bad += len(want) - recs.height
    print(f"text hashes: {ok:,} match, {bad:,} do not")
    assert bad == 0
    (args.out / "verify.json").write_text(json.dumps({"rows": ex["rows"], "vectors_checked": checked_vectors,
                                                      "text_hashes_checked": ok, "mismatches": bad}))


if __name__ == "__main__":
    main()
