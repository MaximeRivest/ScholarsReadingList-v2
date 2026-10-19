"""Embed the works of a rescue corpus (pipeline/rescue/build.py) as shards beside the backfill's, in their own group.

    pipeline/embeddings/.venv/bin/python pipeline/rescue/embed.py RESCUE_DIR [--device cuda:1|cpu] [--limit N]

Same encoder, settings and storage as the backfill and the daily updates (pipeline/updates/embed.py: the backfill's
DenseEncoder and commit_shard with the backfill manifest's embedding settings), so the vectors share one space and
carry the backfill's plan_id; task.source says they come from the rescue.

Shards: <backfill>/shards/year=Y/rescue-<rescue run>/part-<start>, 10,000 works each. The group name does not start
with 'rg-': every existing reader of shards (the nightly release, positions, the smr collection) looks for
'year=*/rg-*/part-*' and does not see them; a release takes them only when asked to (rows.py --extra-groups).
Resumable: works already in a committed rescue shard are skipped. --limit N embeds the N most cited of the rest. GPU: only one with 3 GiB free and under 50 % busy,
GPU 1 first (GPU 0 is the site's rewriter and training); with --device cpu it runs on the CPU (slow).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "embeddings"))
sys.path.insert(0, str(HERE.parent / "updates"))
import embed_content as ec  # noqa: E402
from embed import pick_device  # noqa: E402  (pipeline/updates/embed.py)

BASE = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts")
SHARD_ROWS = 10_000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rescue", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="embed at most N works this run")
    a = ap.parse_args()
    corpus = a.rescue / "corpus"
    if not (corpus / "DONE.json").exists():
        raise SystemExit(f"{corpus}: not finished")
    group = f"rescue-{a.rescue.name}"
    manifest = json.loads((BASE / "manifest.json").read_text())
    doc = {"plan_id": manifest["plan_id"], "embedding": manifest["embedding"]}
    done = set()
    for c in (BASE / "shards").glob(f"year=*/{group}/part-*/commit.json"):
        done.update(np.load(c.parent / "work_ids.npy").tolist())
    parts = [p for p in sorted((corpus / "parts").glob("*.parquet")) if pq.ParquetFile(p).metadata.num_rows]
    # choose from the small columns first (7.8M abstracts do not fit beside the model): most cited first
    t = pa.concat_tables([pq.read_table(p, columns=["id", "year", "citations"]).append_column(
        "part", pa.array(np.full(pq.ParquetFile(p).metadata.num_rows, i, np.int32))) for i, p in enumerate(parts)])
    if done:
        t = t.filter(pc.invert(pc.is_in(t["id"], value_set=pa.array(sorted(done), pa.int64()))))
    t = t.sort_by([("citations", "descending"), ("id", "ascending")])
    if a.limit:
        t = t.slice(0, a.limit)
    t = t.sort_by([("year", "descending"), ("id", "ascending")])
    if t.num_rows == 0:
        log(f"every work of {a.rescue} has a vector")
        return
    device = pick_device(a.device)
    if device == "cpu" and not a.device:
        raise SystemExit("no GPU has 3 GiB free and is under 50 % busy; try later, or --device cpu (slow)")
    log(f"{t.num_rows:,} works to embed ({len(done):,} done before) on {device}")
    enc = ec.DenseEncoder(doc["embedding"], device, a.batch_size, cpu_threads=8,
                          allocation_gib=3 if device.startswith("cuda") else None)
    years = t["year"].to_pylist()

    # one pass over the parts: the chosen works' titles and abstracts (2M abstracts: ~3 GB)
    want = pa.array(t["id"].to_numpy())
    held = []
    for p in parts:
        for b in pq.ParquetFile(p).iter_batches(batch_size=200_000, columns=["id", "title", "abstract"]):
            x = pa.Table.from_batches([b])
            held.append(x.filter(pc.is_in(x["id"], value_set=want)))
    held = pa.concat_tables(held)
    where = {int(w): i for i, w in enumerate(held["id"].to_pylist())}
    log(f"texts read: {held.num_rows:,}")

    def texts_of(ids: np.ndarray) -> pa.Table:
        return held.take(pa.array([where[int(w)] for w in ids]))

    t0, n, start = time.time(), 0, 0
    while start < t.num_rows:
        y = years[start]
        end = start
        while end < t.num_rows and years[end] == y and end - start < SHARD_ROWS:
            end += 1
        chunk = texts_of(t["id"].slice(start, end - start).to_numpy())
        rec = pa.table({"id": pc.binary_join_element_wise(pa.scalar("W"), pc.cast(chunk["id"], pa.string()), ""),
                        "title": chunk["title"], "abstract": chunk["abstract"]})
        ids, texts, hashes = ec.records(rec)
        e0 = time.time()
        vectors = enc.encode(texts)
        stats = {"device": device, "batch_size": enc.batch_size, "inference_dtype": doc["embedding"]["inference_dtype"],
                 "encode_seconds": time.time() - e0, **enc.truncation_sample(texts, ids)}
        key = f"year={y if y is not None else -1}/{group}/part-{int(ids[0]):012d}"
        task = {"key": key, "rows": len(ids), "selected_start": start,
                "source": {"kind": "rescue", "rescue": str(a.rescue)}}
        ec.commit_shard(BASE, task, doc, ids, hashes, vectors, stats)
        n += len(ids)
        start = end
        log(f"  {key}: {len(ids):,} works ({n:,}/{t.num_rows:,}, {n / (time.time() - t0):,.0f}/s)")
    log(f"embedded {n:,} works in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
