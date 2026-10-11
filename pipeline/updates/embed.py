"""Daily: embed the works of the newest updates corpus that have no vector yet, as shards beside the backfill's.

    pipeline/embeddings/.venv/bin/python pipeline/updates/embed.py [--device cuda:1|cpu]

Same model, revision, prompt, token cap, text recipe and storage as the backfill: the encoder and the shard
writer are the backfill's own (pipeline/embeddings/embed_content.py: DenseEncoder, records, commit_shard), with
the embedding settings read from the backfill's manifest. That is why a daily shard carries the backfill's
plan_id: it marks vectors of one space; `task.source` says where the texts came from.

Shards: <backfill>/shards/year=Y/rg-u<run>/part-<date>-<start>, at most 10,000 works, one publication date
each; committed under the updates lock. Only works of a finished updates corpus are embedded, so every vector
has its details (a release refuses a vector without them).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "embeddings"))
import common as C  # noqa: E402
import embed_content as ec  # noqa: E402

SHARD_ROWS = 10_000


def embedded_ids() -> set[int]:
    """Every W-number already in a daily shard (they are few: read them all)."""
    out: set[int] = set()
    for c in (C.SHARDS_OUT / "shards").glob(f"year=*/{C.SHARD_GROUP_PREFIX}*/part-*/commit.json"):
        out.update(np.load(c.parent / "work_ids.npy").tolist())
    return out


def pick_device(want: str | None, min_free_mib: int = 3072) -> str:
    if want:
        return want
    try:
        gpus = ec.gpu_candidates()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "cpu"
    ok = [g for g in gpus if g["free_mib"] >= min_free_mib and g["utilization"] <= 50]
    ok.sort(key=lambda g: (g["index"] != 1, -g["free_mib"]))   # GPU 1 first: GPU 0 serves the site's rewrites
    return f"cuda:{ok[0]['index']}" if ok else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="cuda:N or cpu (default: a GPU with 3 GiB free, else the CPU)")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    corpus = C.latest_corpus()
    if corpus is None:
        C.log("no updates corpus yet: nothing to embed")
        return
    manifest = json.loads((C.BASE_EMBEDDINGS / "manifest.json").read_text())
    doc = {"plan_id": manifest["plan_id"], "embedding": manifest["embedding"]}
    done = embedded_ids()
    parts = [pq.read_table(p, columns=["id", "title", "abstract", "date"])
             for p in sorted((corpus / "parts").glob("*.parquet")) if pq.ParquetFile(p).metadata.num_rows]
    t = pa.concat_tables(parts) if parts else None
    if t is None or t.num_rows == 0:
        C.log("updates corpus is empty")
        return
    todo = pc.invert(pc.is_in(t["id"], value_set=pa.array(sorted(done), pa.int64())))
    t = t.filter(todo).sort_by([("date", "descending"), ("id", "ascending")])
    if t.num_rows == 0:
        C.log(f"every work of {corpus.name} has a vector")
        return
    device = pick_device(a.device)
    C.log(f"{t.num_rows:,} works to embed from {corpus.name} on {device}")
    enc = ec.DenseEncoder(doc["embedding"], device, a.batch_size, cpu_threads=8,
                          allocation_gib=3 if device.startswith("cuda") else None)
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dates = t["date"].to_pylist()
    t0, n = time.time(), 0
    start = 0
    within: dict = {}            # date -> works of that date put in shards so far (the shard name's start)
    while start < t.num_rows:
        date = dates[start]
        end = start
        while end < t.num_rows and dates[end] == date and end - start < SHARD_ROWS:
            end += 1
        chunk = t.slice(start, end - start)
        # embed_content.records wants OpenAlex ids as text, like the snapshot's content table
        rec = pa.table({"id": pc.binary_join_element_wise(pa.scalar("W"), pc.cast(chunk["id"], pa.string()), ""),
                        "title": chunk["title"], "abstract": chunk["abstract"]})
        ids, texts, hashes = ec.records(rec)
        e0 = time.time()
        vectors = enc.encode(texts)
        stats = {"device": device, "batch_size": enc.batch_size, "inference_dtype": doc["embedding"]["inference_dtype"],
                 "encode_seconds": time.time() - e0, **enc.truncation_sample(texts, ids)}
        key = f"year={str(date)[:4]}/{C.SHARD_GROUP_PREFIX}{run}/part-{date}-{within.get(date, 0):09d}"
        within[date] = within.get(date, 0) + len(ids)
        task = {"key": key, "publication_date": str(date), "rows": len(ids), "selected_start": start,
                "source": {"kind": "openalex-api-daily", "base": C.BASE_ID, "updates_corpus": corpus.name}}
        with C.locked("shard commit"):
            ec.commit_shard(C.SHARDS_OUT, task, doc, ids, hashes, vectors, stats)
        n += len(ids)
        start = end
        C.log(f"  {key}: {len(ids):,} works ({n:,}/{t.num_rows:,}, {n / (time.time() - t0):,.0f}/s)")
    C.log(f"embedded {n:,} works in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
