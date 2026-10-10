"""The semantic index of a release: every kept row (publications, one per distinct text).

    cd search && uv run python ../pipeline/release/build_index.py --rows ROWS --quantizer Q --out OUT

Copies the trained (empty) index from train_quantizer.py and adds the kept rows' vectors with
id = row number, read in place from the embedding shards. The output has the format
srl_search.engine.Engine already reads (manifest.json, ivf.faiss, work_ids.npy, by_work.npy):
ids are row positions, `shards` is the rows' shard list, so reranking reads the exact vectors
and `/similar` and map placement work for every embedded row, while search returns kept rows
only (keep.npy, a hard link of the rows' column, tells eval_index.py which rows count).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np

SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--quantizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    if (a.out / "manifest.json").exists():
        raise SystemExit(f"{a.out} is finished")
    a.out.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(a.threads)
    t0 = time.time()
    rows_meta = json.loads((a.rows / "meta.json").read_text())
    shards = json.loads((a.rows / "shards.json").read_text())
    keep = np.load(a.rows / "keep.npy")
    qman = json.loads((a.quantizer / "manifest.json").read_text())
    if qman["types"] != rows_meta["scholarly_types"]:
        raise SystemExit("quantizer and rows were made for different publication types")
    index = faiss.read_index(str(a.quantizer / "trained.faiss"))
    if index.ntotal:
        raise SystemExit("trained index is not empty")
    batch_x, batch_i, pending, added = [], [], 0, 0

    def flush():
        nonlocal batch_x, batch_i, pending, added
        if pending:
            index.add_with_ids(np.concatenate(batch_x), np.concatenate(batch_i))
            added += pending
        batch_x, batch_i, pending = [], [], 0

    for n, s in enumerate(shards):
        k = keep[s["first"]:s["first"] + s["rows"]]
        if not k.any():
            continue
        x = np.load(SHARDS / s["dir"] / "vectors.npy")
        rows = np.flatnonzero(k)
        batch_x.append(x[rows].astype(np.float32))
        batch_i.append(rows.astype(np.int64) + s["first"])
        pending += len(rows)
        if pending >= 500_000:
            flush()
            print(f"  added {added:,} / {int(keep.sum()):,} ({added / (time.time() - t0):,.0f}/s)", flush=True)
    flush()
    if index.ntotal != int(keep.sum()):
        raise SystemExit(f"index holds {index.ntotal:,}, rows keep {int(keep.sum()):,}")
    faiss.write_index(index, str(a.out / "ivf.faiss.tmp"))
    os.replace(a.out / "ivf.faiss.tmp", a.out / "ivf.faiss")
    for name, src in (("work_ids.npy", "work.npy"), ("by_work.npy", "by_work.npy"), ("keep.npy", "keep.npy")):
        if not (a.out / name).exists():
            os.link(a.rows / src, a.out / name)
    sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)])
    dates = sorted(s["dir"].split("/")[-1][5:15] for s in shards)
    doc = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "min_date": dates[0], "papers": int(index.ntotal), "rows": rows_meta["rows"],
           "shards": [{"dir": s["dir"], "rows": s["rows"]} for s in shards],
           "embedding_plan_id": rows_meta["embedding_plan"], "index_dir": str(a.out),
           "collection": "publications: scholarly types, one row per distinct title+abstract",
           "quantizer": qman["id"],
           "index": {"factory": qman["factory"], "metric": "inner_product", "list_size_max": int(sizes.max()),
                     "list_size_mean": float(sizes.mean()), "empty_lists": int((sizes == 0).sum()),
                     "file_bytes": (a.out / "ivf.faiss").stat().st_size},
           "seconds": {"add": round(time.time() - t0)}, "threads": a.threads}
    (a.out / "manifest.json.tmp").write_text(json.dumps(doc, indent=1))
    os.replace(a.out / "manifest.json.tmp", a.out / "manifest.json")
    print(json.dumps({k: v for k, v in doc.items() if k != "shards"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
