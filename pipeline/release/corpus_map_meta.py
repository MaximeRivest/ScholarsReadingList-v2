"""Per-paper facts the map needs from the details corpus, as flat arrays (built once per corpus).

    search/.venv/bin/python pipeline/release/corpus_map_meta.py CORPUS_DIR

Writes CORPUS_DIR/map-meta/: id.npy (int64, sorted W-numbers), citations.npy (int32),
topic.npy (uint16 code, 0 = none), topics.json (code -> OpenAlex primary topic name).
~203M rows, ~3 GB, a few minutes with many processes.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq


def read(f: str):
    pf = pq.ParquetFile(f)
    if "id" not in pf.schema_arrow.names:          # an empty part (no rows, no schema)
        return np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), []
    t = pf.read(columns=["id", "citations", "topic"])
    topic = t["topic"].combine_chunks().dictionary_encode()
    return (t["id"].to_numpy(), t["citations"].fill_null(0).to_numpy().astype(np.int32),
            topic.indices.fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int32),
            topic.dictionary.to_pylist())


def main() -> None:
    corpus = Path(sys.argv[1])
    out = corpus / "map-meta"
    if (out / "topics.json").exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    files = sorted(str(f) for f in (corpus / "parts").glob("*.parquet"))
    names: dict[str, int] = {}
    ids, cit, top = [], [], []
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "16"))) as ex:
        for n, (i, c, t, d) in enumerate(ex.map(read, files, chunksize=4)):
            codes = np.array([names.setdefault(x, len(names) + 1) for x in d] + [0], np.uint16)
            ids.append(i); cit.append(c); top.append(codes[t])        # -1 -> last entry (0 = none)
            if n % 100 == 0:
                print(f"{n + 1}/{len(files)} files {time.time() - t0:.0f}s", flush=True)
    ids, cit, top = np.concatenate(ids), np.concatenate(cit), np.concatenate(top)
    o = np.argsort(ids, kind="stable")
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(exist_ok=True)
    np.save(tmp / "id.npy", ids[o]); np.save(tmp / "citations.npy", cit[o]); np.save(tmp / "topic.npy", top[o])
    (tmp / "topics.json").write_text(json.dumps([""] + list(names)))
    os.replace(tmp, out)
    print(f"{len(ids):,} papers, {len(names):,} topics, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
