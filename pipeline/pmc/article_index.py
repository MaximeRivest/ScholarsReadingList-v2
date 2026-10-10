"""One small table describing every article in the mirror (no bodies).

  uv run python article_index.py --mirror /mnt/bulk/pmc/oa-2026-10-05

Writes MIRROR/derived/articles.parquet:
  pmcid, version, pmid, doi (lowercase), license_code, article_type (JATS <article article-type>),
  xml_bytes, txt_bytes, error
"""
from __future__ import annotations

import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TYPE_RE = re.compile(rb'<article\b[^>]*?\barticle-type="([^"]+)"')


def one(path: str) -> pa.Table:
    t = pq.read_table(path, columns=["pmcid", "version", "pmid", "doi", "license_code", "xml", "txt", "error"])
    heads = pc.binary_slice(t["xml"], 0, 6000).to_pylist()
    types = []
    for h in heads:
        m = TYPE_RE.search(h) if h else None
        types.append(m.group(1).decode("utf-8", "replace") if m else None)
    return pa.table({
        "pmcid": t["pmcid"], "version": t["version"], "pmid": t["pmid"],
        "doi": pc.utf8_lower(t["doi"]), "license_code": t["license_code"],
        "article_type": pa.array(types, pa.string()),
        "xml_bytes": pc.binary_length(t["xml"]), "txt_bytes": pc.binary_length(t["txt"]),
        "error": t["error"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=8)
    a = ap.parse_args()
    shards = sorted(str(p) for p in (a.mirror / "shards").glob("shard-*.parquet"))
    plan = json.loads((a.mirror / "plan.json").read_text())
    if len(shards) != plan["shards"]:
        raise SystemExit(f"{len(shards)} shard files, plan has {plan['shards']}")
    with Pool(a.processes) as pool:
        parts = []
        for i, t in enumerate(pool.imap(one, shards, chunksize=4)):
            parts.append(t)
            if i % 200 == 0:
                print(f"  {i}/{len(shards)}", flush=True)
    table = pa.concat_tables(parts)
    if table.num_rows != plan["articles"]:
        raise SystemExit(f"{table.num_rows} rows, plan has {plan['articles']}")
    out = a.mirror / "derived" / "articles.parquet"
    out.parent.mkdir(exist_ok=True)
    pq.write_table(table, str(out) + ".tmp", compression="zstd")
    Path(str(out) + ".tmp").rename(out)
    lic = table.group_by("license_code").aggregate([("pmcid", "count")]).sort_by([("pmcid_count", "descending")])
    typ = table.group_by("article_type").aggregate([("pmcid", "count")]).sort_by([("pmcid_count", "descending")])
    print("licences:", lic.slice(0, 10).to_pylist())
    print("types:", typ.slice(0, 12).to_pylist())


if __name__ == "__main__":
    main()
