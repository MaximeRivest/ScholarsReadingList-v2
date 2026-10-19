"""Every authorship of the raw OpenAlex works snapshot as one flat row: the input of author identity (authors.py).

    nice -n 10 ionice -c3 search/.venv/bin/python pipeline/identity/extract_authorships.py [--procs 3]

One row per (work, author position), including authorships OpenAlex left without an author record:
  work UBIGINT, pos SMALLINT (0-based place in the byline), n_authors SMALLINT, year SMALLINT, type VARCHAR,
  is_xpac BOOLEAN, author UBIGINT (A-number, NULL = no record), author_name VARCHAR (the record's display name),
  author_orcid VARCHAR (the record's ORCID, bare), raw_name VARCHAR (as printed), raw_orcid VARCHAR (the ORCID the
  publisher sent, bare), insts UBIGINT[] (I-numbers this authorship is linked to), countries VARCHAR[]
Writes /mnt/fast/scholarsreadinglist/identity/<snapshot>/authorships/<partition>__<file>.parquet (resumable: a file is
written under a temporary name and renamed) and DONE.json. ~1.0B rows; reads only the authorship columns.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WORKS = Path("/mnt/bulk/openalex/works")
SNAPSHOT = "openalex-2026-09-23"
OUT = Path(f"/mnt/fast/scholarsreadinglist/identity/{SNAPSHOT}/authorships")

Q = """
SELECT CAST(substr(id, 23) AS UBIGINT) AS work, CAST(t.pos - 1 AS SMALLINT) AS pos,
       CAST(len(authorships) AS SMALLINT) AS n_authors, CAST(publication_year AS SMALLINT) AS year, type,
       coalesce(is_xpac, false) AS is_xpac,
       TRY_CAST(substr(t.a.author.id, 23) AS UBIGINT) AS author, t.a.author.display_name AS author_name,
       nullif(replace(t.a.author.orcid, 'https://orcid.org/', ''), '') AS author_orcid,
       t.a.raw_author_name AS raw_name,
       nullif(upper(regexp_replace(t.a.raw_orcid, '^(https?://)?(www\\.)?orcid\\.org/', '')), '') AS raw_orcid,
       list_transform(t.a.institutions, i -> TRY_CAST(substr(i.id, 23) AS UBIGINT)) AS insts,
       t.a.countries AS countries
FROM read_parquet(?) w, UNNEST(w.authorships) WITH ORDINALITY AS t(a, pos)
"""


def one(src: str) -> tuple[str, int, float]:
    import duckdb
    p = Path(src)
    dst = OUT / f"{p.parent.name}__{p.stem}.parquet"
    if dst.exists():
        return dst.name, -1, 0.0
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='8GB'; SET preserve_insertion_order=false")
    tmp = dst.with_suffix(".tmp")
    con.execute(f"COPY ({Q}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)", [src])
    n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(tmp)]).fetchone()[0]
    os.replace(tmp, dst)
    return dst.name, n, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first N files only (trial)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in WORKS.glob("updated_date=*/part_*.parquet"))
    if a.limit:
        files = files[:a.limit]
    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(a.procs) as ex:
        for i, (name, n, s) in enumerate(ex.map(one, files), 1):
            total += max(n, 0)
            if i % 50 == 0 or i == len(files):
                print(f"{time.time() - t0:7.0f}s {i}/{len(files)} files, {total:,} rows this run", flush=True)
    if not a.limit:
        (OUT / "DONE.json").write_text(json.dumps({"files": len(files), "snapshot": SNAPSHOT,
                                                   "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
