# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.30", "pyarrow>=17", "numpy>=2"]
# ///
"""Live check: does the API path give the same embedded text as the snapshot path, for works in both?

    uv run pipeline/updates/check_api.py [--n 100]       # costs $0.0001 per 100 works

Takes publications from the base content snapshot, fetches them from the API with fetch.py's own request and
conversion, and compares the backfill's text hash (BLAKE2b-128 of title + " " + abstract) and every content
column. The text must match exactly; labels (topic, keywords) and counts may have changed at OpenAlex since.
Run it after changing fetch.py or when OpenAlex changes its API.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch as F  # noqa: E402
import openalex_content as oc  # noqa: E402

SNAPSHOT_2026 = Path("/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts/year=2026/data.parquet")


def text_hash(title, abstract) -> str:
    return hashlib.blake2b(f"{title or ''} {abstract}".strip().encode(), digest_size=16).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    f = pq.ParquetFile(SNAPSHOT_2026)
    t = pl.from_arrow(f.read_row_group(rng.randrange(f.num_row_groups), columns=oc.COLUMNS))
    t = t.filter(pl.col("language") == "en", pl.col("abstract").str.strip_chars() != "",
                 pl.col("type").is_in(list(F.C.SCHOLARLY_TYPES)))
    ids = rng.sample(t["id"].to_list(), min(a.n, t.height))
    cl = F.Client(F.api_key(), max_usd=1, reserve_usd=0)
    works = []
    for i in range(0, len(ids), 100):
        works += cl.get(filter="openalex_id:" + "|".join(x.rsplit("/", 1)[1] for x in ids[i:i + 100]),
                        per_page=100, select=F.SELECT)["results"]
    api = F.to_content(works, F.raw_schema())
    j = api.join(t, on="id", suffix="_snap")
    rows = list(j.iter_rows(named=True))
    same = [text_hash(r["title"], r["abstract"]) == text_hash(r["title_snap"], r["abstract_snap"]) for r in rows]
    # a record OpenAlex changed since the snapshot may have a new abstract or title: only the others must match
    untouched = [r["updated_date"] == r["updated_date_snap"] for r in rows]
    bad = [r["id"] for r, s, u in zip(rows, same, untouched) if u and not s]
    print(f"{len(works)} of {len(ids)} works returned (the rest deleted or merged since); embedded text identical "
          f"for {sum(same)}/{len(rows)}; of the {sum(untouched)} records unchanged since the snapshot, "
          f"{sum(untouched) - len(bad)} identical")
    for c in oc.COLUMNS:
        if c != "id":
            d = int((~j[c].cast(pl.String).eq_missing(j[c + "_snap"].cast(pl.String))).sum())
            if d:
                print(f"  {c:22s} differs for {d}/{j.height} (changed at OpenAlex since the snapshot, or a bug)")
    if bad:
        raise SystemExit(f"the embedded text differs on unchanged records {bad[:5]}: daily vectors would not "
                         "match the backfill's")


if __name__ == "__main__":
    main()
