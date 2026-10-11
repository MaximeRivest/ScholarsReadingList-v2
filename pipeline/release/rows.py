"""The release's row space: one row per committed embedding, with what every product needs.

    cd search && uv run python ../pipeline/release/rows.py --out DIR

Row order = shard commit order (then shard key), so it is append-only: a later release starts
with the same rows in the same order and adds the newly committed shards at the end. Row numbers
are internal to a release: anything that leaves a release (URLs, the site, other projects) uses
OpenAlex W-numbers.

Columns (numpy, aligned, N rows):
  work.npy      uint64  OpenAlex W-number
  year.npy      int16   publication year (corpus)          type.npy   uint8  index into meta types
  field.npy     uint8   atlas field code                    subfield.npy uint16 atlas subfield code
  smr.npy       bool    ScienceMadeReadable can open it (PMC CC BY research article)
  keep.npy      bool    in the publications index: a publication type (citewalk/science/scope.py)
                        and the smallest W-number among works with byte-identical title+abstract
  by_work.npy   uint32  rows sorted by work (lookups by W-number)
  shards.json   [{dir, rows, first, committed_utc}] in row order
  counts.json   exact publications per year and atlas subfield over the WHOLE corpus (203.5M works,
                embedded or not, duplicates included): what the map's bar race and growth views
                should use instead of counting the 1 % dots
  meta.json     counts, code lists, sources, checks
Field/subfield codes are the atlas build's (its meta.json lists), so the map, its tiles and the
search results share one vocabulary; subfield comes from the OpenAlex primary topic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "citewalk" / "science"))
import scope  # noqa: E402

SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")
PLAN = SHARDS.parent / "plan-summary.json"
TOPICS = Path("/mnt/fast/citewalk/20261007T003120Z-50ce3865/topics.json")
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.time() - T0:7.0f}s {msg}", flush=True)


def committed_shards(plan_id: str) -> list[dict]:
    out = []
    for c in SHARDS.glob("year=*/rg-*/part-*/commit.json"):     # published shards only
        rec = json.loads(c.read_text())
        if rec["plan_id"] != plan_id:
            raise SystemExit(f"{c}: another embedding plan")
        out.append({"dir": str(c.parent.relative_to(SHARDS)), "rows": int(rec["task"]["rows"]),
                    "committed_utc": rec["committed_utc"]})
    out.sort(key=lambda s: (s["committed_utc"], s["dir"]))
    first = 0
    for s in out:
        s["first"] = first
        first += s["rows"]
    return out


def read_corpus(corpus: Path, fields: list[str], subfields: list[str], types: list[str], threads: int):
    """corpus parts -> arrays sorted by work id: id, year, type, field, subfield, smr codes."""
    topics = json.loads(TOPICS.read_text())
    topic_names = pa.array([t["name"] for t in topics])
    sub_of_topic = np.array([subfields.index(t["subfield"]) if t["subfield"] in subfields
                             else subfields.index("Unclassified") for t in topics] + [subfields.index("Unclassified")],
                            np.uint16)
    field_names = pa.array(fields)
    unf = fields.index("Unclassified")
    type_names = pa.array(types)

    def one(path: Path):
        t = pq.read_table(path, columns=["id", "year", "type", "field", "topic", "smr"])
        f = pc.index_in(t["field"], value_set=field_names).to_numpy(zero_copy_only=False)
        tp = pc.index_in(t["topic"], value_set=topic_names).to_numpy(zero_copy_only=False)
        ty = pc.index_in(t["type"], value_set=type_names).to_numpy(zero_copy_only=False)
        return (t["id"].to_numpy().astype(np.uint64),
                t["year"].fill_null(0).to_numpy().clip(-32768, 32767).astype(np.int16),
                np.where(np.isnan(ty.astype(float)), 255, ty).astype(np.uint8),
                np.where(np.isnan(f.astype(float)), unf, f).astype(np.uint8),
                sub_of_topic[np.where(np.isnan(tp.astype(float)), len(topics), tp).astype(np.int64)],
                t["smr"].fill_null(0).to_numpy().astype(bool))

    files = [f for f in sorted((corpus / "parts").glob("*.parquet")) if pq.ParquetFile(f).metadata.num_rows]
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(one, files))
    cols = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    o = np.argsort(cols[0], kind="stable")
    cols = [c[o] for c in cols]
    if np.any(cols[0][1:] == cols[0][:-1]):
        raise SystemExit("duplicate work ids in the corpus")
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, action="append",
                    help="a details corpus; repeat it for the base and the daily updates corpus (pipeline/updates)")
    ap.add_argument("--atlas", type=Path, required=True, help="atlas build directory (its field/subfield lists)")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    if (a.out / "meta.json").exists():
        raise SystemExit(f"{a.out} is finished")
    a.out.mkdir(parents=True, exist_ok=True)
    plan_id = json.loads(PLAN.read_text())["plan_id"]
    atlas = json.loads((a.atlas / "meta.json").read_text())
    fields = [f["name"] for f in atlas["fields"]]
    subfields = [s["name"] for s in atlas["subfields"]]
    types = json.loads((Path("/mnt/fast/citewalk/20261007T003120Z-50ce3865/meta.json")).read_text())["types"]

    shards = committed_shards(plan_id)
    N = shards[-1]["first"] + shards[-1]["rows"]
    log(f"{len(shards):,} committed shards, {N:,} rows")

    work = np.empty(N, np.uint64)
    hashes = np.empty((N, 2), np.uint64)              # BLAKE2b-128 of the full title + abstract

    def load(s):
        d = SHARDS / s["dir"]
        w = np.load(d / "work_ids.npy")
        h = np.load(d / "text_hashes.npy")
        if len(w) != s["rows"] or h.shape != (s["rows"], 16):
            raise SystemExit(f"{d}: rows differ from its commit")
        work[s["first"]:s["first"] + s["rows"]] = w
        hashes[s["first"]:s["first"] + s["rows"]] = h.view(np.uint64)
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(load, shards))
    log("shard ids and text hashes read")

    cols = [read_corpus(c, fields, subfields, types, a.threads) for c in a.corpus]
    cols = [np.concatenate([c[k] for c in cols]) for k in range(6)]
    o = np.argsort(cols[0], kind="stable")
    cid, cyear, ctype, cfield, csub, csmr = [c[o] for c in cols]
    del cols, o
    if np.any(cid[1:] == cid[:-1]):
        raise SystemExit("a work id is in two corpora")
    log(f"corpus read: {len(cid):,} works")
    j = np.searchsorted(cid, work)
    j[j >= len(cid)] = 0
    found = cid[j] == work
    if not found.all():
        raise SystemExit(f"{int((~found).sum()):,} embedded works missing from the corpora {a.corpus}")
    year, typ, field, sub, smr = cyear[j], ctype[j], cfield[j], csub[j], csmr[j]
    sch_codes = [types.index(t) for t in scope.SCHOLARLY_TYPES if t in types]
    all_pub = np.isin(ctype, sch_codes) & (cyear >= 1800) & (cyear <= 2026)
    grid = np.zeros((227, len(subfields)), np.int64)
    np.add.at(grid, (cyear[all_pub].astype(np.int64) - 1800, csub[all_pub].astype(np.int64)), 1)
    (a.out / "counts.json").write_text(json.dumps({
        "what": "publications (scholarly types) per publication year and atlas subfield, whole corpus",
        "years": list(range(1800, 2027)), "subfields": subfields, "counts": grid.tolist(),
        "total_per_year": grid.sum(1).tolist()}))
    del cid, cyear, ctype, cfield, csub, csmr, j

    scholarly = np.isin(typ, [types.index(t) for t in scope.SCHOLARLY_TYPES if t in types])
    # one row per distinct text among publications: the smallest W-number wins
    cand = np.flatnonzero(scholarly)
    o = cand[np.lexsort((work[cand], hashes[cand, 1], hashes[cand, 0]))]
    hs = hashes[o]
    first = np.r_[True, (hs[1:] != hs[:-1]).any(1)]
    keep = np.zeros(N, bool)
    keep[o[first]] = True
    log(f"publications {int(scholarly.sum()):,}; distinct texts kept {int(keep.sum()):,}")

    by_work = np.argsort(work, kind="stable").astype(np.uint32 if N < 2**32 else np.uint64)
    if np.any(work[by_work][1:] == work[by_work][:-1]):
        raise SystemExit("a work id appears in two shards")
    for name, arr in [("work", work), ("year", year), ("type", typ), ("field", field), ("subfield", sub),
                      ("smr", smr), ("keep", keep), ("by_work", by_work)]:
        np.save(a.out / f"{name}.npy", arr)
    (a.out / "shards.json").write_text(json.dumps(shards))
    years = np.bincount(np.clip(year[keep].astype(int) - 1800, 0, 230), minlength=231)
    doc = {"rows": int(N), "shards": len(shards), "embedding_plan": plan_id,
           "publications": int(scholarly.sum()), "kept": int(keep.sum()),
           "duplicates_dropped": int(scholarly.sum() - keep.sum()), "smr": int(smr.sum()),
           "smr_kept": int((smr & keep).sum()),
           "oldest_kept_year_with_100k": int(1800 + np.flatnonzero(years >= 100_000)[0]) if (years >= 100_000).any() else None,
           "types": types, "scholarly_types": sorted(scope.SCHOLARLY_TYPES),
           "fields": fields, "subfields": subfields, "atlas": atlas["id"], "corpus": str(a.corpus[0]),
           "corpora": [str(c) for c in a.corpus],
           "last_commit_utc": shards[-1]["committed_utc"], "seconds": round(time.time() - T0)}
    (a.out / "meta.json.tmp").write_text(json.dumps(doc, indent=1))
    os.replace(a.out / "meta.json.tmp", a.out / "meta.json")
    log(json.dumps({k: v for k, v in doc.items() if k not in ("types", "fields", "subfields", "scholarly_types")}))


if __name__ == "__main__":
    main()
