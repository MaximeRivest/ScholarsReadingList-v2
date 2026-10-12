# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.3", "pyarrow==26.0.0", "polars==1.44.2"]
# ///
"""Export a data release's paper embeddings as a Hugging Face dataset folder (Parquet).

    uv run pipeline/hf/export.py --out /mnt/bulk/hf-export/openalex-denseon            # current release
    uv run pipeline/hf/export.py --out /tmp/hf-sample --groups 2020 --max-shards 3   # a quick sample

One Parquet row per embedded work, in the release's row order (`row` = the row number the
release's search index uses), so a published FAISS index or any later export lines up with it.
Rows are append-only between releases: a later export holds the same rows, plus new ones.

Folders (each file <= ROWS_PER_FILE rows, row groups of ROWS_PER_GROUP):
    data/<year>/part-NNNNN.parquet      1900 ... 2026, publication year from the OpenAlex snapshot
    data/after-2026/  data/before-1900/  data/no-year/      the few odd dates
    data/updates/<run>/part-NNNNN.parquet                    papers added by the daily updates

Metadata comes from the exact OpenAlex records that were embedded (the backfill's input
snapshot, or the daily updates' content files). Every worker checks the join twice: every row
must find its record, and for the group's first shard the BLAKE2b-128 hash of title + abstract
must equal the hash stored next to the vector (so the title really belongs to the vector).

Two stages: (1) per group, the records are joined and checked once -> work/meta/<group>.parquet;
(2) the dataset files are written in parallel, one job per file. work/ is not part of the dataset.
Resumable: every output is written to a temporary name and renamed; finished outputs are skipped.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

RELEASES = Path("/mnt/fast/scholarsreadinglist/release")
SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")
BACKFILL_MANIFEST = SHARDS.parent / "manifest.json"
BASE_CONTENT = Path("/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts")  # what was embedded
UPDATES = Path("/mnt/fast/scholarsreadinglist/updates")
DIM = 768
ROWS_PER_FILE = 500_000   # ~0.75 GB per file: few files, small enough for a partial download
ROWS_PER_GROUP = 50_000   # ~77 MB row groups: what streaming readers fetch at a time

META_COLUMNS = ["id", "doi", "title", "date", "year", "type", "venue", "field", "subfield", "topic",
                "citations", "is_retracted", "is_open_access", "license"]

SCHEMA = pa.schema([
    ("row", pa.int64()),
    ("openalex_id", pa.string()),
    ("doi", pa.string()),
    ("title", pa.string()),
    ("publication_date", pa.date32()),
    ("year", pa.int16()),
    ("type", pa.string()),
    ("venue", pa.string()),
    ("field", pa.string()),
    ("subfield", pa.string()),
    ("topic", pa.string()),
    ("cited_by_count", pa.int32()),
    ("is_retracted", pa.bool_()),
    ("is_open_access", pa.bool_()),
    ("license", pa.string()),
    ("in_search_index", pa.bool_()),
    ("atlas_x", pa.float32()),
    ("atlas_y", pa.float32()),
    ("text_hash", pa.binary(16)),
    ("embedding", pa.list_(pa.float16(), DIM)),
])


# ---------------------------------------------------------------- 1. which shards go where

def group_name(shard_dir: str) -> str:
    """'year=2020/rg-000003/part-...' -> '2020';  daily shards -> 'updates/<run>'."""
    year_part, rg, _ = shard_dir.split("/")
    if rg.startswith("rg-u"):
        return f"updates/{rg[4:]}"
    year = int(year_part.removeprefix("year="))
    if year < 0:
        return "no-year"
    if year < 1900:
        return "before-1900"
    if year > 2026:
        return "after-2026"
    return str(year)


def plan_groups(shards: list[dict]) -> dict[str, list[dict]]:
    """Shards in row order, cut into contiguous groups (a group must not reappear later)."""
    groups: dict[str, list[dict]] = {}
    last = None
    for s in shards:
        g = group_name(s["dir"])
        if g in groups and g != last:
            raise ValueError(f"group {g} is not contiguous in row order (at {s['dir']})")
        groups.setdefault(g, []).append(s)
        last = g
    return groups


# ---------------------------------------------------------------- 2. the records behind the vectors

def content_files(group: str, shards: list[dict], updates_dir: Path) -> list[Path]:
    if group.startswith("updates/"):
        return sorted((updates_dir / "content").glob("*.parquet"))  # small: all runs, matched by id
    years = sorted({s["dir"].split("/")[0] for s in shards})
    return [BASE_CONTENT / y / "data.parquet" for y in years]


def load_metadata(files: list[Path], ids: np.ndarray) -> pl.DataFrame:
    """One metadata row per id, in the order of `ids`. Fails if any id has no record."""
    want = pl.DataFrame({"_order": np.arange(len(ids), dtype=np.int64), "_wid": ids.astype(np.uint64)})
    recs = (pl.scan_parquet(files)
            .select(META_COLUMNS)
            .with_columns(_wid=pl.col("id").str.strip_prefix("https://openalex.org/W").cast(pl.UInt64))
            .join(want.lazy(), on="_wid", how="inner")
            .collect(engine="streaming")
            .sort("_order"))
    if recs.height != len(ids):  # fewer: a work without its record; more: a work with two records
        raise ValueError(f"{len(ids)} embedded works matched {recs.height} records in {files[0].parent}")
    return recs


def check_text_hashes(files: list[Path], shard: dict, ids: np.ndarray, hashes: np.ndarray) -> int:
    """Re-hash title + abstract for one shard's works; return how many matched (all must)."""
    recs = (pl.scan_parquet(files)
            .select("id", "title", "abstract")
            .with_columns(_wid=pl.col("id").str.strip_prefix("https://openalex.org/W").cast(pl.UInt64))
            .join(pl.LazyFrame({"_wid": ids.astype(np.uint64)}), on="_wid", how="semi")
            .collect(engine="streaming"))
    text = {w: f"{t or ''} {a}".strip() for w, t, a in zip(recs["_wid"], recs["title"], recs["abstract"])}
    ok = 0
    for w, h in zip(ids.tolist(), hashes):
        t = text.get(w)
        if t is not None and hashlib.blake2b(t.encode(), digest_size=16).digest() == h.tobytes():
            ok += 1
    if ok != len(ids):
        raise ValueError(f"{shard['dir']}: only {ok}/{len(ids)} titles+abstracts hash to the stored text hash")
    return ok


# ---------------------------------------------------------------- 3. one group -> its metadata, then its files

def prepare_group(group: str, shards: list[dict], out: Path, updates_dir: Path) -> dict:
    """Stage 1: the group's records, checked, in row order -> work/meta/<group>.parquet."""
    t0 = time.time()
    path = out / "work" / "meta" / f"{group}.parquet"
    if path.exists():
        return {"group": group, "stage": "meta", "skipped": True}
    for s, nxt in zip(shards, shards[1:]):
        assert s["first"] + s["rows"] == nxt["first"], (s["dir"], nxt["dir"])
    ids = np.concatenate([np.load(SHARDS / s["dir"] / "work_ids.npy") for s in shards])
    files = content_files(group, shards, updates_dir)
    first = shards[0]
    checked = check_text_hashes(files, first, ids[:first["rows"]],
                                np.load(SHARDS / first["dir"] / "text_hashes.npy"))
    meta = load_metadata(files, ids)
    assert (meta["_wid"].to_numpy() == ids).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    meta.drop("id", "_order").write_parquet(tmp, compression="lz4")
    os.replace(tmp, path)
    return {"group": group, "stage": "meta", "rows": len(ids), "hash_checked": checked,
            "seconds": round(time.time() - t0, 1)}


def write_file(group: str, i: int, chunk: list[dict], group_start: int, out: Path, rows_dir: Path,
               positions_dir: Path, file_metadata: dict) -> dict:
    """Stage 2: one Parquet file of the dataset: metadata slice + vectors + map position."""
    t0 = time.time()
    final = out / "data" / group / f"part-{i:05d}.parquet"
    if final.exists():
        return {"file": str(final.relative_to(out)), "skipped": True}
    n = sum(s["rows"] for s in chunk)
    a = chunk[0]["first"] - group_start
    m = pl.scan_parquet(out / "work" / "meta" / f"{group}.parquet").slice(a, n).collect()
    vec = np.concatenate([np.load(SHARDS / s["dir"] / "vectors.npy") for s in chunk])
    ids = np.concatenate([np.load(SHARDS / s["dir"] / "work_ids.npy") for s in chunk])
    hsh = np.ascontiguousarray(np.concatenate([np.load(SHARDS / s["dir"] / "text_hashes.npy") for s in chunk]))
    xy = np.concatenate([load_xy(positions_dir / s["dir"], s["rows"]) for s in chunk])
    keep = np.load(rows_dir / "keep.npy", mmap_mode="r")
    rows = np.arange(chunk[0]["first"], chunk[0]["first"] + n, dtype=np.int64)
    assert vec.shape == (n, DIM) and vec.dtype == np.float16 and m.height == n
    assert (m["_wid"].to_numpy() == ids).all()
    table = pa.table({
        "row": rows,
        "openalex_id": ("W" + m["_wid"].cast(pl.String)).to_arrow(),
        "doi": m["doi"].str.strip_prefix("https://doi.org/").to_arrow(),
        "title": m["title"].to_arrow(),
        "publication_date": m["date"].to_arrow(),
        "year": m["year"].cast(pl.Int16).to_arrow(),
        "type": m["type"].to_arrow(),
        "venue": m["venue"].to_arrow(),
        "field": m["field"].to_arrow(),
        "subfield": m["subfield"].to_arrow(),
        "topic": m["topic"].to_arrow(),
        "cited_by_count": m["citations"].cast(pl.Int32).to_arrow(),
        "is_retracted": m["is_retracted"].to_arrow(),
        "is_open_access": m["is_open_access"].to_arrow(),
        "license": m["license"].to_arrow(),
        "in_search_index": np.asarray(keep[rows[0]:rows[-1] + 1]),
        "atlas_x": xy[:, 0],
        "atlas_y": xy[:, 1],
        "text_hash": pa.FixedSizeBinaryArray.from_buffers(pa.binary(16), n, [None, pa.py_buffer(hsh.tobytes())]),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vec.ravel(), type=pa.float16()), DIM),
    }).cast(SCHEMA).replace_schema_metadata(file_metadata)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, row_group_size=ROWS_PER_GROUP, compression="zstd", compression_level=3,
                   use_dictionary=[c for c in ("type", "venue", "field", "subfield", "topic", "license")],
                   write_statistics=[c for c in SCHEMA.names if c not in ("embedding", "text_hash")])
    os.replace(tmp, final)
    return {"file": str(final.relative_to(out)), "rows": n, "seconds": round(time.time() - t0, 1)}


def cut_files(shards: list[dict]) -> list[list[dict]]:
    """Whole shards per file, a new file once ROWS_PER_FILE would be passed."""
    files, cur, n = [], [], 0
    for s in shards:
        if cur and n + s["rows"] > ROWS_PER_FILE:
            files.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += s["rows"]
    if cur:
        files.append(cur)
    return files


def load_xy(d: Path, rows: int) -> np.ndarray:
    p = d / "xy.npy"
    if not p.exists():  # not placed yet: no position
        return np.full((rows, 2), np.nan, dtype=np.float32)
    xy = np.load(p)
    assert xy.shape == (rows, 2), (d, xy.shape)
    return xy.astype(np.float32)


# ---------------------------------------------------------------- 4. the whole release

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="current", help="release id under releases/, or 'current'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12, help="files written at once")
    ap.add_argument("--meta-workers", type=int, default=4, help="groups' metadata prepared at once")
    ap.add_argument("--groups", nargs="*", help="only these groups (e.g. 2020 updates/20261012T010958Z)")
    ap.add_argument("--max-shards", type=int, help="sample: only the first N shards of each group")
    args = ap.parse_args()

    rel = json.loads((RELEASES / ("current" if args.release == "current" else f"releases/{args.release}")
                      / "manifest.json").read_text())
    rows_dir = Path(rel["paths"]["rows"])
    positions_dir = Path(rel["paths"]["positions"])
    updates_dir = Path(rel["paths"]["updates"]).parents[1]
    shards = json.loads((rows_dir / "shards.json").read_text())
    assert sum(s["rows"] for s in shards) == rel["embeddings"]["rows"]
    emb = json.loads(BACKFILL_MANIFEST.read_text())["embedding"]

    groups = plan_groups(shards)
    if args.groups:
        groups = {g: groups[g] for g in args.groups}
    if args.max_shards:
        groups = {g: s[:args.max_shards] for g, s in groups.items()}
    file_metadata = {
        "release": rel["id"], "embedding_model": emb["model"], "model_revision": emb["revision"],
        "document_prompt": emb["document_prompt"], "query_prompt": emb["query_prompt"],
        "input_format": emb["input_format"], "max_seq_length": str(emb["max_seq_length"]),
        "normalized": "L2 (cosine = dot product)", "atlas": Path(rel["paths"]["positions"]).name,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"release {rel['id']}: {sum(len(s) for s in groups.values())} shards in {len(groups)} groups "
          f"-> {args.out}", flush=True)

    reports = []
    order = sorted(groups, key=lambda g: -sum(s["rows"] for s in groups[g]))  # biggest first
    with ProcessPoolExecutor(args.meta_workers) as pool:  # stage 1: memory-heavy, few at a time
        jobs = [pool.submit(prepare_group, g, groups[g], args.out, updates_dir) for g in order]
        for j in as_completed(jobs):
            reports.append(j.result())
            print(json.dumps(reports[-1]), flush=True)
    with ProcessPoolExecutor(args.workers) as pool:       # stage 2: one file per job
        jobs = [pool.submit(write_file, g, i, chunk, groups[g][0]["first"], args.out, rows_dir,
                            positions_dir, file_metadata)
                for g in order for i, chunk in enumerate(cut_files(groups[g]))]
        for k, j in enumerate(as_completed(jobs), 1):
            r = j.result()
            if not r.get("skipped"):
                print(f"[{k}/{len(jobs)}] {json.dumps(r)}", flush=True)

    files = sorted((args.out / "data").rglob("part-*.parquet"))
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    summary = {"release": rel["id"], "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "rows": rows, "files": len(files), "bytes": sum(f.stat().st_size for f in files),
               "searchable": rel["counts"]["searchable"], "embedding": emb,
               "openalex_snapshot": updates_dir.name.removeprefix("openalex-"),  # the base the updates extend
               "updates_corpus": Path(rel["paths"]["updates"]).name, "atlas": file_metadata["atlas"],
               "groups": sorted([r for r in reports if not r.get("skipped")], key=lambda r: r["group"])}
    if not args.groups and not args.max_shards:
        assert rows == rel["embeddings"]["rows"], (rows, rel["embeddings"]["rows"])
    (args.out / "export.json").write_text(json.dumps(summary, indent=1))
    print(f"done: {rows:,} rows in {len(files)} files, {summary['bytes'] / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
