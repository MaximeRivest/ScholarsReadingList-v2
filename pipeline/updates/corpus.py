"""Daily: the updates corpus, a small companion of the base corpus built from every work fetched since the base.

    search/.venv/bin/python pipeline/updates/corpus.py [--force] [--gc]

Same layout and code as the base corpus (search/build_corpus.py: parts, works.sqlite, keywords; and
pipeline/release/corpus_map_meta.py: map-meta), so the search API and the release read it like the base one.
Rebuilt whole each time from content/*.parquet (cumulative: a quarter of updates is ~10M rows, minutes), into
corpus/<run>/; DONE.json is written last and the directory renamed under the updates lock, so a reader never sees
a half-built corpus. Skipped when no content file was added since the newest corpus.

--gc deletes corpora that no release names and that are not among the newest two.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "search"))
import common as C  # noqa: E402
import build_corpus as bc  # noqa: E402

RELEASES = Path("/mnt/fast/scholarsreadinglist/release/releases")


def quality(out: Path, ids: np.ndarray) -> int:
    """out/quality/: the junk model's record facts (srl_search/quality.py COLUMNS) for the corpus' works, newest
    facts file first (fetch.py writes one per raw file). Works without facts are left out: the model then treats
    them as unknown records. Returns how many works got facts."""
    import pyarrow as pa
    import pyarrow.compute as pc
    files = sorted(C.FACTS.glob("*.parquet"))
    q = out / "quality"
    q.mkdir()
    t = pa.concat_tables([pq.read_table(f) for f in files]) if files else None
    if t is None or t.num_rows == 0:
        np.save(q / "ids.npy", np.zeros(0, np.int64))
        return 0
    # newest record of a work wins: files are in time order; keep each id's last row
    t = t.append_column("order", pa.array(np.arange(t.num_rows)))
    t = t.filter(pc.is_in(t["id"], value_set=pa.array(ids)))
    last = t.group_by("id").aggregate([("order", "max")])
    t = t.filter(pc.is_in(t["order"], value_set=last["order_max"])).sort_by("id")
    np.save(q / "ids.npy", t["id"].to_numpy())
    for c in t.column_names:
        if c in ("id", "order"):
            continue
        a = t[c]
        arr = a.to_numpy(zero_copy_only=False)
        if c == "citation_pct":
            arr = arr.astype(np.float32)          # nulls -> nan, like build_quality.py
        np.save(q / f"{c}.npy", arr)
    (q / "manifest.json").write_text(json.dumps({"works": t.num_rows, "of": int(len(ids)),
                                                 "from": "pipeline/updates facts/*.parquet (API records)"}))
    return t.num_rows


def build(force: bool) -> Path | None:
    files = C.content_files()
    if not files:
        C.log("no updates fetched yet: nothing to build")
        return None
    prev = C.latest_corpus()
    names = [f.name for f in files]
    raws = sorted(f.name for f in C.FACTS.glob("*.parquet"))
    if prev and not force and json.loads((prev / "DONE.json").read_text()).get("raw") == raws \
            and json.loads((prev / "DONE.json").read_text())["content"] == names:
        C.log(f"{prev.name}: already holds every content file")
        return prev
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tmp = C.CORPUS / f".{run}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "parts").mkdir(parents=True)
    t0 = time.time()
    pmc = str(C.BASE_CORPUS / "pmc-by-doi.parquet")   # the base's DOI -> PMC table: same flags as the base
    rows = 0
    for f in files:
        n = pq.ParquetFile(f).metadata.num_row_groups
        rows += bc.one_part((str(f), list(range(n)), str(tmp / "parts" / f.name), pmc))["rows"]
    C.log(f"parts: {rows:,} works from {len(files)} files ({time.time() - t0:.0f}s)")
    # disjoint from the base: a work in both would be searched twice and break the release's row space
    base = np.load(C.BASE_CORPUS / "map-meta" / "id.npy", mmap_mode="r")
    ids = np.concatenate([pq.read_table(p, columns=["id"])["id"].to_numpy() for p in bc.parts(tmp)])
    i = np.minimum(np.searchsorted(base, ids), len(base) - 1)
    clash = int((base[i] == ids).sum())
    if clash:
        raise SystemExit(f"{clash} fetched works are in the base corpus too")
    nq = quality(tmp, ids)
    C.log(f"quality facts for {nq:,} of {len(ids):,} works")
    bc.cmd_sqlite(tmp, threads=8, memory="8GB")
    bc.cmd_keywords(tmp, threads=4, heap_gb=2)
    shutil.rmtree(tmp / "duckdb-tmp", ignore_errors=True)
    subprocess.run([sys.executable, str(REPO / "pipeline" / "release" / "corpus_map_meta.py"), str(tmp)],
                   check=True, env={**os.environ, "PROCS": "4"}, stdout=subprocess.DEVNULL)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "works": rows,
                "base": C.BASE_ID, "base_corpus": str(C.BASE_CORPUS),
                "selection": "OpenAlex API: language en, non-empty abstract, publication types (pipeline/updates)",
                "content": names, "pmc_by_doi": pmc, "quality_facts": nq}
    dates = [d for p in bc.parts(tmp) for d in pq.read_table(p, columns=["date"])["date"].to_pylist() if d]
    manifest["dates"] = {"first": min(dates), "last": max(dates)} if dates else None
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (tmp / "DONE.json").write_text(json.dumps({"content": names, "raw": raws, "works": rows,
                                               "seconds": round(time.time() - t0)}))
    final = C.CORPUS / run
    with C.locked("corpus swap"):
        os.replace(tmp, final)
    C.log(f"corpus {final.name}: {rows:,} works in {time.time() - t0:.0f}s")
    return final


def gc() -> None:
    named = set()
    for m in RELEASES.glob("*/manifest.json"):
        u = json.loads(m.read_text())["paths"].get("updates")
        if u:
            named.add(Path(u).name)
    done = sorted(p.parent.name for p in C.CORPUS.glob("*/DONE.json"))
    for name in done[:-2]:
        if name not in named:
            C.log(f"gc: deleting corpus {name}")
            shutil.rmtree(C.CORPUS / name)
    for t in C.CORPUS.glob(".*.tmp"):
        if time.time() - t.stat().st_mtime > 86400:
            shutil.rmtree(t, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if nothing new was fetched")
    ap.add_argument("--gc", action="store_true", help="then delete corpora no release uses (keeps the newest two)")
    a = ap.parse_args()
    C.CORPUS.mkdir(parents=True, exist_ok=True)
    build(a.force)
    if a.gc:
        gc()


if __name__ == "__main__":
    main()
