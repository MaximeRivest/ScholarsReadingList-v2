"""Build, check and publish a data release for the site (search, map, citation walk). See DATA.md.

    python pipeline/release/release.py build [--gpu-wait]   # new release from what is committed now
    python pipeline/release/release.py publish ID           # check it, point `current` at it, restart the API
    python pipeline/release/release.py rollback             # back to the previous release
    python pipeline/release/release.py nightly              # build if anything changed, then publish
    python pipeline/release/release.py status | gc [--keep 2] [--yes]

A release is a directory releases/<id>/manifest.json naming every product it uses. Products are
built under their own directories (rows/<id>, index/<id>, smr/<id>, tiles/<id>) or shared across
releases when their inputs did not change (corpus, citation index, atlas geography, quantizer,
per-shard positions). `current` is a symlink; switching it is atomic, the old release stays on
disk until gc. Every step is resumable: a finished product (its marker file) is never rebuilt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RELEASE_ROOT", "/mnt/fast/scholarsreadinglist/release"))
SEARCH_PY = REPO / "search" / ".venv" / "bin" / "python"
EMBED_PY = REPO / "pipeline" / "embeddings" / ".venv" / "bin" / "python"
SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")

# Inputs shared by releases until deliberately changed (DATA.md, "What changes when").
CORPUS = Path("/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05")
# 2026-10-09: same geography, placement rule "mode" (DATA.md, citewalk/science/derive_mode.py)
ATLAS = Path("/mnt/fast/science-atlas/build/20261009T190750-tsne-mode")
CITATIONS = Path("/mnt/fast/citewalk/20261007T003120Z-50ce3865")
API = "http://127.0.0.1:8810"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def run(py: Path, script: str, *args: str, cwd: Path = REPO / "search") -> float:
    t = time.time()
    log(f"$ {script} {' '.join(args)}")
    subprocess.run([str(py), str(REPO / script), *args], cwd=cwd, check=True,
                   env={**os.environ, "OPENBLAS_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false"})
    return round(time.time() - t)


def quantizer() -> Path:
    done = sorted(p.parent for p in (ROOT / "quantizer").glob("*/manifest.json"))
    if not done:
        raise SystemExit("no trained quantizer: run pipeline/release/train_quantizer.py once")
    return done[-1]


def latest_commit() -> str:
    last = ""
    for c in SHARDS.glob("year=*/rg-*/part-*/commit.json"):
        last = max(last, json.loads(c.read_text())["committed_utc"])
    return last


def current() -> dict | None:
    m = ROOT / "current" / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else None


def build(gpu_wait: bool, resume: str | None = None) -> str:
    """resume: the id of a build that stopped; its finished products are kept, the rest is built."""
    rid = resume or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rel = ROOT / "releases" / rid
    rel.mkdir(parents=True, exist_ok=bool(resume))
    atlas_id = json.loads((ATLAS / "meta.json").read_text())["id"]
    secs = {}
    rows, index, smr, tiles = ROOT / "rows" / rid, ROOT / "index" / rid, ROOT / "smr" / rid, ROOT / "tiles" / rid
    positions = ROOT / "positions" / atlas_id
    step = run
    if resume:   # a finished product refuses to be rebuilt (its script exits "is finished"): skip it
        done = {"pipeline/release/rows.py": rows / "meta.json", "pipeline/release/build_index.py": index / "manifest.json",
                "search/build_smr.py": smr / "manifest.json", "pipeline/release/tiles.py": tiles / "index.json",
                "pipeline/release/mapdata.py": tiles / "v2" / "map.json"}

        def step(py, script, *args, **kw):
            if script in done and done[script].exists():
                log(f"{script}: finished before, kept")
                return 0
            return run(py, script, *args, **kw)
    secs["rows"] = step(SEARCH_PY, "pipeline/release/rows.py", "--out", str(rows), "--corpus", str(CORPUS),
                       "--atlas", str(ATLAS))
    try:   # GPU step: best effort unless asked to wait (the map then shows what is placed)
        args = ["--atlas", str(ATLAS)] + ([] if gpu_wait else ["--min-free-mib", "2048", "--no-wait"])
        secs["positions"] = step(EMBED_PY, "pipeline/release/positions.py", *args, cwd=REPO / "pipeline" / "embeddings")
    except subprocess.CalledProcessError as e:
        log(f"positions skipped ({e}); tiles will cover the shards placed so far")
    secs["index"] = step(SEARCH_PY, "pipeline/release/build_index.py", "--rows", str(rows),
                        "--quantizer", str(quantizer()), "--out", str(index))
    secs["smr"] = step(SEARCH_PY, "search/build_smr.py", "--out", str(smr))
    secs["tiles"] = step(SEARCH_PY, "pipeline/release/tiles.py", "--rows", str(rows), "--positions", str(positions),
                        "--atlas", str(ATLAS), "--out", str(tiles))
    try:   # the map engine's data (mapdata.py): best effort, the site falls back to its images without it
        secs["mapdata"] = step(SEARCH_PY, "pipeline/release/mapdata.py", "--rows", str(rows), "--xy", str(tiles / "xy.npy"),
                              "--atlas", str(ATLAS), "--corpus", str(CORPUS), "--out", str(tiles / "v2"))
    except subprocess.CalledProcessError as e:
        log(f"map data skipped ({e})")
    rm = json.loads((rows / "meta.json").read_text())
    tm = json.loads((tiles / "index.json").read_text())
    doc = {"id": rid, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "built",
           "paths": {"rows": str(rows), "index": str(index), "smr": str(smr), "tiles": str(tiles),
                     "positions": str(positions), "corpus": str(CORPUS), "atlas": str(ATLAS),
                     "citations": str(CITATIONS), "quantizer": str(quantizer())},
           "embeddings": {"plan": rm["embedding_plan"], "rows": rm["rows"], "last_commit_utc": rm["last_commit_utc"]},
           "counts": {"publications": rm["publications"], "searchable": rm["kept"], "readable_on_site": rm["smr_kept"],
                      "on_map": tm["papers"], "kept_without_position": tm["kept_without_position"]},
           "seconds": secs}
    (rel / "manifest.json").write_text(json.dumps(doc, indent=1))
    log(f"built release {rid}: {json.dumps(doc['counts'])}")
    return rid


def check(rid: str) -> dict:
    """Checks a release before it serves: files present, counts sane vs the current release."""
    m = json.loads((ROOT / "releases" / rid / "manifest.json").read_text())
    p = {k: Path(v) for k, v in m["paths"].items()}
    need = [p["rows"] / "meta.json", p["index"] / "manifest.json", p["index"] / "ivf.faiss",
            p["smr"] / "manifest.json", p["tiles"] / "index.json", p["corpus"] / "works.sqlite",
            p["atlas"] / "reference.npz", p["citations"] / "meta.json"]
    missing = [str(f) for f in need if not f.exists()]
    if missing:
        raise SystemExit(f"release {rid} incomplete: {missing}")
    cur = current()
    if cur and cur["id"] != rid:
        for k in ("searchable", "readable_on_site"):
            if m["counts"][k] < 0.98 * cur["counts"][k]:
                raise SystemExit(f"{k} fell from {cur['counts'][k]:,} to {m['counts'][k]:,}: refusing")
    return m


def health(timeout: float = 600) -> dict:
    t = time.time()
    while time.time() - t < timeout:
        try:
            with urllib.request.urlopen(API + "/health", timeout=10) as r:
                return json.load(r)
        except Exception:
            time.sleep(5)
    raise SystemExit("search API did not come back")


def smoke() -> None:
    q = urllib.request.quote("sleep deprivation and memory")
    with urllib.request.urlopen(f"{API}/search?q={q}&k=5&map=true", timeout=60) as r:
        d = json.load(r)
    if len(d["results"]) < 5 or "map" not in d:
        raise SystemExit(f"smoke search failed: {d}")
    # The walk depends on the citewalk service, not on the release: a failure there is reported
    # but does not roll the release back.
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}/walk?id=W3177828909", timeout=60) as r:
                w = json.load(r)
            if w.get("answers") and w["map"]["placed"]:
                return
        except Exception as e:  # noqa: BLE001
            log(f"walk check: {type(e).__name__} {e}")
        time.sleep(20)
    log("WARNING: citation walk check failed (citewalk service?); search release kept")


def point(rid: str | None) -> None:
    link, tmp = ROOT / "current", ROOT / ".current.tmp"
    if tmp.is_symlink():
        tmp.unlink()
    os.symlink(f"releases/{rid}", tmp)
    os.replace(tmp, link)


def publish(rid: str) -> None:
    m = check(rid)
    prev = current()
    point(rid)
    log(f"current -> {rid}; restarting srl-search")
    subprocess.run(["systemctl", "--user", "restart", "srl-search"], check=True)
    try:
        h = health()
        smoke()
    except SystemExit as e:
        if prev:
            log(f"{e}; rolling back to {prev['id']}")
            point(prev["id"])
            subprocess.run(["systemctl", "--user", "restart", "srl-search"], check=True)
        raise
    m["state"] = "published"
    m["published_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    m["previous"] = prev["id"] if prev else None
    (ROOT / "releases" / rid / "manifest.json").write_text(json.dumps(m, indent=1))
    log(f"published {rid}: {json.dumps(h.get('semantic', {}))[:300]}")


def gc(keep: int, yes: bool) -> None:
    """Delete releases other than current, its previous one and the `keep` newest."""
    cur = current()
    keep_ids = {cur["id"], cur.get("previous")} if cur else set()
    rels = sorted(p.name for p in (ROOT / "releases").glob("*"))
    keep_ids |= set(rels[-keep:])
    for r in rels:
        if r in keep_ids:
            continue
        dirs = [ROOT / k / r for k in ("rows", "index", "smr", "tiles")] + [ROOT / "releases" / r]
        log(("deleting " if yes else "would delete ") + " ".join(str(d) for d in dirs if d.exists()))
        if yes:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--gpu-wait", action="store_true")
    b.add_argument("--resume", metavar="ID", help="finish a build that stopped (keeps its finished products)")
    pb = sub.add_parser("publish"); pb.add_argument("id")
    sub.add_parser("rollback")
    sub.add_parser("nightly")
    sub.add_parser("status")
    g = sub.add_parser("gc"); g.add_argument("--keep", type=int, default=2); g.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        print(build(a.gpu_wait, a.resume))
    elif a.cmd == "publish":
        publish(a.id)
    elif a.cmd == "rollback":
        cur = current()
        if not cur or not cur.get("previous"):
            raise SystemExit("no previous release")
        point(cur["previous"]); subprocess.run(["systemctl", "--user", "restart", "srl-search"], check=True)
        health(); log(f"current -> {cur['previous']}")
    elif a.cmd == "nightly":
        cur = current()
        last = latest_commit()
        if cur and cur["embeddings"]["last_commit_utc"] >= last:
            log("no new embeddings since the current release; nothing to do"); return
        publish(build(gpu_wait=False))
        gc(keep=2, yes=True)        # current + previous stay; ~30-50 GB per release
    elif a.cmd == "status":
        cur = current()
        print(json.dumps({"current": cur and {k: cur[k] for k in ("id", "state", "counts", "embeddings")},
                          "releases": sorted(p.name for p in (ROOT / "releases").glob("*"))}, indent=1))
    elif a.cmd == "gc":
        gc(a.keep, a.yes)


if __name__ == "__main__":
    main()
