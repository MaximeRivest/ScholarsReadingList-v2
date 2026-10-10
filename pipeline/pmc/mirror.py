"""Mirror the PubMed Central Open Access full texts (NCBI's free AWS bucket) onto local disk.

Source: https://pmc-oa-opendata.s3.amazonaws.com  (no login, no cost)
  metadata/PMC123.4.json   pmid, doi, license_code, xml_url/text_url with md5, ...
  PMC123.4/PMC123.4.xml    JATS full text (what ScienceMadeReadable reads)
  PMC123.4/PMC123.4.txt    NCBI's plain-text rendering

Output (one directory, resumable):
  plan.json                inventory date, article count, shard size, sha256 of the id list
  ids.txt                  the pinned list "PMC123.4", sorted by PMCID number
  shards/shard-000123.parquet   up to SHARD_SIZE articles, zstd-compressed columns
  shards/shard-000123.json      commit record (written last: its presence means done)
  status.json              live progress

Each shard is written to a temporary name, checked, then renamed; its commit record
is written after. An interrupted run redoes only unfinished shards. Every XML/TXT body
is checked against the md5 that NCBI publishes in the metadata; bodies are stored as
exact bytes.

Commands:
  mirror.py plan  --inventory latest.txt --inventory-date 2026-10-05 --out DIR
  mirror.py run   --out DIR [--concurrency 192] [--shards-in-flight 6] [--limit-shards N]
  mirror.py verify --out DIR
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
SHARD_SIZE = int(os.environ.get("PMC_MIRROR_SHARD_SIZE", "5000"))  # override only for tests
FORMAT_VERSION = 1
ID_RE = re.compile(r"^PMC(\d+)\.(\d+)$")

SCHEMA = pa.schema([
    ("pmcid", pa.string()),            # "PMC123"
    ("version", pa.int32()),
    ("metadata", pa.string()),         # the JSON record exactly as published
    ("license_code", pa.string()),
    ("doi", pa.string()),
    ("pmid", pa.int64()),
    ("xml", pa.large_binary()),        # exact bytes, md5-verified; null if none published
    ("txt", pa.large_binary()),
    ("error", pa.string()),            # null when everything published was fetched and verified
])


# ---------------------------------------------------------------- plan

def make_plan(inventory: Path, inventory_date: str, out: Path) -> dict:
    ids = [l.strip() for l in inventory.read_text().split() if l.strip()]
    for i in ids:
        if not ID_RE.match(i):
            raise SystemExit(f"bad id in inventory: {i!r}")
    ids.sort(key=lambda s: (int(ID_RE.match(s).group(1)), int(ID_RE.match(s).group(2))))
    if len({ID_RE.match(i).group(1) for i in ids}) != len(ids):
        raise SystemExit("inventory has several versions of one article; keep only the latest")
    body = "\n".join(ids) + "\n"
    digest = hashlib.sha256(body.encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "format_version": FORMAT_VERSION,
        "source": BUCKET,
        "inventory_date": inventory_date,
        "articles": len(ids),
        "shard_size": SHARD_SIZE,
        "shards": (len(ids) + SHARD_SIZE - 1) // SHARD_SIZE,
        "ids_sha256": digest,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    existing = out / "plan.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if old["ids_sha256"] != digest or old["shard_size"] != SHARD_SIZE:
            raise SystemExit(f"{out} already holds a different plan; use a new directory")
        return old
    (out / "ids.txt").write_text(body)
    tmp = out / "plan.json.tmp"
    tmp.write_text(json.dumps(plan, indent=1))
    tmp.rename(existing)
    return plan


def load_plan(out: Path) -> tuple[dict, list[str]]:
    plan = json.loads((out / "plan.json").read_text())
    body = (out / "ids.txt").read_text()
    if hashlib.sha256(body.encode()).hexdigest() != plan["ids_sha256"]:
        raise SystemExit("ids.txt does not match plan.json")
    return plan, body.split()


def shard_paths(out: Path, n: int) -> tuple[Path, Path]:
    d = out / "shards"
    return d / f"shard-{n:06d}.parquet", d / f"shard-{n:06d}.json"


# ---------------------------------------------------------------- fetching

class Fetcher:
    def __init__(self, session: aiohttp.ClientSession, concurrency: int):
        self.session = session
        self.sem = asyncio.Semaphore(concurrency)
        self.requests = 0
        self.bytes = 0
        self.retries = 0

    async def get(self, key: str, attempts: int = 8) -> bytes | None:
        """Body of an object, None if it does not exist (404). Retries everything else."""
        delay = 1.0
        last = ""
        for attempt in range(attempts):
            try:
                async with self.sem:
                    async with self.session.get(BUCKET + key) as r:
                        self.requests += 1
                        if r.status == 404:
                            return None
                        if r.status == 200:
                            body = await r.read()
                            self.bytes += len(body)
                            return body
                        last = f"HTTP {r.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"[:200]
            self.retries += 1
            await asyncio.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, 60)
        raise RuntimeError(f"{key}: {last}")


def md5_of(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"[?&]md5=([0-9a-f]{32})", url)
    return m.group(1) if m else None


def key_of(url: str) -> str:
    """s3://pmc-oa-opendata/PMC1.1/PMC1.1.xml?md5=... -> PMC1.1/PMC1.1.xml"""
    u = url.split("?", 1)[0]
    prefix = "s3://pmc-oa-opendata/"
    if not u.startswith(prefix):
        raise ValueError(f"unexpected url {url}")
    return u[len(prefix):]


async def fetch_body(f: Fetcher, url: str | None) -> tuple[bytes | None, str | None]:
    """(body, error). No url published: (None, None)."""
    if not url:
        return None, None
    want = md5_of(url)
    for _ in range(3):
        body = await f.get(key_of(url))
        if body is None:
            return None, f"missing {key_of(url)}"
        if want is None or hashlib.md5(body).hexdigest() == want:
            return body, None
        f.retries += 1
    return None, f"md5 mismatch {key_of(url)}"


async def fetch_article(f: Fetcher, vid: str) -> dict:
    m = ID_RE.match(vid)
    row = {"pmcid": f"PMC{m.group(1)}", "version": int(m.group(2)), "metadata": None,
           "license_code": None, "doi": None, "pmid": None, "xml": None, "txt": None, "error": None}
    try:
        raw = await f.get(f"metadata/{vid}.json")
        if raw is None:
            row["error"] = "metadata missing"
            return row
        meta = json.loads(raw)
        row["metadata"] = raw.decode("utf-8")
        row["license_code"] = meta.get("license_code")
        row["doi"] = meta.get("doi")
        pmid = meta.get("pmid")
        row["pmid"] = int(pmid) if isinstance(pmid, (int, str)) and str(pmid).isdigit() else None
        (xml, e1), (txt, e2) = await asyncio.gather(fetch_body(f, meta.get("xml_url")),
                                                    fetch_body(f, meta.get("text_url")))
        row["xml"], row["txt"] = xml, txt
        errors = [e for e in (e1, e2) if e]
        row["error"] = "; ".join(errors) or None
    except Exception as e:  # recorded per article; `verify` lists them, a later run can repair
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    return row


# ---------------------------------------------------------------- shards

def write_shard(out: Path, n: int, ids: list[str], rows: list[dict]) -> dict:
    path, commit = shard_paths(out, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=6,
                   row_group_size=1000, use_dictionary=["license_code"], write_statistics=["pmcid"])
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    back = pq.read_table(tmp, columns=["pmcid", "version"])
    expect = [(f"PMC{ID_RE.match(i).group(1)}", int(ID_RE.match(i).group(2))) for i in ids]
    if list(zip(back["pmcid"].to_pylist(), back["version"].to_pylist())) != expect:
        tmp.unlink()
        raise RuntimeError(f"shard {n}: rows read back differ from the plan")
    tmp.rename(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = {
        "shard": n, "rows": len(rows), "first": ids[0], "last": ids[-1],
        "with_xml": sum(r["xml"] is not None for r in rows),
        "with_txt": sum(r["txt"] is not None for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "raw_bytes": sum(len(r["xml"] or b"") + len(r["txt"] or b"") for r in rows),
        "file_bytes": path.stat().st_size, "sha256": sha,
        "committed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ctmp = commit.with_suffix(".json.tmp")
    ctmp.write_text(json.dumps(rec))
    ctmp.rename(commit)
    return rec


def done_shards(out: Path) -> set[int]:
    d = out / "shards"
    if not d.exists():
        return set()
    return {int(p.stem.split("-")[1]) for p in d.glob("shard-*.json")}


# ---------------------------------------------------------------- run

async def run(out: Path, concurrency: int, in_flight: int, limit: int | None) -> None:
    plan, ids = load_plan(out)
    size = plan["shard_size"]
    done = done_shards(out)
    todo = [n for n in range(plan["shards"]) if n not in done]
    if limit is not None:
        todo = todo[:limit]
    for stale in (out / "shards").glob("*.tmp") if (out / "shards").exists() else []:
        stale.unlink()
    print(f"{len(done)} shards done, {len(todo)} to fetch", flush=True)

    timeout = aiohttp.ClientTimeout(total=180, sock_connect=20, sock_read=60)
    conn = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300, enable_cleanup_closed=True)
    started = time.time()
    committed = {"shards": 0, "rows": 0, "raw": 0, "file": 0, "errors": 0}
    status_path = out / "status.json"

    async with aiohttp.ClientSession(timeout=timeout, connector=conn,
                                     headers={"User-Agent": "scholarsreadinglist-pmc-mirror/1"}) as session:
        f = Fetcher(session, concurrency)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for n in todo:
            queue.put_nowait(n)

        async def worker() -> None:
            while True:
                try:
                    n = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                chunk = ids[n * size:(n + 1) * size]
                rows = await asyncio.gather(*(fetch_article(f, v) for v in chunk))
                rec = await asyncio.to_thread(write_shard, out, n, chunk, list(rows))
                committed["shards"] += 1
                committed["rows"] += rec["rows"]
                committed["raw"] += rec["raw_bytes"]
                committed["file"] += rec["file_bytes"]
                committed["errors"] += rec["errors"]

        async def reporter() -> None:
            while True:
                el = max(time.time() - started, 1e-9)
                total_done = len(done) + committed["shards"]
                left_rows = max(plan["articles"] - (len(done) * size) - committed["rows"], 0)
                rate = committed["rows"] / el
                st = {
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "shards_done": total_done, "shards_total": plan["shards"],
                    "articles_this_run": committed["rows"], "articles_per_s": round(rate, 1),
                    "requests": f.requests, "retries": f.retries,
                    "download_MB_per_s": round(f.bytes / el / 1e6, 1),
                    "raw_GB_this_run": round(committed["raw"] / 1e9, 2),
                    "disk_GB_this_run": round(committed["file"] / 1e9, 2),
                    "article_errors_this_run": committed["errors"],
                    "eta_hours": round(left_rows / rate / 3600, 1) if rate > 0 else None,
                }
                tmp = status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(st, indent=1))
                tmp.rename(status_path)
                print(json.dumps(st), flush=True)
                await asyncio.sleep(30)

        rep = asyncio.create_task(reporter())
        try:
            await asyncio.gather(*(worker() for _ in range(in_flight)))
        finally:
            rep.cancel()
    el = time.time() - started
    print(f"finished {committed['shards']} shards, {committed['rows']} articles in {el/3600:.2f} h; "
          f"{committed['errors']} article errors", flush=True)


# ---------------------------------------------------------------- verify

def verify(out: Path) -> int:
    plan, ids = load_plan(out)
    size = plan["shard_size"]
    done = done_shards(out)
    missing = [n for n in range(plan["shards"]) if n not in done]
    bad = 0
    totals = {"rows": 0, "with_xml": 0, "with_txt": 0, "errors": 0, "file_bytes": 0, "raw_bytes": 0}
    for n in sorted(done):
        path, commit = shard_paths(out, n)
        rec = json.loads(commit.read_text())
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
            print(f"shard {n}: file missing or checksum differs"); bad += 1; continue
        if rec["rows"] != len(ids[n * size:(n + 1) * size]):
            print(f"shard {n}: wrong row count"); bad += 1; continue
        for k in totals:
            totals[k] += rec[k]
    print(json.dumps({"shards_done": len(done), "shards_missing": len(missing), "bad": bad, **totals}, indent=1))
    return 1 if bad or missing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--inventory-date", required=True); p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--concurrency", type=int, default=192)
    r.add_argument("--shards-in-flight", type=int, default=6)
    r.add_argument("--limit-shards", type=int)
    v = sub.add_parser("verify"); v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(make_plan(a.inventory, a.inventory_date, a.out), indent=1))
    elif a.cmd == "run":
        asyncio.run(run(a.out, a.concurrency, a.shards_in_flight, a.limit_shards))
    else:
        sys.exit(verify(a.out))


if __name__ == "__main__":
    main()
