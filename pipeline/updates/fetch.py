# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.30", "pyarrow>=17", "numpy>=2"]
# ///
"""Daily: new English publications with an abstract, from the OpenAlex API, as rows of the content table.

    uv run pipeline/updates/fetch.py [--max-usd 0.8] [--dry-run]

The free public snapshot is refreshed four times a year; the API every day. Between two snapshots this fetches
the works the base does not hold yet, in the same columns `openalex_content.py` makes from the snapshot (its own
`flatten`, so the text that gets embedded is identical: checked on real works, see README).

What is asked for: language en, a non-empty abstract, a publication type (common.SCHOLARLY_TYPES), with OpenAlex's
"expansion pack" records included like in the snapshot. Which papers: publication dates from 30 days before the
base snapshot to 60 days ahead. Free API filters cannot select "created since" (a paid filter), so the API is
asked how many works each (publication date, field) cell holds; a cell is downloaded again only when that count
changed since the last time it was downloaded. Costs (2026-10): a count or a page of 100 works = $0.0001;
the free key gives $1/day (1 million works), no key $0.10/day.

Missed by design until the next snapshot: works published long ago that OpenAlex adds now, and changes to works
already held (new abstracts, corrected titles, citation counts, deletions, merges).

Writes, under common.UPDATES: content/<run>-<n>.parquet (new works only), raw/<run>-<n>.jsonl.gz (the API's
records for them, to rebuild the content if the flattening changes), state.json (cell counts), runs/<run>.json.
Budget-aware: it stops when the day's budget nears its end and continues where it stopped on the next run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import openalex_content as oc  # noqa: E402

API = "https://api.openalex.org/works"
FILTER = "language:en,has_abstract:true,type:" + "|".join(C.SCHOLARLY_TYPES)
LOOKBACK_DAYS, AHEAD_DAYS = 30, 60
SELECT = ("id,doi,title,display_name,type,language,publication_date,publication_year,authorships,primary_location,"
          "best_oa_location,biblio,primary_topic,keywords,mesh,open_access,funders,awards,cited_by_count,"
          "referenced_works_count,referenced_works,is_retracted,updated_date,created_date,abstract_inverted_index,"
          # what the search's junk model reads about a record (search/build_quality.py; authors = len(authorships))
          "is_xpac,is_paratext,indexed_in,locations_count,fwci,citation_normalized_percentile")
QUALITY_FIELDS = ("is_xpac", "is_paratext", "indexed_in", "locations_count", "fwci", "citation_normalized_percentile")
FLUSH_WORKS = 20_000
PRICE = 0.0001        # a count, a group-by or a page of up to 100 works


def api_key() -> str | None:
    k = os.environ.get("OPENALEX_API_KEY")
    f = Path.home() / ".config" / "openalex" / "api_key"
    return (k or (f.read_text().strip() if f.exists() else "")) or None


class Client:
    """GET with retries; keeps the budget left today from the API's own headers."""

    def __init__(self, key: str | None, max_usd: float, reserve_usd: float):
        self.key, self.max_usd, self.reserve = key, max_usd, reserve_usd
        self.spent = 0.0
        self.left: float | None = None     # USD left today (free + prepaid), as last reported
        self.calls = 0
        self.lock = threading.Lock()

    def can_spend(self, usd: float) -> bool:
        with self.lock:
            if self.spent + usd > self.max_usd:
                return False
            return self.left is None or self.left - usd >= self.reserve

    def get(self, **params) -> dict:
        params = {**params, "include_xpac": "true"}
        if self.key:
            params["api_key"] = self.key
        url = API + "?" + urllib.parse.urlencode(params, safe=":,|*")
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "scholarsreadinglist-updates"}),
                                            timeout=120) as r:
                    body = json.load(r)
                    h = r.headers
                with self.lock:
                    self.calls += 1
                    self.spent += float(h.get("X-RateLimit-Cost-USD") or PRICE)
                    if h.get("X-RateLimit-Remaining-USD") is not None:
                        self.left = float(h["X-RateLimit-Remaining-USD"]) + float(h.get("X-RateLimit-Prepaid-Remaining-USD") or 0)
                return body
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # too many requests per second, or today's budget spent: wait and retry; give up for today
                    # only when it keeps saying the budget is empty (a single 429 has said so wrongly)
                    left = e.headers.get("X-RateLimit-Remaining-USD")
                    C.log(f"  429 (budget left: {left}): {e.read()[:200]!r}")
                    if attempt >= 3 and left is not None and float(left) < PRICE:
                        raise BudgetSpent("the API says today's budget is spent") from e
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                if e.code >= 500:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(f"{e.code} from {url.split('api_key=')[0]}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"no answer after 8 tries: {url.split('api_key=')[0]}")


class BudgetSpent(Exception):
    pass


# ---------------------------------------------------------------- what we hold already

class Known:
    """W-numbers held: the base corpus (sorted array, 203M) and every work fetched by earlier runs."""

    def __init__(self):
        self.base = np.load(C.BASE_CORPUS / "map-meta" / "id.npy", mmap_mode="r")
        self.extra: set[int] = set()
        for f in C.content_files():
            ids = pq.read_table(f, columns=["id"])["id"].to_pylist()
            self.extra.update(int(i.rsplit("W", 1)[1]) for i in ids)

    def __contains__(self, w: int) -> bool:
        if w in self.extra:
            return True
        i = int(np.searchsorted(self.base, w))
        return i < len(self.base) and int(self.base[i]) == w


# ---------------------------------------------------------------- API records -> content rows

def raw_schema() -> pa.Schema:
    """The snapshot's raw works schema (saved once: the snapshot files are replaced at each release)."""
    path = C.UPDATES / "raw-schema.arrow"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        s = pq.read_schema(C.RAW_SCHEMA_SOURCE)
        with ipc.new_file(str(path) + ".tmp", s) as w:
            w.write_table(s.empty_table())
        os.replace(str(path) + ".tmp", path)
    with ipc.open_file(path) as r:
        return r.schema


def _ts(v: str | None):
    if not v:
        return None
    t = dt.datetime.fromisoformat(v)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def to_raw(works: list[dict], schema: pa.Schema) -> pa.Table:
    """API JSON -> the snapshot's raw columns (dates typed, the abstract index as its JSON text)."""
    rows = []
    for w in works:
        r = {k: w.get(k) for k in schema.names}
        a = w.get("abstract_inverted_index")
        r["abstract_inverted_index"] = None if a is None else json.dumps(a)
        r["publication_date"] = dt.date.fromisoformat(w["publication_date"]) if w.get("publication_date") else None
        r["updated_date"], r["created_date"] = _ts(w.get("updated_date")), _ts(w.get("created_date"))
        rows.append(r)
    return pa.Table.from_pylist(rows, schema=schema)


def to_content(works: list[dict], schema: pa.Schema):
    """API JSON -> content rows (openalex_content.flatten), kept only if the backfill's selection takes them."""
    import polars as pl
    flat = oc.flatten(pl.from_arrow(to_raw(works, schema)).lazy()).collect()
    return flat.filter((pl.col("language") == "en")
                       & pl.col("abstract").is_not_null() & (pl.col("abstract").str.strip_chars() != "")
                       & pl.col("type").is_in(list(C.SCHOLARLY_TYPES)))


# search/build_quality.py's encodings (its SOURCE_TYPES and INDEXES): the same record facts, from API records
SOURCE_TYPES = ["journal", "repository", "ebook platform", "book series", "conference", "other", "metadata"]
INDEXES = ["crossref", "pubmed", "doaj", "arxiv", "datacite"]


def facts(works: list[dict]) -> pa.Table:
    """The junk model's record facts (srl_search/quality.py COLUMNS) for the works whose record has every field."""
    rows = []
    for w in works:
        if not all(k in w for k in QUALITY_FIELDS):
            continue
        src = (w.get("primary_location") or {}).get("source") or {}
        st, sid = src.get("type"), src.get("id")
        pct = (w.get("citation_normalized_percentile") or {}).get("value")
        rows.append({"id": int(w["id"].rsplit("W", 1)[1]), "is_xpac": bool(w.get("is_xpac")),
                     "is_paratext": bool(w.get("is_paratext")),
                     "source_type": SOURCE_TYPES.index(st) + 1 if st in SOURCE_TYPES else 0,
                     "source_is_core": bool(src.get("is_core")), "source_in_doaj": bool(src.get("is_in_doaj")),
                     "indexed_in": sum(1 << b for b, n in enumerate(INDEXES) if n in (w.get("indexed_in") or [])),
                     "refs": min(w.get("referenced_works_count") or 0, 32767),
                     "authors": min(len(w.get("authorships") or []), 32767),
                     "locations": min(w.get("locations_count") or 0, 32767),
                     "citation_pct": None if pct is None else float(pct),
                     "source": int(sid.rsplit("S", 1)[1]) if sid else 0})
    return pa.Table.from_pylist(rows, schema=FACTS_SCHEMA)


FACTS_SCHEMA = pa.schema([("id", pa.int64()), ("is_xpac", pa.bool_()), ("is_paratext", pa.bool_()),
                          ("source_type", pa.uint8()), ("source_is_core", pa.bool_()), ("source_in_doaj", pa.bool_()),
                          ("indexed_in", pa.uint8()), ("refs", pa.int16()), ("authors", pa.int16()),
                          ("locations", pa.int16()), ("citation_pct", pa.float32()), ("source", pa.int64())])


def save_raw(name: str, works: list[dict]) -> None:
    """raw/<name>.jsonl.gz and facts/<name>.parquet, each written whole then renamed."""
    with gzip.open(C.RAW / f"{name}.jsonl.gz.tmp", "wt") as g:
        for w in works:
            g.write(json.dumps(w) + "\n")
    os.replace(C.RAW / f"{name}.jsonl.gz.tmp", C.RAW / f"{name}.jsonl.gz")
    pq.write_table(facts(works), C.FACTS / f"{name}.parquet.tmp")
    os.replace(C.FACTS / f"{name}.parquet.tmp", C.FACTS / f"{name}.parquet")


# ---------------------------------------------------------------- cells

def window(today: dt.date) -> list[str]:
    """Dates to watch, in the order they are looked at: today back to the base, then ahead, then before the base."""
    base = dt.date.fromisoformat(C.BASE_SNAPSHOT_DATE)
    days = lambda a, b: [a + dt.timedelta(d) for d in range((b - a).days + 1)]   # noqa: E731
    new = days(base, today)[::-1]
    ahead = days(today + dt.timedelta(1), today + dt.timedelta(AHEAD_DAYS))
    before = days(base - dt.timedelta(LOOKBACK_DAYS), base - dt.timedelta(1))[::-1]
    return [d.isoformat() for d in new + ahead + before]


def field_cells(cl: Client, date: str, total: int) -> dict[str, int]:
    """(field id or 'none') -> works on that date. One group-by call."""
    g = cl.get(filter=f"{FILTER},publication_date:{date}", group_by="primary_topic.field.id")
    cells = {x["key"].rsplit("/", 1)[-1]: int(x["count"]) for x in g["group_by"] if x["key"] not in ("unknown", None)}
    none = total - sum(cells.values())
    if none > 0:
        cells["none"] = none
    return cells


def fetch_cell(cl: Client, date: str, field: str) -> tuple[int, list[dict]]:
    """Every work of one cell (cursor paging). Returns the API's count and the works."""
    f = f"{FILTER},publication_date:{date},primary_topic.field.id:{'null' if field == 'none' else field}"
    out, cursor, count = [], "*", None
    while cursor:
        page = cl.get(filter=f, per_page=100, cursor=cursor, select=SELECT)
        count = page["meta"]["count"] if count is None else count
        out += page["results"]
        cursor = page["meta"].get("next_cursor")
        if not page["results"]:
            break
    return int(count or 0), out


def refresh_records(cl: Client, run: str) -> None:
    """Works saved before the API record had every field the search needs (QUALITY_FIELDS): their record again,
    100 per call, saved as raw/<run>-refresh-<n>.jsonl.gz (the corpus step reads the newest record of a work).
    raw-checked.json lists the raw files already read, so each file is read once."""
    checked_path = C.UPDATES / "raw-checked.json"
    checked = json.loads(checked_path.read_text()) if checked_path.exists() else {"files": [], "stale": []}
    ok = set(checked["files"])
    stale = set(checked["stale"])
    for f in sorted(C.RAW.glob("*.jsonl.gz")):          # run names sort in time; a refresh after its run's files
        if f.name in ok:
            continue
        with gzip.open(f, "rt") as g:
            for line in g:
                w = json.loads(line)
                i = w["id"].rsplit("/", 1)[-1]
                (stale.discard if all(k in w for k in QUALITY_FIELDS) else stale.add)(i)
        ok.add(f.name)
    C.atomic_json(checked_path, {"files": sorted(ok), "stale": sorted(stale)})
    if not stale:
        return
    todo = sorted(stale)
    C.log(f"re-reading {len(todo):,} records saved without every field")
    for n, i in enumerate(range(0, len(todo), 5000)):
        batch = todo[i:i + 5000]
        if not cl.can_spend(PRICE * (len(batch) // 100 + 1)):
            C.log("  budget: the rest next run")
            return
        works = []
        for j in range(0, len(batch), 100):
            works += cl.get(filter="openalex_id:" + "|".join(batch[j:j + 100]), per_page=100, select=SELECT)["results"]
        name = f"{run}-refresh-{n:03d}"
        save_raw(name, works)
        stale -= set(batch)        # the ones not returned were deleted or merged: not asked for again
        ok.add(f"{name}.jsonl.gz")
        C.atomic_json(checked_path, {"files": sorted(ok), "stale": sorted(stale)})
        C.log(f"  {len(works):,} records -> raw/{name}.jsonl.gz")


# ---------------------------------------------------------------- run

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-usd", type=float, default=0.80, help="spend at most this much in this run")
    ap.add_argument("--reserve-usd", type=float, help="leave this much of today's budget (default: 0.10 with a key, "
                                                      "0.01 without), for other uses of the same key")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="count only: say what would be downloaded")
    ap.add_argument("--today", help="pretend today is this date (tests)")
    ap.add_argument("--only", help="look only at these publication dates, comma-separated (tests)")
    a = ap.parse_args()

    key = api_key()
    cl = Client(key, a.max_usd, a.reserve_usd if a.reserve_usd is not None else (0.10 if key else 0.01))
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for d in (C.CONTENT, C.RAW, C.FACTS, C.UPDATES / "runs"):
        d.mkdir(parents=True, exist_ok=True)
    state = json.loads(C.STATE.read_text()) if C.STATE.exists() else {"base": C.BASE_ID, "dates": {}, "cells": {}}
    if state.get("base") != C.BASE_ID:
        raise SystemExit(f"{C.STATE} belongs to base {state.get('base')}, not {C.BASE_ID}")
    schema = raw_schema()
    known = Known()
    today = dt.date.fromisoformat(a.today) if a.today else dt.datetime.now(dt.timezone.utc).date()
    C.log(f"base {C.BASE_ID}: {len(known.base):,} works, {len(known.extra):,} fetched before; "
          f"{'API key' if key else 'no API key ($0.10/day)'}; run {run}")

    buffer: dict[int, dict] = {}       # new works not written yet
    pending: list[tuple[str, int]] = []   # cells whose works are all in the buffer: (cell key, API count)
    pending_dates: dict[str, int] = {}
    report = {"run": run, "dates_checked": 0, "dates_changed": 0, "cells_fetched": 0, "works_seen": 0,
              "new_works": 0, "written": 0, "stopped": None}
    nflush = 0

    def flush(final: bool = False) -> None:
        nonlocal nflush
        if buffer:
            works = list(buffer.values())
            name = f"{run}-{nflush:03d}"
            content = to_content(works, schema)
            save_raw(name, works)
            content.write_parquet(C.CONTENT / f"{name}.parquet.tmp", compression="zstd")
            os.replace(C.CONTENT / f"{name}.parquet.tmp", C.CONTENT / f"{name}.parquet")
            report["written"] += content.height
            known.extra.update(buffer)
            C.log(f"  wrote {content.height:,} new works ({len(works) - content.height} dropped by the selection) "
                  f"-> content/{name}.parquet")
            nflush += 1
            buffer.clear()
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        for k, n in pending:
            state["cells"][k] = {"count": n, "fetched_utc": now}
        for d, n in pending_dates.items():
            state["dates"][d] = {"total": n, "checked_utc": now}
        pending.clear(); pending_dates.clear()
        C.atomic_json(C.STATE, state)

    pool = ThreadPoolExecutor(a.threads)
    if not a.dry_run:
        refresh_records(cl, run)
    try:
        dates = window(today) if not a.only else a.only.split(",")
        for date in dates:
            if not cl.can_spend(PRICE * 3):
                report["stopped"] = "budget"
                break
            total = int(cl.get(filter=f"{FILTER},publication_date:{date}", per_page=1, select="id")["meta"]["count"])
            report["dates_checked"] += 1
            if state["dates"].get(date, {}).get("total") == total:
                continue
            report["dates_changed"] += 1
            cells = field_cells(cl, date, total) if total else {}
            todo = [(f, n) for f, n in cells.items() if state["cells"].get(f"{date}|{f}", {}).get("count") != n]
            cost = sum(PRICE * (1 + n // 100) for _, n in todo)
            if a.dry_run:
                C.log(f"{date}: {total:,} works; {len(todo)} cells changed, {sum(n for _, n in todo):,} works to look at"
                      f" (${cost:.4f})")
                continue
            if not cl.can_spend(cost):
                report["stopped"] = f"budget (next: {date}, ${cost:.4f})"
                break
            seen_new = 0
            for (fld, _), (count, works) in zip(todo, pool.map(lambda c: fetch_cell(cl, date, c[0]), todo)):
                report["cells_fetched"] += 1
                report["works_seen"] += len(works)
                for w in works:
                    n = int(w["id"].rsplit("W", 1)[1])
                    if n not in known and n not in buffer:
                        buffer[n] = w
                        seen_new += 1
                        report["new_works"] += 1
                pending.append((f"{date}|{fld}", count))
            pending_dates[date] = total
            C.log(f"{date}: {total:,} works, {len(todo)} cells fetched, {seen_new:,} new; "
                  f"${cl.spent:.4f} spent, {'?' if cl.left is None else f'${cl.left:.3f}'} left today")
            if len(buffer) >= FLUSH_WORKS:
                flush()
    except BudgetSpent as e:
        report["stopped"] = str(e)
    finally:
        pool.shutdown(wait=True)
        if not a.dry_run:
            flush(final=True)
    report.update(spent_usd=round(cl.spent, 4), calls=cl.calls, left_usd=cl.left,
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    if not a.dry_run:
        C.atomic_json(C.UPDATES / "runs" / f"{run}.json", report)
    C.log(json.dumps(report))


if __name__ == "__main__":
    main()
