# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.30"]
# ///
"""OpenAlex works snapshot -> tidy content table, partitioned by publication year.

    uv run pipeline/openalex_content.py            # process new/changed files, rebuild touched years
    uv run pipeline/openalex_content.py --check    # compare against the slow reference on a sample
    uv run pipeline/openalex_content.py --dry-run  # say what would be done

Two stages, both incremental:

1. flatten  works/updated_date=*/part_*.parquet  ->  staging/year=Y/<partition>__<part>.parquet
   One job per source file, skipped when its (size, mtime) is unchanged.
2. publish  staging/year=Y/*  ->  content/year=Y/data.parquet
   OpenAlex puts a work in a new updated_date folder each time it changes, so
   the same work can appear several times. Only its newest version is kept,
   across all years (a work whose year changed leaves the old year).
   Only years touched by stage 1 are rewritten.

Columns: one row per work, one plain value per cell. Lists are joined with
" | "; a "|" inside an item becomes "¦", so split(" | ") recovers the items.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("OPENALEX_ROOT", "/mnt/bulk/openalex"))
SOURCE = ROOT / "works"
STAGING = ROOT / "_staging" / "content"
CONTENT = ROOT / "content"
STATE = ROOT / "_staging" / "content_state.json"

SEP = " | "
WORKERS = 12          # processes
THREADS = 4           # polars threads per process

COLUMNS = [
    "id", "doi", "url", "pdf_url",
    "title", "type", "language", "date", "year",
    "authors", "author_orcids", "corresponding_authors", "institutions", "countries",
    "venue", "publisher", "issn", "volume", "issue", "pages",
    "abstract", "keywords", "topic", "subfield", "field", "domain", "mesh_terms",
    "is_open_access", "open_access_status", "license",
    "funders", "grants",
    "citations", "references_count", "references",
    "is_retracted", "updated_date",
]

def with_flags(lf):
    """Add is_complete: English, has an abstract and has references (a 'good' record).
    Added at publish time, so changing the rule never needs a re-flatten."""
    import polars as pl
    return lf.with_columns(is_complete=(
        (pl.col("language") == "en").fill_null(False)
        & pl.col("abstract").is_not_null()
        & pl.col("references").is_not_null()))


# ── stage 1: flatten one file ────────────────────────────────────────────────

def _joined(items):
    """List[str] expr -> one string: items cleaned, empties dropped, None when nothing is left."""
    import polars as pl
    clean = items.list.eval(
        pl.element().str.replace_all("|", "¦", literal=True).str.strip_chars()
        .filter(pl.element().is_not_null() & (pl.element() != "")))
    out = clean.list.join(SEP)
    return pl.when(out != "").then(out)


def _abstracts(lf):
    """(row, abstract_inverted_index) -> (row, abstract): words put back in position order."""
    import polars as pl
    # {"word": [0, 7], "other": [1]}  ->  one (word, positions) pair per match
    pair = r'"((?:[^"\\]|\\.)*)":\s*\[([0-9,\s]*)\]'
    return (
        lf.select("row", pairs=pl.col("abstract_inverted_index").str.extract_all(pair))
        .explode("pairs", empty_as_null=True)
        .drop_nulls("pairs")
        .select(
            "row",
            # the key is a JSON string: decode it so \" \u00e9 \n come out right
            word=pl.concat_str(pl.lit('"'), pl.col("pairs").str.extract(pair, 1), pl.lit('"'))
                 .str.json_decode(pl.String),
            pos=pl.col("pairs").str.extract(pair, 2).str.split(",")
                .list.eval(pl.element().str.strip_chars().cast(pl.Int64, strict=False)),
        )
        .explode("pos", empty_as_null=True)
        .drop_nulls("pos")
        .sort("row", "pos")
        .group_by("row", maintain_order=True)
        .agg(abstract=pl.col("word").str.join(" "))
        .with_columns(abstract=pl.when(pl.col("abstract") != "").then("abstract"))
    )


def flatten(raw):
    """Raw works LazyFrame -> content LazyFrame (same rows, flat columns)."""
    import polars as pl
    a = pl.col("authorships")
    loc = pl.col("primary_location")
    src = loc.struct.field("source")
    oa = pl.col("best_oa_location")
    bib = pl.col("biblio")
    topic = pl.col("primary_topic")
    first, last = bib.struct.field("first_page"), bib.struct.field("last_page")
    el = pl.element()

    raw = raw.with_row_index("row")
    flat = raw.select(
        "row",
        "id", "doi",
        url=loc.struct.field("landing_page_url"),
        pdf_url=pl.coalesce(oa.struct.field("pdf_url"), loc.struct.field("pdf_url")),
        title=pl.col("title"),
        type=pl.col("type"),
        language=pl.col("language"),
        date=pl.col("publication_date"),
        year=pl.col("publication_year").cast(pl.Int64),
        authors=_joined(a.list.eval(pl.coalesce(
            el.struct.field("author").struct.field("display_name"), el.struct.field("raw_author_name")))),
        author_orcids=_joined(a.list.eval(pl.coalesce(
            el.struct.field("author").struct.field("orcid"), pl.lit("-")))),
        corresponding_authors=_joined(a.list.eval(
            el.struct.field("author").struct.field("display_name").filter(el.struct.field("is_corresponding")))),
        institutions=_joined(a.list.eval(
            el.struct.field("institutions").list.explode(keep_nulls=False, empty_as_null=False).struct.field("display_name").unique(maintain_order=True))),
        countries=_joined(a.list.eval(
            el.struct.field("countries").list.explode(keep_nulls=False, empty_as_null=False).unique(maintain_order=True))),
        venue=pl.coalesce(src.struct.field("display_name"), loc.struct.field("raw_source_name")),
        publisher=src.struct.field("host_organization_name"),
        issn=src.struct.field("issn_l"),
        volume=bib.struct.field("volume"),
        issue=bib.struct.field("issue"),
        pages=pl.when(first.is_not_null() & (first != "") & last.is_not_null() & (last != "") & (first != last))
                .then(pl.concat_str(first, pl.lit("–"), last))
                .otherwise(pl.when(first != "").then(first)),
        keywords=_joined(pl.col("keywords").list.eval(el.struct.field("display_name"))),
        topic=topic.struct.field("display_name"),
        subfield=topic.struct.field("subfield").struct.field("display_name"),
        field=topic.struct.field("field").struct.field("display_name"),
        domain=topic.struct.field("domain").struct.field("display_name"),
        mesh_terms=_joined(pl.col("mesh").list.eval(el.struct.field("descriptor_name").unique(maintain_order=True))),
        is_open_access=pl.col("open_access").struct.field("is_oa"),
        open_access_status=pl.col("open_access").struct.field("oa_status"),
        license=pl.coalesce(oa.struct.field("license"), loc.struct.field("license")),
        funders=_joined(pl.col("funders").list.eval(el.struct.field("display_name"))),
        grants=_joined(pl.col("awards").list.eval(
            pl.concat_str(el.struct.field("funder_display_name").fill_null("?"), pl.lit(": "),
                          el.struct.field("funder_award_id"))
            .filter(el.struct.field("funder_award_id").is_not_null()))),
        citations=pl.col("cited_by_count").cast(pl.Int64),
        references_count=pl.col("referenced_works_count").cast(pl.Int64),
        references=_joined(pl.col("referenced_works")),
        is_retracted=pl.col("is_retracted"),
        updated_date=pl.col("updated_date"),
    )
    abstracts = _abstracts(raw.select("row", "abstract_inverted_index"))
    return flat.join(abstracts, on="row", how="left").sort("row").select(COLUMNS)


def _staging_name(src: Path) -> str:
    return f"{src.parent.name.replace('updated_date=', '')}__{src.stem}.parquet"


def flatten_file(src: str) -> tuple[str, list[int], int, float]:
    """Stage 1 for one source file. Returns (src, years written, rows, seconds)."""
    import polars as pl
    t = time.time()
    src_p = Path(src)
    name = _staging_name(src_p)
    df = flatten(pl.scan_parquet(src_p)).collect()
    years = []
    for (year,), part in df.group_by("year"):
        y = -1 if year is None else int(year)
        out = STAGING / f"year={y}" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(out.with_suffix(".tmp"), compression="zstd")
        os.replace(out.with_suffix(".tmp"), out)
        years.append(y)
    return src, years, df.height, time.time() - t


# ── stage 2: publish touched years, newest version of each work only ─────────

def publish(years: set[int]) -> None:
    import polars as pl
    all_staged = str(STAGING / "*" / "*.parquet")
    t = time.time()
    # the newest version of every work, over every year: (id, updated_date, year).
    # Split by year once, so each year then checks against its own works only.
    # IDs as numbers ("https://openalex.org/W123" -> 123): ~4x less memory than strings
    num = pl.col("id").str.slice(22).cast(pl.Int64)
    latest = (pl.scan_parquet(all_staged, hive_partitioning=False)
              .select(idn=num, updated_date="updated_date", year=pl.col("year").fill_null(-1))
              .group_by("idn")
              .agg(pl.col("updated_date").max(),
                   pl.col("year").get(pl.col("updated_date").arg_max()))
              .collect(engine="streaming"))
    by_year = latest.partition_by("year", as_dict=True, include_key=False)
    print(f"  newest-version map: {latest.height:,} distinct works ({time.time() - t:.0f}s)", flush=True)
    del latest
    for y in sorted(years):
        t = time.time()
        files = sorted((STAGING / f"year={y}").glob("*.parquet"))
        keep = by_year.get((y,))
        out_dir = CONTENT / f"year={y}"
        if not files or keep is None:
            shutil.rmtree(out_dir, ignore_errors=True)
            continue
        df = (pl.scan_parquet(files, hive_partitioning=False)
              .with_columns(idn=num)
              .join(keep.lazy(), on=["idn", "updated_date"], how="semi")
              .drop("idn")
              .pipe(with_flags)
              .unique("id", keep="any")
              .sort("date", "id", nulls_last=True)
              .collect())
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "data.parquet.tmp"
        df.write_parquet(tmp, compression="zstd", row_group_size=100_000)
        os.replace(tmp, out_dir / "data.parquet")
        if df.height > 1_000_000 or time.time() - t > 30:
            print(f"  year {y}: {df.height:,} works ({time.time() - t:.0f}s)", flush=True)
    print(f"  published {len(years)} years", flush=True)


# ── driver ───────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"files": {}}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=0))
    os.replace(tmp, STATE)


def _init_worker():
    os.environ["POLARS_MAX_THREADS"] = str(THREADS)


def run(dry_run: bool = False, workers: int = WORKERS) -> None:
    state = _load_state()
    sources = sorted(SOURCE.glob("updated_date=*/*.parquet"))
    sig = {str(p): [p.stat().st_size, int(p.stat().st_mtime)] for p in sources}
    todo = [p for p in sig if state["files"].get(p, {}).get("sig") != sig[p]]
    gone = [p for p in state["files"] if p not in sig]
    print(f"{len(sources)} source files · {len(todo)} new or changed · {len(gone)} removed", flush=True)
    if dry_run:
        print(f"{len(state.get('pending_years', []))} years waiting to be published", flush=True)
        return

    touched: set[int] = set()
    for p in gone:     # a deleted source file: drop its staged pieces
        for y in state["files"][p]["years"]:
            (STAGING / f"year={y}" / _staging_name(Path(p))).unlink(missing_ok=True)
            touched.add(y)
        del state["files"][p]

    t0, rows = time.time(), 0
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        jobs = {pool.submit(flatten_file, p): p for p in todo}
        for i, job in enumerate(as_completed(jobs), 1):
            src, years, n, secs = job.result()
            for y in state["files"].get(src, {}).get("years", []):   # changed file: old pieces may be stale
                if y not in years:
                    (STAGING / f"year={y}" / _staging_name(Path(src))).unlink(missing_ok=True)
                    touched.add(y)
            state["files"][src] = {"sig": sig[src], "years": years, "rows": n}
            touched.update(years)
            rows += n
            if i % 10 == 0 or i == len(todo):
                _save_state(state)
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {rows:,} rows · {el / 60:.1f} min · "
                      f"~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)

    touched |= set(state.get("pending_years", []))
    state["pending_years"] = sorted(touched)
    _save_state(state)
    if touched:
        print(f"publishing {len(touched)} years", flush=True)
        publish(touched)
    state["pending_years"] = []
    _save_state(state)
    print(f"done in {(time.time() - t0) / 60:.1f} min → {CONTENT}", flush=True)


def check(n: int = 20_000) -> None:
    """Compare against a plain-Python reference on the newest file."""
    import polars as pl
    src = sorted(SOURCE.glob("updated_date=*/*.parquet"))[-1]
    raw = pl.scan_parquet(src).head(n).collect()
    t = time.time()
    fast = flatten(raw.lazy()).collect()
    print(f"polars: {n:,} rows in {time.time() - t:.2f}s")
    ref = _reference(raw)
    bad = 0
    for c in COLUMNS:
        a, b = fast[c].to_list(), ref[c]
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if diff:
            bad += 1
            i = diff[0]
            print(f"  {c}: {len(diff)} differ, e.g. row {i}: {str(a[i])[:80]!r} vs {str(b[i])[:80]!r}")
    print("all columns match" if not bad else f"{bad} columns differ")
    sys.exit(1 if bad else 0)


def _reference(raw) -> dict:
    """The same table, written the obvious slow way, to check the fast one."""
    def j(items):
        out = [s for s in (str(x).replace("|", "¦").strip() for x in items or [] if x is not None) if s]
        return SEP.join(out) if out else None
    def g(d, *path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    cols = {c: [] for c in COLUMNS}
    for w in raw.iter_rows(named=True):
        A = w["authorships"] or []
        loc, oa, bib, tp = w["primary_location"] or {}, w["best_oa_location"] or {}, w["biblio"] or {}, w["primary_topic"]
        f, l = bib.get("first_page"), bib.get("last_page")
        ab = None
        if w["abstract_inverted_index"]:
            # object_pairs_hook keeps repeated keys: OpenAlex sometimes lists a word twice,
            # and plain json.loads would silently drop the earlier positions
            idx = json.loads(w["abstract_inverted_index"], object_pairs_hook=list) or []
            ab = " ".join(x for _, x in sorted(((p, x) for x, ps in idx for p in ps), key=lambda t: t[0])) or None
        row = dict(
            id=w["id"], doi=w["doi"], url=loc.get("landing_page_url"),
            pdf_url=oa.get("pdf_url") or loc.get("pdf_url"),
            title=w["title"], type=w["type"], language=w["language"], date=w["publication_date"],
            year=w["publication_year"],
            authors=j(g(x, "author", "display_name") or x.get("raw_author_name") for x in A),
            author_orcids=j(g(x, "author", "orcid") or "-" for x in A),
            corresponding_authors=j(g(x, "author", "display_name") for x in A if x.get("is_corresponding")),
            institutions=j(dict.fromkeys(i.get("display_name") for x in A for i in x.get("institutions") or [])),
            countries=j(dict.fromkeys(c for x in A for c in x.get("countries") or [])),
            venue=g(loc, "source", "display_name") or loc.get("raw_source_name"),
            publisher=g(loc, "source", "host_organization_name"), issn=g(loc, "source", "issn_l"),
            volume=bib.get("volume"), issue=bib.get("issue"),
            pages=f"{f}–{l}" if f and l and f != l else (f or None),
            abstract=ab,
            keywords=j(k.get("display_name") for k in w["keywords"] or []),
            topic=g(tp, "display_name"), subfield=g(tp, "subfield", "display_name"),
            field=g(tp, "field", "display_name"), domain=g(tp, "domain", "display_name"),
            mesh_terms=j(dict.fromkeys(m.get("descriptor_name") for m in w["mesh"] or [])),
            is_open_access=g(w["open_access"], "is_oa"), open_access_status=g(w["open_access"], "oa_status"),
            license=oa.get("license") or loc.get("license"),
            funders=j(x.get("display_name") for x in w["funders"] or []),
            grants=j(f"{x.get('funder_display_name') or '?'}: {x['funder_award_id']}"
                     for x in w["awards"] or [] if x.get("funder_award_id")),
            citations=w["cited_by_count"], references_count=w["referenced_works_count"],
            references=j(w["referenced_works"]), is_retracted=w["is_retracted"], updated_date=w["updated_date"],
        )
        for c in COLUMNS:
            cols[c].append(row[c])
    return cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the slow reference and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    check() if args.check else run(args.dry_run, args.workers)
