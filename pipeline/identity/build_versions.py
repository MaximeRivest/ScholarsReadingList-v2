"""Paper versions: OpenAlex records that are one paper (a preprint and its journal article, a mirror copy of an
article, a conference paper reprinted as a chapter), grouped, with one canonical record per group.

    nice -n 10 search/.venv/bin/python pipeline/identity/build_versions.py [--out DIR] [--threads 12]

Reads the flat content snapshot (/mnt/bulk/openalex/content/year=*/data.parquet, all 476M records). Two records
are versions of one paper when, among publication types (citewalk/science/scope.py):
  * their titles are equal after folding (letters and digits only, accents and case dropped; at least 25
    characters and 4 words: names.title_key),
  * their publication years are at most 2 apart, and
  * they share a surname among their first 10 authors (names.sql_surname_tokens; both orders of a name, so
    'Wei Lei' meets 'Lei Wei'). Records without a personal author are never grouped (organisations and placeholders like 'None
Available' do not count: names.is_person_name). A group holding two different DOIs of one publisher and one type (a
book linked to each of its chapters) is a chain, not one paper, and is left ungrouped.
Groups are the connected components of those pairs. Titles shared by more than MAX_SAME_TITLE records are
generic ('Book reviews', 'Letter to the editor ...') and skipped.

Canonical record of a group, best first: not a mirror (DOI prefix 10.60692, 'Greater South Information System'),
a publisher DOI, published type (article/review > chapter/conference/book/letter/editorial > report/thesis/abstract > preprint),
a publisher DOI (> no DOI > repository DOI: arXiv, Zenodo, Research Square, bioRxiv/medRxiv, SSRN, ...), has an
abstract, most cited, smallest W-number.

Writes DIR (default /mnt/fast/scholarsreadinglist/identity/<snapshot>/versions/):
  members.parquet   work, cluster, rank (0 = canonical), type, year, doi, venue, has_abstract, language,
                    citations: every record in a group of 2+
  work.npy uint64 (sorted) / cluster.npy uint32 / rank.npy uint16   W-number -> group, place in it (0 = canonical)
  c_off.npy uint64, c_work.npy uint64               group -> its records, canonical first
  meta.json         counts, rules, checks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "citewalk" / "science"))
import names as N  # noqa: E402
from scope import SCHOLARLY_TYPES  # noqa: E402

CONTENT = Path("/mnt/bulk/openalex/content")
SNAPSHOT = "openalex-2026-09-23"
MAX_SAME_TITLE = 200
YEAR_GAP = 2
MIRROR_PREFIXES = ("10.60692",)
REPOSITORY_PREFIXES = ("10.48550", "10.5281", "10.6084", "10.21203", "10.1101", "10.20944", "10.2139", "10.26434",
                       "10.31219", "10.31234", "10.31235", "10.31222", "10.31223", "10.31224", "10.31226", "10.31730",
                       "10.32942", "10.36227", "10.22541", "10.55458", "10.17605", "10.5167", "10.17863", "10.18154",
                       "10.13140", "10.17615", "10.7910", "10.6092", "10.25384")
TYPE_RANK = {"article": 0, "review": 0, "book-chapter": 1, "conference-paper": 1, "book": 1, "letter": 1,
             "editorial": 1, "data-paper": 1, "software-paper": 1, "report": 2, "dissertation": 2,
             "conference-abstract": 2, "preprint": 3}
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.time() - T0:7.0f}s {msg}", flush=True)


def files(years: str | None = None) -> list[str]:
    pat = f"year={years}/data.parquet" if years else "year=*/data.parquet"
    return [str(f) for f in sorted(CONTENT.glob(pat)) if pq.ParquetFile(f).metadata.num_rows]


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of n nodes and edges (a, b): label = smallest node of its component."""
    lab = np.arange(n, dtype=np.int64)
    while True:                                  # label propagation with pointer jumping; groups are tiny
        m = np.minimum(lab[a], lab[b])
        changed = (lab[a] != m) | (lab[b] != m)
        np.minimum.at(lab, a, m)
        np.minimum.at(lab, b, m)
        lab = lab[lab]
        if not changed.any():
            return lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(f"/mnt/fast/scholarsreadinglist/identity/{SNAPSHOT}/versions"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--memory", default="48GB")
    ap.add_argument("--years", help="glob over year= directories, for a trial run (e.g. '202[01]')")
    a = ap.parse_args()
    if (a.out / "meta.json").exists():
        raise SystemExit(f"{a.out} is finished")
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.memory}'; SET preserve_insertion_order=false; "
                f"SET temp_directory='{tmp}'")
    fl = files(a.years)
    types = sorted(SCHOLARLY_TYPES)
    key = N.sql_title_key("title")
    src = f"read_parquet({fl!r})"
    where = f"type IN {tuple(types)!r}"

    keys = tmp / "dup_keys.parquet"
    if not keys.exists():
        con.execute(f"""COPY (SELECT hash({key}) AS h, count(*) AS n FROM {src} WHERE {where} AND {key} IS NOT NULL
                     GROUP BY 1 HAVING count(*) > 1) TO '{keys}' (FORMAT PARQUET)""")
    nk = con.execute(f"SELECT count(*), sum(n), sum((n > {MAX_SAME_TITLE})::int), sum(CASE WHEN n > {MAX_SAME_TITLE} THEN n END) FROM '{keys}'").fetchone()
    log(f"titles held by 2+ records: {nk[0]:,} ({nk[1]:,} records); generic (> {MAX_SAME_TITLE}): {nk[2]:,} ({nk[3] or 0:,} records)")

    cand = tmp / "candidates.parquet"
    if not cand.exists():
        con.execute(f"CREATE TEMP TABLE k AS SELECT h FROM '{keys}' WHERE n <= {MAX_SAME_TITLE}")
        toks = N.sql_surname_tokens("string_split(authors, ' | ')")
        con.execute(f"""COPY (
            SELECT CAST(substr(id, 23) AS UBIGINT) AS work, hash({key}) AS h, year, type,
                   lower(regexp_replace(doi, '^https://doi.org/', '')) AS doi, venue, language, coalesce(citations, 0) AS citations,
                   (abstract IS NOT NULL AND length(trim(abstract)) >= 100) AS has_abstract,
                   CASE WHEN authors IS NULL OR trim(authors) = '' THEN [] ELSE {toks} END AS toks
            FROM {src} WHERE {where} AND {key} IS NOT NULL AND hash({key}) IN (SELECT h FROM k)
            ) TO '{cand}' (FORMAT PARQUET)""")
    nc = con.execute(f"SELECT count(*), sum((len(toks) = 0)::int) FROM '{cand}'").fetchone()
    log(f"candidate records {nc[0]:,}; without authors (never grouped) {nc[1]:,}")

    pairs = tmp / "pairs.parquet"
    if not pairs.exists():
        con.execute(f"CREATE TEMP TABLE c AS SELECT row_number() OVER () - 1 AS i, * FROM '{cand}'")
        con.execute(f"""COPY (SELECT x.i AS a, y.i AS b FROM c x JOIN c y ON x.h = y.h AND x.i < y.i
                     WHERE abs(coalesce(x.year, -9999) - coalesce(y.year, -9999)) <= {YEAR_GAP}
                       AND len(x.toks) > 0 AND len(y.toks) > 0 AND len(list_intersect(x.toks, y.toks)) > 0
                       -- two DOIs of one publisher name two works (letters of an archive, chapters of a book, entries
                       -- of a database), except a repository's versioned uploads, a preprint and its article at one
                       -- publisher (Copernicus), or a DOI that extends the other (a truncated deposit)
                       AND NOT (x.doi IS NOT NULL AND y.doi IS NOT NULL AND x.doi <> y.doi
                                AND split_part(x.doi, '/', 1) = split_part(y.doi, '/', 1)
                                AND split_part(x.doi, '/', 1) NOT IN {REPOSITORY_PREFIXES + MIRROR_PREFIXES!r}
                                AND x.type = y.type
                                AND NOT starts_with(x.doi, y.doi) AND NOT starts_with(y.doi, x.doi))
                     ) TO '{pairs}' (FORMAT PARQUET)""")
        con.execute(f"COPY (SELECT * EXCLUDE (toks) FROM c ORDER BY i) TO '{tmp / 'cand_rows.parquet'}' (FORMAT PARQUET)")
    p = pq.read_table(pairs)
    pa_, pb_ = p["a"].to_numpy().astype(np.int64), p["b"].to_numpy().astype(np.int64)
    rows = pq.read_table(tmp / "cand_rows.parquet")
    n = rows.num_rows
    log(f"pairs {len(pa_):,} over {n:,} candidates")
    lab = components(n, pa_, pb_)
    grouped = np.zeros(n, bool)
    grouped[pa_] = True
    grouped[pb_] = True

    work = rows["work"].to_numpy()
    typ = rows["type"].to_pylist()
    doi = rows["doi"].to_pylist()
    venue = rows["venue"].to_pylist()
    has_abs = rows["has_abstract"].to_numpy(zero_copy_only=False)
    cites = rows["citations"].to_numpy()
    import re
    valid = re.compile(r"^10\.\d{4,9}/[^\s\"<>]+$")
    pre = [(d or "").split("/", 1)[0] if d and valid.match(d) else "" for d in doi]   # a malformed DOI counts as none
    mirror = np.array([p_ in MIRROR_PREFIXES or v == "Greater South Information System" for p_, v in zip(pre, venue)])
    trank = np.array([TYPE_RANK.get(t, 2) for t in typ], np.int8)
    srank = np.array([2 if p_ in REPOSITORY_PREFIXES else (1 if not p_ else 0) for p_ in pre], np.int8)
    # A record can link to several that may not link to each other (a book to each of its chapters, all titled like
    # the book): such chains hold two different DOIs of one publisher and one type. Those groups are left ungrouped.
    pre_all = np.array([(d or "").split("/", 1)[0] for d in rows["doi"].to_pylist()], object)
    gi = np.flatnonzero(grouped)
    import pyarrow as pa
    gtab = pa.table({"g": lab[gi], "p": pre_all[gi].astype(str), "t": pa.array(rows["type"].to_pylist()).take(gi),
                     "d": rows["doi"].take(gi)})
    # chains: two DOIs of one publisher and type where neither extends the other ('X' and 'X.v1' or 'X/pdf' are one)
    bad = con.execute(f"""WITH x AS (SELECT g, p, t, d FROM gtab WHERE p <> '' AND p NOT IN {REPOSITORY_PREFIXES + MIRROR_PREFIXES!r}),
                         r AS (SELECT g, p, t, min_by(d, length(d)) AS root, count(DISTINCT d) AS nd FROM x GROUP BY 1, 2, 3)
                         SELECT DISTINCT x.g FROM x JOIN r USING (g, p, t) WHERE r.nd > 1 AND NOT starts_with(x.d, r.root)
                      """).fetchnumpy()["g"].astype(np.int64)
    conflicted = np.isin(lab, bad) & grouped
    pq.write_table(rows.filter(pa.array(conflicted)).select(["work", "type", "year", "doi", "venue"])
                   .append_column("chain", pa.array(lab[conflicted])), a.out / "chained.parquet", compression="zstd")
    grouped &= ~conflicted
    log(f"chained groups left ungrouped: {len(bad):,} ({int(conflicted.sum()):,} records)")
    idx = np.flatnonzero(grouped)
    # sort by group, then canonical order
    o = idx[np.lexsort((work[idx], -cites[idx], ~has_abs[idx], srank[idx], trank[idx], srank[idx] > 0, mirror[idx], lab[idx]))]
    g = lab[o]
    start = np.r_[True, g[1:] != g[:-1]]
    gid = np.cumsum(start) - 1
    rank = np.arange(len(o)) - np.flatnonzero(start)[gid]
    G = int(gid[-1]) + 1
    c_off = np.r_[np.flatnonzero(start), len(o)].astype(np.uint64)
    c_work = work[o].astype(np.uint64)
    ow = np.argsort(c_work, kind="stable")
    if np.any(c_work[ow][1:] == c_work[ow][:-1]):
        raise SystemExit("a work in two candidate rows")
    np.save(a.out / "work.npy", c_work[ow])
    np.save(a.out / "cluster.npy", gid[ow].astype(np.uint32))
    np.save(a.out / "rank.npy", rank[ow].astype(np.uint16))
    np.save(a.out / "c_off.npy", c_off)
    np.save(a.out / "c_work.npy", c_work)
    sel = rows.take(o)
    import pyarrow as pa
    sel = sel.append_column("cluster", pa.array(gid.astype(np.uint32))).append_column("rank", pa.array(rank.astype(np.uint16)))
    pq.write_table(sel.drop(["i", "h"]), a.out / "members.parquet", compression="zstd")

    sizes = np.diff(c_off.astype(np.int64))
    canon = o[start]
    m = {"what": "groups of OpenAlex records that are one paper; canonical first", "snapshot": SNAPSHOT,
         "rules": {"title": "names.title_key (>= 25 folded characters, >= 4 words)", "year_gap": YEAR_GAP,
                   "authors": "share a surname token among the first 10 authors", "max_same_title": MAX_SAME_TITLE,
                   "types": types, "mirror_prefixes": MIRROR_PREFIXES, "repository_prefixes": REPOSITORY_PREFIXES,
                   "type_rank": TYPE_RANK},
         "titles_shared": int(nk[0]), "generic_titles_skipped": int(nk[2]), "candidates": int(n),
         "candidates_without_authors": int(nc[1]), "pairs": int(len(pa_)),
         "chained_groups_dropped": int(len(bad)), "chained_records": int(conflicted.sum()),
         "groups": G, "records_in_groups": int(len(o)), "surplus_records": int(len(o) - G),
         "group_sizes": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(sizes, 10), return_counts=True))},
         "groups_with_mirror": int(np.add.reduceat(mirror[o].astype(np.int64), c_off[:-1].astype(np.int64)).astype(bool).sum()),
         "groups_with_preprint": int(np.add.reduceat((trank[o] == 3).astype(np.int64), c_off[:-1].astype(np.int64)).astype(bool).sum()),
         "canonical_without_abstract_but_a_version_has_one": int(
             (~has_abs[canon] & np.add.reduceat(has_abs[o].astype(np.int64), c_off[:-1].astype(np.int64)).astype(bool)).sum()),
         "seconds": round(time.time() - T0), "files": len(fl)}
    (a.out / "meta.json.tmp").write_text(json.dumps(m, indent=1))
    os.replace(a.out / "meta.json.tmp", a.out / "meta.json")
    log(json.dumps({k: v for k, v in m.items() if k != "rules"}))


if __name__ == "__main__":
    main()
