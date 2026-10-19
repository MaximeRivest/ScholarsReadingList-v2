"""Abstract rescue: OpenAlex publications that have no abstract in the snapshot, given one from a trusted source, as a
small corpus beside the base one (same layout as the daily updates corpus, so the release and the API read it alike).

    nice -n 10 search/.venv/bin/python pipeline/rescue/build.py [--out DIR] [--sources pubmed,versions]

Sources, in order of preference for a work:
  pubmed    PubMed's own record of the paper (pipeline/pubmed: baseline + updates), matched by the PMID OpenAlex
            holds for the work, else by DOI; English records only (PubMed language 'eng'), abstract >= 100 characters.
  versions  another record of the same paper (pipeline/identity/build_versions.py: a mirror copy, a repository copy, a
            preprint) that has an English abstract; given to the group's canonical record when it has none.
Only publication types (citewalk/science/scope.py), OpenAlex language 'en' or unknown, works the base corpus and the
updates corpus do not hold.

Writes DIR (default /mnt/fast/scholarsreadinglist/rescue/<snapshot>/<run>/):
  content/part-*.parquet   rows in the content snapshot's columns (search/build_corpus.py COLUMNS), abstract filled
  sources.parquet          id, abstract_source ('pubmed:<pmid>' | 'version:W<n>'), abstract_copyright (PubMed's)
  corpus/                  parts, works.sqlite, keywords, map-meta (as pipeline/updates/corpus.py builds them)
  manifest.json, DONE.json
"""
from __future__ import annotations

import argparse
import re
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "search"))
sys.path.insert(0, str(REPO / "citewalk" / "science"))
import build_corpus as bc  # noqa: E402
from scope import SCHOLARLY_TYPES  # noqa: E402

SNAPSHOT = "openalex-2026-09-23"
CONTENT = Path("/mnt/bulk/openalex/content")
TIDY = Path("/mnt/bulk/openalex/tidy/generations/20261007T003120Z-50ce3865/manifest.json")
PUBMED = Path("/mnt/bulk/pubmed/pubmed.parquet")
VERSIONS = Path(f"/mnt/fast/scholarsreadinglist/identity/{SNAPSHOT}/versions")
BASE_CORPUS = Path("/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05")
MIN_ABSTRACT = 100
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.time() - T0:7.0f}s {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sources", default="pubmed,versions")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--years", help="glob over year= directories (trial)")
    a = ap.parse_args()
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = a.out or Path(f"/mnt/fast/scholarsreadinglist/rescue/{SNAPSHOT}/{run}")
    if (out / "DONE.json").exists():
        raise SystemExit(f"{out} is finished")
    (out / "content").mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    srcs = set(a.sources.split(","))
    # the working database is on disk: a stopped run resumes after its last finished table
    files = [str(f) for f in sorted(CONTENT.glob(f"year={a.years or '*'}/data.parquet")) if pq.ParquetFile(f).metadata.num_rows]
    if (out / "sources.parquet").exists() and any((out / "content").glob("*.parquet")):   # the selection is done
        return make_corpus(out, files)
    con = duckdb.connect(str(tmp / "work.duckdb"))
    con.execute(f"SET threads={a.threads}; SET memory_limit='32GB'; SET preserve_insertion_order=false; SET temp_directory='{tmp}'")
    have = lambda t: con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [t]).fetchone()[0] > 0  # noqa: E731
    _ex = con.execute

    def run(sql, *args):
        m_ = re.match(r"\s*CREATE TABLE (\w+)", sql)
        if m_ and have(m_.group(1)):
            return con
        if re.match(r"\s*INSERT INTO want", sql) and have("rows"):
            return con
        return _ex(sql, *args)
    types = tuple(sorted(SCHOLARLY_TYPES))
    files = [str(f) for f in sorted(CONTENT.glob(f"year={a.years or '*'}/data.parquet")) if pq.ParquetFile(f).metadata.num_rows]

    # works already held (base + updates): never rescued
    held = np.load(BASE_CORPUS / "map-meta" / "id.npy")
    upd = sorted(Path(f"/mnt/fast/scholarsreadinglist/updates/{SNAPSHOT}/corpus").glob("2*/DONE.json"))
    if upd:
        held = np.union1d(held, np.load(upd[-1].parent / "map-meta" / "id.npy"))
    import pyarrow as pa
    held_tab = pa.table({"id": held.astype(np.uint64)})
    con.register("held", held_tab)                       # zero-copy view, kept registered
    log(f"held by base + updates: {len(held):,}")

    # candidate targets by source
    run("CREATE TABLE want (id UBIGINT, src VARCHAR, ref VARCHAR, rank INT)")
    run("CREATE TABLE pm (pmid BIGINT, doi VARCHAR)")
    if "pubmed" in srcs:
        if not have("rows") and con.execute("SELECT count(*) FROM pm").fetchone()[0] == 0:
            con.execute("DROP TABLE pm")
        # ids only: the abstracts are read from the file for the works picked, at the end
        run(f"""CREATE TABLE pm AS SELECT pmid, any_value(doi) AS doi FROM '{PUBMED}'
                        WHERE language = 'eng' AND length(abstract) >= {MIN_ABSTRACT} GROUP BY 1""")
        tidy = json.loads(TIDY.read_text())["tables"]["content"]["files"]
        run(f"""CREATE TABLE oa_pmid AS SELECT CAST(substr(content_id, 11) AS UBIGINT) AS id,
                        TRY_CAST(regexp_extract(pmid, '(\\d+)$', 1) AS BIGINT) AS pmid
                        FROM read_parquet({tidy!r}) WHERE pmid IS NOT NULL""")
        run("INSERT INTO want SELECT o.id, 'pubmed', CAST(o.pmid AS VARCHAR), 0 FROM oa_pmid o JOIN pm USING (pmid)")
        log(f"PubMed records with an English abstract: {con.execute('SELECT count(*) FROM pm').fetchone()[0]:,}; "
            f"OpenAlex works with such a PMID: {con.execute('SELECT count(*) FROM want').fetchone()[0]:,}")
    if "versions" in srcs:
        m = VERSIONS / "members.parquet"
        run(f"""INSERT INTO want SELECT c.work, 'version', CAST(d.work AS VARCHAR), 1 FROM '{m}' c
            JOIN (SELECT cluster, arg_min(work, rank) AS work FROM '{m}' WHERE has_abstract AND (language = 'en' OR language IS NULL)
                  AND rank > 0 GROUP BY 1) d USING (cluster)
            WHERE c.rank = 0 AND NOT c.has_abstract""")
        nv = con.execute("SELECT count(*) FROM want WHERE src = 'version'").fetchone()[0]
        log(f"canonical records without an abstract whose version has one: {nv:,}")
    # one pass over the content snapshot: publications without an abstract, English or unknown, that a source can fill
    cols = ", ".join(bc.COLUMNS)
    run(f"""CREATE TABLE rows AS SELECT CAST(substr(id, 23) AS UBIGINT) AS wid, {cols},
        lower(regexp_replace(doi, '^https://doi.org/', '')) AS doi_l FROM read_parquet({files!r})
        WHERE (abstract IS NULL OR trim(abstract) = '') AND type IN {types!r} AND (language = 'en' OR language IS NULL)
          AND title IS NOT NULL AND trim(title) <> ''
          AND (CAST(substr(id, 23) AS UBIGINT) IN (SELECT id FROM want)
               OR lower(regexp_replace(doi, '^https://doi.org/', '')) IN (SELECT doi FROM pm WHERE doi IS NOT NULL))""")
    log(f"publications without an abstract that a source covers: {con.execute('SELECT count(*) FROM rows').fetchone()[0]:,}")
    # every candidate (work, source), best first: PubMed by PMID, PubMed by DOI, another version
    run("""CREATE TABLE pick AS SELECT id, src, ref FROM (
        SELECT w.id, w.src, w.ref, CASE w.src WHEN 'pubmed' THEN 0 ELSE 2 END AS pr FROM want w JOIN rows r ON r.wid = w.id
        UNION ALL
        SELECT r.wid, 'pubmed_doi', CAST(p.pmid AS VARCHAR), 1 FROM rows r JOIN pm p ON p.doi = r.doi_l) x
        ANTI JOIN held h ON h.id = x.id
        QUALIFY row_number() OVER (PARTITION BY x.id ORDER BY pr, ref) = 1""")
    log(f"picked: {con.execute('SELECT src, count(*) FROM pick GROUP BY 1 ORDER BY 1').fetchall()}")
    run(f"""CREATE TABLE pmtext AS SELECT pmid, arg_max(abstract, length(abstract)) AS abstract,
        arg_max(abstract_copyright, length(abstract)) AS abstract_copyright FROM '{PUBMED}'
        WHERE pmid IN (SELECT TRY_CAST(ref AS BIGINT) FROM pick WHERE src LIKE 'pubmed%') AND language = 'eng' GROUP BY 1""")
    # a version's abstract: the donor is English with an abstract, so the base corpus holds it (fast disk)
    base_parts = [str(p) for p in sorted((BASE_CORPUS / "parts").glob("*.parquet")) if pq.ParquetFile(p).metadata.num_rows]
    run(f"""CREATE TABLE dtext AS SELECT CAST(id AS UBIGINT) AS wid, abstract FROM read_parquet({base_parts!r}, union_by_name=true)
        WHERE CAST(id AS UBIGINT) IN (SELECT TRY_CAST(ref AS UBIGINT) FROM pick WHERE src = 'version')""")
    con.execute("""CREATE OR REPLACE TABLE fill AS
        WITH k AS (SELECT id, src, ref, CASE WHEN src LIKE 'pubmed%' THEN TRY_CAST(ref AS BIGINT) END AS ref_pmid,
                          CASE WHEN src = 'version' THEN TRY_CAST(ref AS UBIGINT) END AS ref_w FROM pick)
        SELECT k.id, k.src, k.ref, coalesce(pt.abstract, d.abstract) AS abstract, pt.abstract_copyright
        FROM k LEFT JOIN pmtext pt ON pt.pmid = k.ref_pmid LEFT JOIN dtext d ON d.wid = k.ref_w""")
    con.execute(f"DELETE FROM fill WHERE coalesce(length(trim(abstract)), 0) < {MIN_ABSTRACT}")
    stats = con.execute("SELECT src, count(*) FROM fill GROUP BY 1 ORDER BY 1").fetchall()
    log(f"rescued: {stats}")
    sel = ", ".join(f"f.abstract AS abstract" if c == "abstract" else ("'en' AS language" if c == "language" else f"r.{c}")
                    for c in bc.COLUMNS)
    con.execute(f"""COPY (SELECT {sel} FROM rows r JOIN fill f ON f.id = r.wid ORDER BY r.wid)
                    TO '{out / 'content'}' (FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 100000)""")
    con.execute(f"""COPY (SELECT id, CASE WHEN src = 'version' THEN 'version:W' || ref ELSE 'pubmed:' || ref END AS abstract_source,
                    abstract_copyright FROM fill ORDER BY id) TO '{out / 'sources.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)   # the working database too

    make_corpus(out, files)


def make_corpus(out: Path, files: list[str]) -> None:
    stats = duckdb.sql(f"SELECT split_part(abstract_source, ':', 1), count(*) FROM '{out / 'sources.parquet'}' GROUP BY 1 ORDER BY 1").fetchall()
    # PubMed's title must be this paper's: OpenAlex sometimes holds a wrong PMID (measured on 181k: 98.9 % of titles
    # agree >= 0.9, 0.05 % are another paper). Kept: folded-title Jaro-Winkler >= 0.7, or one title inside the other.
    if not (out / "title_check.json").exists():
        con = duckdb.connect()
        con.execute("SET threads=8; SET memory_limit='24GB'")
        k = "lower(regexp_replace(strip_accents({}), '[^a-zA-Z0-9]', '', 'g'))"
        bad = con.execute(f"""SELECT s.id FROM '{out / 'sources.parquet'}' s
            JOIN (SELECT CAST(substr(id, 23) AS UBIGINT) AS wid, title FROM read_parquet('{out}/content/*.parquet')) r ON r.wid = s.id
            JOIN '{PUBMED}' p ON p.pmid = TRY_CAST(split_part(s.abstract_source, ':', 2) AS BIGINT)
            WHERE s.abstract_source LIKE 'pubmed%'
              AND jaro_winkler_similarity({k.format('r.title')}, {k.format('p.title')}) < 0.7
              AND NOT contains({k.format('r.title')}, {k.format('p.title')}) AND NOT contains({k.format('p.title')}, {k.format('r.title')})""").fetchnumpy()["id"]
        import pyarrow.compute as pc
        drop = pa.array(np.asarray(bad, np.uint64))
        for f in sorted((out / "content").glob("*.parquet")):
            t = pq.read_table(f)
            wid = pc.cast(pc.replace_substring(t["id"], "https://openalex.org/W", ""), pa.uint64())
            t = t.filter(pc.invert(pc.is_in(wid, value_set=drop)))
            pq.write_table(t, f.with_suffix(".tmp"), compression="zstd", row_group_size=100_000)
            os.replace(f.with_suffix(".tmp"), f)
        src = pq.read_table(out / "sources.parquet")
        src = src.filter(pc.invert(pc.is_in(src["id"], value_set=pa.array(np.asarray(bad, np.uint64)).cast(src["id"].type))))
        pq.write_table(src, out / "sources.parquet.tmp", compression="zstd")
        os.replace(out / "sources.parquet.tmp", out / "sources.parquet")
        (out / "title_check.json").write_text(json.dumps({"dropped_title_mismatch": int(len(bad))}))
        log(f"PubMed title is another paper's: {len(bad):,} dropped")
        stats = duckdb.sql(f"SELECT split_part(abstract_source, ':', 1), count(*) FROM '{out / 'sources.parquet'}' GROUP BY 1 ORDER BY 1").fetchall()

    # the snapshot's exact column types (DuckDB writes text as string; the corpus builder expects large_string)
    snap = pq.read_schema(files[0])
    want_schema = pa.schema([snap.field(c) for c in bc.COLUMNS])
    for f in sorted((out / "content").glob("*.parquet")):
        t = pq.read_table(f)
        if t.schema != want_schema:
            pq.write_table(t.select(bc.COLUMNS).cast(want_schema), f.with_suffix(".tmp"), compression="zstd", row_group_size=100_000)
            os.replace(f.with_suffix(".tmp"), f)

    # the corpus, exactly as the daily updates corpus is built
    corpus = out / "corpus"
    (corpus / "parts").mkdir(parents=True, exist_ok=True)
    pmc = str(BASE_CORPUS / "pmc-by-doi.parquet")
    n = 0
    for f in sorted((out / "content").glob("*.parquet")):
        k = pq.ParquetFile(f).metadata.num_row_groups
        if k:
            n += bc.one_part((str(f), list(range(k)), str(corpus / "parts" / f.name), pmc))["rows"]
    log(f"corpus parts: {n:,} works")
    bc.cmd_sqlite(corpus, threads=8, memory="16GB")
    bc.cmd_keywords(corpus, threads=4, heap_gb=4)
    shutil.rmtree(corpus / "duckdb-tmp", ignore_errors=True)
    subprocess.run([sys.executable, str(REPO / "pipeline" / "release" / "corpus_map_meta.py"), str(corpus)],
                   check=True, env={**os.environ, "PROCS": "4"}, stdout=subprocess.DEVNULL)
    (corpus / "quality").mkdir(exist_ok=True)       # no junk-model facts yet: the model treats them as unknown records
    np.save(corpus / "quality" / "ids.npy", np.zeros(0, np.int64))
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "works": n, "snapshot": SNAPSHOT,
                "selection": "publications without an abstract in OpenAlex, abstract from PubMed or another version (pipeline/rescue)",
                "rescued_by_source": dict(stats), "pubmed": str(PUBMED), "versions": str(VERSIONS), "base_corpus": str(BASE_CORPUS)}
    for p in (corpus / "manifest.json", out / "manifest.json"):
        p.write_text(json.dumps(manifest, indent=1))
    (corpus / "DONE.json").write_text(json.dumps({"works": n, "seconds": round(time.time() - T0)}))
    (out / "DONE.json").write_text(json.dumps({"works": n, "seconds": round(time.time() - T0)}))
    log(json.dumps(manifest))


if __name__ == "__main__":
    main()
