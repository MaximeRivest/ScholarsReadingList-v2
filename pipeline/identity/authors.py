"""Author identity, step one: the certain corrections to OpenAlex's author records, and an evaluation set.

    nice -n 10 search/.venv/bin/python pipeline/identity/authors.py [--threads 16] [--memory 64GB]

Input: extract_authorships.py's table (every authorship of the snapshot, with the record OpenAlex chose, the record's
ORCID, the name and ORCID the publisher printed). Nothing here guesses from names alone; every change rests on an
ORCID, and names only veto (names.sql_name_macros: ncompat_orcid, which lets spelling, transliteration, nicknames and
double surnames pass and stops an ORCID on the wrong person: 'Karel Pacák' / 'Abhishek Jha'):

 1. Same ORCID, several records (a split OpenAlex could see): the records whose display name is compatible with the
    record holding most of that ORCID's papers become one person (person id = that record). An incompatible name
    under the same ORCID ('Karel Pacák' / 'Abhishek Jha') is a wrong ORCID, not a merge: left alone, counted.
 2. The publisher printed ORCID R on a publication's authorship that OpenAlex gave to another record (or to none):
    the authorship goes to R's person when the printed name is compatible with that person's name, that person is not
    already on the paper, and R is not an ORCID OpenAlex put on two incompatible names (whose it is is then unknown).
 3. Records that are not people: names that fail names.is_person_name (organisations, placeholders, lists of names),
    and records with 3,000+ publications spanning 60+ years (merges of many people, reprint catalogues). Flagged,
    not deleted: pages can say so and people search can leave them out.
 4. Evaluation set: every authorship with a publisher ORCID, its name block (surname + first initial) and the split
    (test = 10 % of ORCIDs by hash). evaluate.py scores OpenAlex and this overlay on it; ORCID evidence of test
    ORCIDs is marked so a fair score can undo it.

Writes /mnt/fast/scholarsreadinglist/identity/<snapshot>/authors/:
  records.parquet  author, name, orcid, works, year0, year1, person, merged_by_orcid, not_person, not_person_why
  moves.parquet    work, pos, author (NULL = had none), person, raw_orcid, raw_name, why ('orcid_other_record' | 'orcid_orphan')
  eval.parquet     work, pos, raw_orcid, block, author, person_after, split ('train' | 'test')
  meta.json        counts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import names as N  # noqa: E402

SNAPSHOT = "openalex-2026-09-23"
ROOT = Path(f"/mnt/fast/scholarsreadinglist/identity/{SNAPSHOT}")
PUB = ('article', 'review', 'conference-paper', 'conference-abstract', 'preprint', 'book', 'book-chapter',
       'dissertation', 'report', 'letter', 'editorial', 'data-paper', 'software-paper')
TEST = "hash(u.raw_orcid) % 10 = 0"
SMALL, DOMINANT = 3, 0.25          # a record joins its ORCID's person whole when it has <= 3 papers or >= 25 % carry the ORCID
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.time() - T0:7.0f}s {msg}", flush=True)


def reflag(out: Path) -> None:
    """Recompute records.parquet's not_person columns with the current rule (names.is_person_name changed)."""
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='24GB'")
    f = out / "records.parquet"
    con.execute(f"""COPY (SELECT * EXCLUDE (not_person, not_person_why),
        (NOT {N.sql_is_person('name')}) OR (works >= 3000 AND year1 - year0 >= 60) AS not_person,
        CASE WHEN NOT {N.sql_is_person('name')} THEN 'name' WHEN works >= 3000 AND year1 - year0 >= 60 THEN 'size_span' END AS not_person_why
        FROM '{f}') TO '{f}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    os.replace(f"{f}.tmp", f)
    m = json.loads((out / "meta.json").read_text())
    m["not_person"] = list(con.execute(f"""SELECT count(*) FILTER (WHERE not_person), sum(works) FILTER (WHERE not_person),
        count(*) FILTER (WHERE not_person_why = 'size_span') FROM '{f}'""").fetchone())
    (out / "meta.json").write_text(json.dumps(m, indent=1, default=str))
    log(f"not people: {m['not_person']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory", default="64GB")
    ap.add_argument("--authorships", type=Path, default=ROOT / "authorships")
    ap.add_argument("--out", type=Path, default=ROOT / "authors")
    ap.add_argument("--reflag", action="store_true", help="only recompute the not-people flags of a finished run")
    a = ap.parse_args()
    if a.reflag:
        return reflag(a.out)
    if (a.out / "meta.json").exists():
        raise SystemExit(f"{a.out} is finished")
    if not (a.authorships / "DONE.json").exists() and "trial" not in str(a.out):
        raise SystemExit(f"{a.authorships}: extraction not finished")
    a.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(a.out / "work.duckdb"))      # tables kept: a stopped run resumes after its last one
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.memory}'; SET preserve_insertion_order=false; "
                f"SET temp_directory='{a.out / 'tmp'}'")
    N.sql_name_macros(con)
    con.execute(f"CREATE OR REPLACE VIEW au AS SELECT * FROM read_parquet('{a.authorships}/*.parquet')")
    have = lambda t: con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [t]).fetchone()[0] > 0
    M = {}

    # 1. records, and the ORCID merges
    if not have("rec2") and not have("records"):
        con.execute(f"""CREATE OR REPLACE TABLE rec AS SELECT author, any_value(author_name) AS name, upper(any_value(author_orcid)) AS orcid,
            count(*) FILTER (WHERE type IN {PUB}) AS works, min(year) FILTER (WHERE type IN {PUB} AND year > 0) AS year0,
            max(year) FILTER (WHERE type IN {PUB}) AS year1,
            count(*) FILTER (WHERE type IN {PUB} AND raw_orcid = upper(author_orcid)) AS orcid_hits,
            count(*) FILTER (WHERE type IN {PUB} AND raw_orcid IS NOT NULL AND raw_orcid <> upper(author_orcid)) AS orcid_other
            FROM au WHERE author IS NOT NULL GROUP BY 1""")
                # anchor of each ORCID: its record with most ORCID-printed papers (then most papers): the biggest record can be
        # a mixture of namesakes
        con.execute("""CREATE OR REPLACE TABLE anchor AS SELECT orcid, arg_max(author, (orcid_hits, works, -author)) AS person,
            arg_max(name, (orcid_hits, works, -author)) AS pname, count(*) AS nrec FROM rec WHERE orcid IS NOT NULL GROUP BY 1""")
        # the name test only where an ORCID has several records (2.9M of 128M)
        # A whole record joins only when the ORCID evidence covers it: a quarter of its papers carry that printed ORCID
        # (or it has at most 3), and none carries another. OpenAlex's own records can be mixtures ('M. Jones', 291
        # papers from 1975, holding the ORCID of a researcher of the 2000s through a few papers): such a record does
        # not join; its ORCID-printed papers still move one by one (step 2).
        con.execute(f"""CREATE OR REPLACE TABLE multi AS SELECT r.author, x.person,
            ncompat_orcid(r.name, x.pname) AND r.orcid_other = 0 AND (r.works <= {SMALL} OR r.orcid_hits >= {DOMINANT} * r.works) AS ok,
            ncompat_orcid(r.name, x.pname) AS names_ok
            FROM rec r JOIN anchor x USING (orcid) WHERE x.nrec > 1 AND r.author <> x.person""")
        con.execute("""CREATE OR REPLACE TABLE rec2 AS SELECT r.*,
            CASE WHEN m.ok THEN m.person ELSE r.author END AS person,
            coalesce(m.ok, false) AS merged_by_orcid, coalesce(NOT m.names_ok, false) AS orcid_name_conflict,
            coalesce(m.names_ok AND NOT m.ok, false) AS orcid_too_thin
            FROM rec r LEFT JOIN multi m USING (author)""")
        con.execute("DROP TABLE rec")
    src = "rec2" if have("rec2") else "records"
    M["records"] = con.execute(f"SELECT count(*), count(orcid) FROM {src}").fetchone()
    M["orcid_split"] = con.execute("SELECT count(*) FILTER (WHERE nrec > 1), sum(nrec) FILTER (WHERE nrec > 1) FROM anchor").fetchone()
    M["merged"] = con.execute(f"SELECT sum(merged_by_orcid::int), sum(orcid_name_conflict::int), sum(orcid_too_thin::int) FROM {src}").fetchone()
    log(f"records {M['records'][0]:,} (ORCID {M['records'][1]:,}); ORCIDs on 2+ records {M['orcid_split'][0]:,} "
        f"({M['orcid_split'][1]:,} records); merged {M['merged'][0]:,}; name conflicts left alone {M['merged'][1]:,}; "
        f"ORCID evidence too thin to join a whole record {M['merged'][2]:,}")

    # 3. not people (recomputed on every run: cheap, and its rule may change)
    con.execute(f"""CREATE OR REPLACE TABLE records2 AS SELECT author, name, orcid, works, year0, year1, person, merged_by_orcid,
        orcid_name_conflict, orcid_too_thin, orcid_hits, orcid_other,
        (NOT {N.sql_is_person('name')}) OR (works >= 3000 AND year1 - year0 >= 60) AS not_person,
        CASE WHEN NOT {N.sql_is_person('name')} THEN 'name' WHEN works >= 3000 AND year1 - year0 >= 60 THEN 'size_span' END AS not_person_why
        FROM {src}""")
    for t in ("rec2", "records"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute("ALTER TABLE records2 RENAME TO records")
    M["not_person"] = con.execute("""SELECT count(*) FILTER (WHERE not_person), sum(works) FILTER (WHERE not_person),
        count(*) FILTER (WHERE not_person_why = 'size_span') FROM records""").fetchone()
    log(f"not people: {M['not_person'][0]:,} records ({M['not_person'][1]:,} publications); by size+span {M['not_person'][2]:,}")

    # 2. moves: the publisher's ORCID names another person (or the authorship had no record)
    if not have("moves"):
        con.execute("CREATE OR REPLACE TABLE operson AS SELECT orcid, person, pname FROM anchor")
        # candidates first (cheap conditions), the name test only on them
        con.execute(f"""CREATE OR REPLACE TABLE cand AS
            SELECT u.work, u.pos, u.author, o.person, o.pname, u.raw_orcid, coalesce(u.raw_name, u.author_name) AS raw_name,
                   CASE WHEN u.author IS NULL THEN 'orcid_orphan' ELSE 'orcid_other_record' END AS why
            FROM au u JOIN operson o ON o.orcid = u.raw_orcid
            LEFT JOIN records r ON r.author = u.author
            WHERE u.raw_orcid IS NOT NULL AND (u.author IS NULL OR r.person <> o.person) AND u.type IN {PUB}""")
        # an ORCID OpenAlex put on two incompatible names: whose it is is unknown, so it moves nothing
        con.execute("""CREATE OR REPLACE TABLE moves0 AS SELECT work, pos, author, person, raw_orcid, raw_name, why FROM cand
            WHERE raw_orcid NOT IN (SELECT orcid FROM records WHERE orcid_name_conflict AND orcid IS NOT NULL)
              AND ncompat_orcid(raw_name, pname)""")
        M["move_candidates"] = con.execute("SELECT count(*) FROM cand").fetchone()[0]
        # not when that person is already on the paper (another position): one person twice on a byline is an error
        con.execute("""CREATE OR REPLACE TABLE onpaper AS SELECT u.work, u.pos, r.person FROM au u JOIN records r ON r.author = u.author
            WHERE u.work IN (SELECT DISTINCT work FROM moves0)""")
        con.execute("""CREATE OR REPLACE TABLE moves AS SELECT m.* FROM moves0 m
            WHERE NOT EXISTS (SELECT 1 FROM onpaper p WHERE p.work = m.work AND p.pos <> m.pos AND p.person = m.person)
            QUALIFY count(*) OVER (PARTITION BY m.work, m.person) = 1""")
    M["moves"] = con.execute("SELECT why, count(*) FROM moves GROUP BY 1 ORDER BY 1").fetchall()
    M["moves_vetoed"] = con.execute("SELECT (SELECT count(*) FROM moves0) - (SELECT count(*) FROM moves)").fetchone()[0]
    log(f"moves {M['moves']}; vetoed (person already on the paper) {M['moves_vetoed']:,}")

    # 4. evaluation set
    con.execute(f"""CREATE OR REPLACE TABLE eval AS
        SELECT u.work, u.pos, u.raw_orcid, nsur(coalesce(u.raw_name, u.author_name)) || ' ' || left(nfirst(coalesce(u.raw_name, u.author_name)), 1) AS block,
               u.author, coalesce(m.person, r.person) AS person_after, u.year, u.type, u.n_authors,
               CASE WHEN {TEST} THEN 'test' ELSE 'train' END AS split
        FROM au u LEFT JOIN records r ON r.author = u.author LEFT JOIN moves m ON m.work = u.work AND m.pos = u.pos
        WHERE u.raw_orcid IS NOT NULL AND u.type IN {PUB}""")
    M["eval"] = con.execute("SELECT split, count(*), count(DISTINCT raw_orcid) FROM eval GROUP BY 1 ORDER BY 1").fetchall()
    log(f"evaluation authorships {M['eval']}")

    for t in ("records", "moves", "eval"):
        con.execute(f"COPY {t} TO '{a.out / (t + '.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    meta = {"snapshot": SNAPSHOT, "test_split": TEST, "seconds": round(time.time() - T0),
            **{k: (list(map(list, v)) if isinstance(v, list) else (list(v) if isinstance(v, tuple) else v)) for k, v in M.items()}}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    os.remove(a.out / "work.duckdb")
    log(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
