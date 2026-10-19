"""Score author identity against publisher-printed ORCIDs: OpenAlex's records, and our overlay (authors.py).

    search/.venv/bin/python pipeline/identity/evaluate.py [--split test] [--out .../authors/scores.json]

Truth: two authorships are one person when the publisher printed the same ORCID on both. Only authorships with a
printed ORCID are scored (15 % of publication authorships; ORCID holders are more often recent, prolific and
Western/East-Asian: Kim 2021, doi:10.1007/s11192-020-03826-6 — the numbers describe them, not every author).
Authorships OpenAlex left without a record count as people of their own (that is what a reader sees).

Measures, as S2AND reports them (doi:10.1109/jcdl52503.2021.00029):
  B3 precision / recall / F1 over the scored authorships (each authorship: share of its predicted person that is
  truly it; share of its true person found in its predicted person);
  pairwise precision / recall within name blocks (surname + first initial): the pairs a disambiguator must decide.
Variants:
  openalex            the record OpenAlex chose
  overlay             authors.py: ORCID merges and moves, everything
  overlay_fair        the overlay without any evidence from the ORCID being scored (test ORCIDs never helped
                      their own authorships): what the overlay does for authorships whose ORCID it could not use
Breakdowns: by how many scored authorships the ORCID has, by name origin (East-Asian surnames vs others: the hard
case in WhoIsWho), by publication decade.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from authors import ROOT  # noqa: E402

# The commonest Chinese, Korean, Vietnamese and Japanese surnames (romanised): blocks under them are the hardest.
EAST_ASIAN = ("wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu guo he lin gao luo zheng liang xie song "
              "tang han feng deng cao peng zeng xiao tian dong pan yuan cai jiang yu du ye cheng wei su lu ding ren shen "
              "yao jin fu fan zou kim lee park choi jung kang cho yoon jang lim shin han oh seo kwon hwang ahn song "
              "nguyen tran le pham hoang phan vu vo dang bui do ho ngo duong ly sato suzuki takahashi tanaka watanabe "
              "ito yamamoto nakamura kobayashi kato yoshida yamada sasaki yamaguchi matsumoto inoue kimura").split()


def score(con, cluster_expr: str, where: str) -> dict:
    con.execute(f"CREATE OR REPLACE TEMP TABLE s AS SELECT raw_orcid AS t, {cluster_expr} AS c, block FROM e WHERE {where}")
    n = con.execute("SELECT count(*) FROM s").fetchone()[0]
    if not n:
        return {"n": 0}
    p, r = con.execute("""WITH ct AS (SELECT c, t, count(*) AS n FROM s GROUP BY 1, 2),
        cn AS (SELECT c, count(*) AS n FROM s GROUP BY 1), tn AS (SELECT t, count(*) AS n FROM s GROUP BY 1)
        SELECT sum(ct.n * ct.n / cn.n) / (SELECT count(*) FROM s), sum(ct.n * ct.n / tn.n) / (SELECT count(*) FROM s)
        FROM ct JOIN cn USING (c) JOIN tn USING (t)""").fetchone()
    tp, pred, truth = con.execute("""SELECT
        (SELECT sum(n * (n - 1) / 2) FROM (SELECT count(*) n FROM s GROUP BY block, c, t)),
        (SELECT sum(n * (n - 1) / 2) FROM (SELECT count(*) n FROM s GROUP BY block, c)),
        (SELECT sum(n * (n - 1) / 2) FROM (SELECT count(*) n FROM s GROUP BY block, t))""").fetchone()
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0
    return {"n": n, "b3_precision": round(p, 4), "b3_recall": round(r, 4), "b3_f1": round(f(p, r), 4),
            "pair_precision": round(tp / pred, 4) if pred else None, "pair_recall": round(tp / truth, 4) if truth else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authors", type=Path, default=ROOT / "authors")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("SET threads=16; SET memory_limit='48GB'")
    d = a.authors
    # overlay_fair: for a scored ORCID, undo any move made on its evidence and any merge of a record carrying it
    con.execute(f"""CREATE TABLE e AS SELECT ev.*, r.orcid AS rec_orcid, r.person AS rec_person, m.raw_orcid AS moved_on,
        split_part(ev.block, ' ', 1) IN {tuple(EAST_ASIAN)!r} AS east_asian, (ev.year // 10) * 10 AS decade
        FROM '{d}/eval.parquet' ev LEFT JOIN '{d}/records.parquet' r ON r.author = ev.author
        LEFT JOIN '{d}/moves.parquet' m ON m.work = ev.work AND m.pos = ev.pos WHERE ev.split = '{a.split}'""")
    con.execute("CREATE TEMP TABLE prod AS SELECT raw_orcid, count(*) AS k FROM e GROUP BY 1")
    con.execute("CREATE TABLE e2 AS SELECT e.*, p.k FROM e JOIN prod p USING (raw_orcid)")
    con.execute("DROP TABLE e; ALTER TABLE e2 RENAME TO e")
    single = "coalesce(CAST(author AS VARCHAR), 'orphan:' || work || ':' || pos)"
    variants = {
        "openalex": single,
        "overlay": f"coalesce(CAST(person_after AS VARCHAR), {single})",
        # fair: a move made on this ORCID is undone; a merge whose record carries this ORCID is undone
        "overlay_fair": f"""CASE WHEN moved_on = raw_orcid THEN {single}
                                 WHEN rec_orcid = raw_orcid THEN {single}
                                 ELSE coalesce(CAST(person_after AS VARCHAR), {single}) END""",
    }
    out = {"split": a.split, "all": {}, "by_orcid_size": {}, "by_name_origin": {}, "by_decade": {}}
    for v, expr in variants.items():
        out["all"][v] = score(con, expr, "true")
        for lo, hi in ((2, 4), (5, 19), (20, 99), (100, 10**9)):
            out["by_orcid_size"].setdefault(f"{lo}-{hi if hi < 10**9 else ''}", {})[v] = score(con, expr, f"k BETWEEN {lo} AND {hi}")
        for lab, w in (("east_asian", "east_asian"), ("other", "NOT east_asian")):
            out["by_name_origin"].setdefault(lab, {})[v] = score(con, expr, w)
        for dec in (1990, 2000, 2010, 2020):
            out["by_decade"].setdefault(str(dec), {})[v] = score(con, expr, f"decade = {dec}")
        print(v, out["all"][v], flush=True)
    (d / f"scores-{a.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_orcid_size"], indent=1))


if __name__ == "__main__":
    main()
