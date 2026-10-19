"""PubMed XML (baseline + update files) -> one parquet of the current record per PMID: what abstract rescue needs.

    nice -n 10 search/.venv/bin/python pipeline/pubmed/parse.py [--root /mnt/bulk/pubmed] [--procs 8]

Writes ROOT/parsed/<file>.parquet per input file (resumable), then ROOT/pubmed.parquet: one row per PMID, the record
from the latest file (update files after the baseline, in number order), PMIDs deleted by a later file removed.
Columns: pmid BIGINT, doi (lower case), title, abstract (sections joined as 'LABEL: text', inline markup kept as
text), abstract_copyright (the record's CopyrightInformation, if any), language (PubMed's, ISO 639-2), year, journal,
file. OtherAbstract (translations) is ignored.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

WS = re.compile(r"\s+")


def text(el) -> str:
    return WS.sub(" ", "".join(el.itertext())).strip() if el is not None else ""


def one(args) -> tuple[str, int]:
    src, dst = map(Path, args)
    if dst.exists():
        return src.name, -1
    rows, deleted = [], []
    with gzip.open(src, "rb") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "PubmedArticle":
                mc = el.find("MedlineCitation")
                art = mc.find("Article")
                pmid = int(mc.findtext("PMID"))
                ab = art.find("Abstract")
                parts = []
                cp = ""
                if ab is not None:
                    for at in ab.findall("AbstractText"):
                        t = text(at)
                        if not t:
                            continue
                        lab = at.get("Label")
                        parts.append(f"{lab}: {t}" if lab and lab.upper() not in ("UNLABELLED", "UNASSIGNED") else t)
                    cp = text(ab.find("CopyrightInformation"))
                doi = None
                for aid in el.findall("PubmedData/ArticleIdList/ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text.strip().lower()
                if doi is None:
                    for eid in art.findall("ELocationID"):
                        if eid.get("EIdType") == "doi" and eid.text:
                            doi = eid.text.strip().lower()
                y = art.findtext("Journal/JournalIssue/PubDate/Year") or (art.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "")[:4]
                rows.append({"pmid": pmid, "doi": doi, "title": text(art.find("ArticleTitle")),
                             "abstract": " ".join(parts) or None, "abstract_copyright": cp or None,
                             "language": art.findtext("Language"), "year": int(y) if y.isdigit() else None,
                             "journal": art.findtext("Journal/Title")})
                el.clear()
            elif el.tag == "DeleteCitation":
                deleted += [int(p.text) for p in el.findall("PMID")]
                el.clear()
    t = pa.Table.from_pylist(rows, schema=pa.schema([("pmid", pa.int64()), ("doi", pa.string()), ("title", pa.string()),
                                                    ("abstract", pa.string()), ("abstract_copyright", pa.string()),
                                                    ("language", pa.string()), ("year", pa.int32()), ("journal", pa.string())]))
    t = t.append_column("deleted", pa.array([False] * t.num_rows))
    if deleted:
        d = pa.table({"pmid": pa.array(deleted, pa.int64())})
        for c in t.column_names[1:-1]:
            d = d.append_column(c, pa.nulls(len(deleted), t.schema.field(c).type))
        d = d.append_column("deleted", pa.array([True] * len(deleted)))
        t = pa.concat_tables([t, d])
    tmp = dst.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    os.replace(tmp, dst)
    return src.name, t.num_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/bulk/pubmed"))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = a.root / "parsed"
    out.mkdir(exist_ok=True)
    files = sorted((a.root / "baseline").glob("*.xml.gz")) + sorted((a.root / "updatefiles").glob("*.xml.gz"))
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        for i, (name, n) in enumerate(ex.map(one, [(f, out / (f.parent.name + "__" + f.name.replace(".xml.gz", ".parquet"))) for f in files]), 1):
            if i % 100 == 0:
                print(f"{time.time() - t0:6.0f}s {i}/{len(files)}", flush=True)
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET threads=8; SET memory_limit='24GB'; SET preserve_insertion_order=false; SET temp_directory='{a.root / 'tmp'}'")
    final = a.root / "pubmed.parquet"
    src = f"read_parquet('{out}/*.parquet', filename=true)"
    # file order = baseline before updates, numbers ascending: a PMID's record in its last file is current (the
    # choice reads two small columns only; a deletion in a later file removes it)
    con.execute(f"CREATE TABLE last AS SELECT pmid, max(filename) AS filename FROM {src} GROUP BY 1")
    con.execute(f"""COPY (SELECT r.* EXCLUDE (deleted, filename), r.filename AS file FROM {src} r JOIN last USING (pmid, filename)
        WHERE NOT r.deleted) TO '{final}.tmp' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    os.replace(f"{final}.tmp", final)
    n = con.execute(f"SELECT count(*), count(abstract), count(doi) FROM '{final}'").fetchone()
    (a.root / "pubmed.json").write_text(json.dumps({"pmids": n[0], "with_abstract": n[1], "with_doi": n[2],
                                                    "files": len(files), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(f"{time.time() - t0:.0f}s pubmed.parquet: {n[0]:,} PMIDs, {n[1]:,} with an abstract, {n[2]:,} with a DOI", flush=True)


if __name__ == "__main__":
    main()
