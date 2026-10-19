"""After build_versions.py: rank.npy (if missing) and versions.sqlite, the API's lookup of a paper's versions.

    search/.venv/bin/python pipeline/identity/versions_lookup.py [DIR]

versions.sqlite: members(cluster, rank, work, type, year, doi, venue, has_abstract), PRIMARY KEY (cluster, rank).
"""
import os, sqlite3, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

d = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/fast/scholarsreadinglist/identity/openalex-2026-09-23/versions")
off = np.load(d / "c_off.npy").astype(np.int64)
cw = np.load(d / "c_work.npy")
if not (d / "rank.npy").exists():
    rank = np.arange(len(cw)) - np.repeat(off[:-1], np.diff(off))
    o = np.argsort(cw, kind="stable")
    assert (cw[o] == np.load(d / "work.npy")).all()
    np.save(d / "rank.npy", rank[o].astype(np.uint16))
t0 = time.time()
tmp = d / "versions.sqlite.tmp"
tmp.unlink(missing_ok=True)
con = sqlite3.connect(tmp)
con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE members (cluster INTEGER, rank INTEGER, "
                  "work INTEGER, type TEXT, year INTEGER, doi TEXT, venue TEXT, has_abstract INTEGER, PRIMARY KEY (cluster, rank)) WITHOUT ROWID;")
pf = pq.ParquetFile(d / "members.parquet")
for b in pf.iter_batches(batch_size=500_000, columns=["cluster", "rank", "work", "type", "year", "doi", "venue", "has_abstract"]):
    con.executemany("INSERT INTO members VALUES (?,?,?,?,?,?,?,?)", zip(*[b.column(i).to_pylist() for i in range(8)]))
con.commit(); con.close()
os.replace(tmp, d / "versions.sqlite")
print(f"versions.sqlite: {pf.metadata.num_rows:,} records in {time.time() - t0:.0f}s, {os.path.getsize(d / 'versions.sqlite') / 1e9:.1f} GB")
