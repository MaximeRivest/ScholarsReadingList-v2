"""Random version groups from the finished build, with titles from the citation index (fast, no snapshot read)."""
import sqlite3, sys, numpy as np
C = "/mnt/fast/citewalk/20261007T003120Z-50ce3865/"
D = "/mnt/fast/scholarsreadinglist/identity/openalex-2026-09-23/versions/"
k, seed = int(sys.argv[1]), int(sys.argv[2])
wn = np.load(C + "wn.npy", mmap_mode="r"); tr = np.load(C + "title_row.npy", mmap_mode="r")
to = np.load(C + "title_off.npy", mmap_mode="r"); tb = np.memmap(C + "titles.bin", np.uint8, "r")
def title(w):
    n = int(np.searchsorted(wn, w)); r = int(tr[n]) if n < len(wn) and wn[n] == w else None
    return bytes(tb[to[r]:to[r + 1]]).decode("utf-8", "replace") if r is not None else "?"
con = sqlite3.connect(f"file:{D}versions.sqlite?mode=ro", uri=True)
G = int(np.load(D + "c_off.npy", mmap_mode="r").shape[0]) - 1
for g in np.random.default_rng(seed).choice(G, k, replace=False):
    rows = con.execute("SELECT rank, type, year, venue, doi, has_abstract, work FROM members WHERE cluster=? ORDER BY rank", (int(g),)).fetchall()
    print(f"--- {len(rows)}  {title(rows[0][6])[:120]}")
    for r in rows[:5]:
        print(f"   {r[0]} {r[1]:<16} {r[2]} abs={r[5]} {str(r[3])[:40]:<40} {r[4]}")
