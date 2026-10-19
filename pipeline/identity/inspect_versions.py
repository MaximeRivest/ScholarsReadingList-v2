"""Print random groups (and the biggest) of a versions build with titles and first authors, for reading by hand."""
import sys, duckdb, glob
out, years, k = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40
con = duckdb.connect(); con.execute("SET threads=8")
fl = sorted(glob.glob(f"/mnt/bulk/openalex/content/year={years}/data.parquet"))
con.execute(f"CREATE TABLE m AS SELECT * FROM '{out}/members.parquet'")
mode = sys.argv[4] if len(sys.argv) > 4 else "random"
if mode == "big":
    gs = con.execute("SELECT cluster FROM m GROUP BY 1 ORDER BY count(*) DESC LIMIT ?", [k]).fetchall()
else:
    gs = con.execute("SELECT DISTINCT cluster FROM m ORDER BY hash(cluster + 7) LIMIT ?", [k]).fetchall()
gs = [g[0] for g in gs]
con.execute(f"CREATE TABLE t AS SELECT CAST(substr(id,23) AS UBIGINT) w, title, split_part(authors,' | ',1) a1, len(string_split(authors,' | ')) na FROM read_parquet({fl!r}) WHERE CAST(substr(id,23) AS UBIGINT) IN (SELECT work FROM m WHERE cluster IN (SELECT unnest(?)))", [gs])
for g in gs:
    rows = con.execute("SELECT m.rank, m.type, m.year, m.venue, m.doi, m.has_abstract, t.a1, t.na, t.title FROM m LEFT JOIN t ON t.w=m.work WHERE cluster=? ORDER BY rank", [g]).fetchall()
    print(f"--- group {g} ({len(rows)})  {(rows[0][8] or '')[:110]}")
    for r in rows[:8]:
        print(f"   {r[0]} {r[1]:<17} {r[2]} abs={int(r[5])} {str(r[3])[:38]:<38} {str(r[4])[:34]:<34} {r[6]} (+{r[7]-1})")
