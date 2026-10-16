"""Our matched references vs OpenAlex's referenced_works, for the pilot papers that have them (2019-21 group).
OpenAlex's lists are not the truth either (Microsoft Academic's PDF parsing); agreement both ways is what we see."""
import json, sys
import duckdb, numpy as np
matched = sys.argv[1]
s = json.load(open("/mnt/bulk/arxiv/pilot/sample.json"))
grp = [x for x in s if x["group"] == "2019-21"]
con = duckdb.connect()
oa = con.execute("""SELECT CAST(substr(id, 23) AS BIGINT) w, "references" FROM read_parquet('/mnt/bulk/openalex/content/year=*/data.parquet')
                    WHERE year BETWEEN 2018 AND 2022 AND CAST(substr(id, 23) AS BIGINT) IN (SELECT unnest(?))""", [[x["w"] for x in grp]]).fetchall()
oa = {w: {int(u.rsplit("W", 1)[1]) for u in (r or "").split(" | ") if "W" in u} for w, r in oa}
ours = {}
for aid, w in con.execute(f"SELECT aid, w FROM read_parquet('{matched}') WHERE w IS NOT NULL").fetchall():
    ours.setdefault(aid, set()).add(w)
P, R, both, only_ours, only_oa = [], [], 0, 0, 0
for x in grp:
    a, b = ours.get(x["aid"], set()), oa.get(x["w"], set())
    if not b or not a: continue
    i = len(a & b); both += i; only_ours += len(a - b); only_oa += len(b - a)
    P.append(i / len(a)); R.append(i / len(b))
print(f"papers {len(P)}: of our links, {np.mean(P):.0%} are also in OpenAlex's list (median paper {np.median(P):.0%}); "
      f"of OpenAlex's links, {np.mean(R):.0%} are also ours (median {np.median(R):.0%}). totals: both {both}, only ours {only_ours}, only OpenAlex {only_oa}")
