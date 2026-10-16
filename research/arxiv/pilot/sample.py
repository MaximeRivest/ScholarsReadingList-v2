"""Pilot sample: arXiv papers (new-style ids) first posted Jan-Sep 2024 (inside unarXive 2024, to compare its
OpenAlex links with ours) and 2019-2021 ones whose OpenAlex record has references (a second yardstick).
Writes /mnt/bulk/arxiv/pilot/sample.json: [{aid, version, w, group}]."""
import glob, json, random, re
import pyarrow.parquet as pq
m = pq.read_table(sorted(glob.glob("/mnt/bulk/arxiv/metadata-snapshot/*.parquet")), columns=["id", "versions"]).to_pylist()
latest = {r["id"]: r["versions"][-1]["version"] for r in m if r["versions"]}
oa = pq.read_table("/mnt/bulk/arxiv/openalex_vs_arxiv.parquet", columns=["id", "aid", "references_count"]).to_pylist()
rows = [r for r in oa if re.fullmatch(r"\d{4}\.\d{4,5}", r["aid"] or "") and r["aid"] in latest]
random.seed(20261016)
a = random.sample([r for r in rows if "2401" <= r["aid"][:4] <= "2409"], 400)
b = random.sample([r for r in rows if "1901" <= r["aid"][:4] <= "2112" and (r["references_count"] or 0) >= 5], 100)
out = [{"aid": r["aid"], "version": latest[r["aid"]], "w": int(r["id"].rsplit("W", 1)[1]), "group": g}
       for g, s in (("2024", a), ("2019-21", b)) for r in s]
json.dump(out, open("/mnt/bulk/arxiv/pilot/sample.json", "w"))
print(len(out), out[:2])
