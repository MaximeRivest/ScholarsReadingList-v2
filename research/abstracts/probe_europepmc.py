"""Probe: for English publications that OpenAlex holds WITHOUT an abstract, how often does Europe PMC (PubMed + PMC +
preprints) have one? 1,000 random DOIs from the audit's 1M sample (research/audit/sample), 2000-2024, articles/reviews.
Free REST API, ~5 requests/s. Output: probe_results.json."""
import duckdb, json, time, urllib.request, urllib.parse, collections, random
con = duckdb.connect()
rows = con.execute("""SELECT id, lower(replace(doi,'https://doi.org/','')) d, publication_year, pmid IS NOT NULL has_pmid, source_name
  FROM read_parquet('research/audit/sample/*.parquet')
  WHERE NOT has_abstract AND doi IS NOT NULL AND language='en' AND type IN ('article','review') AND NOT is_paratext
    AND publication_year BETWEEN 2000 AND 2024 ORDER BY hash(id) LIMIT 1000""").fetchall()
out = []
for wid, doi, yr, has_pmid, src in rows:
    q = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "resultType": "core", "format": "json", "pageSize": 1})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q, timeout=30)); break
        except Exception as e:
            time.sleep(2); r = None
    res = (r or {}).get("resultList", {}).get("result", [])
    ab = res[0].get("abstractText") if res else None
    out.append({"work": wid, "doi": doi, "year": yr, "openalex_has_pmid": has_pmid, "source": src, "in_europepmc": bool(res),
                "europepmc_source": res[0].get("source") if res else None, "abstract_len": len(ab or ""), "pmid": res[0].get("pmid") if res else None})
    time.sleep(0.15)
n = len(out); hit = sum(o["abstract_len"] >= 100 for o in out)
pref = collections.defaultdict(lambda: [0, 0])
for o in out:
    p = o["doi"].split("/")[0]; pref[p][0] += 1; pref[p][1] += o["abstract_len"] >= 100
summ = {"n": n, "found_in_europepmc": sum(o["in_europepmc"] for o in out), "with_abstract": hit,
        "with_abstract_and_openalex_has_pmid": sum(o["abstract_len"] >= 100 and o["openalex_has_pmid"] for o in out),
        "openalex_has_pmid": sum(o["openalex_has_pmid"] for o in out),
        "by_prefix": {k: v for k, v in sorted(pref.items(), key=lambda x: -x[1][0])[:15]}}
print(json.dumps(summ, indent=1))
json.dump({"summary": summ, "rows": out}, open("research/abstracts/probe_results.json", "w"), indent=1)
