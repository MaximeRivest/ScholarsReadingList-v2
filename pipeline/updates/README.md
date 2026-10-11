# Daily updates: new OpenAlex papers between snapshots

The base data (details corpus, embeddings, citation index, map) comes from one public OpenAlex snapshot.
OpenAlex publishes that snapshot **four times a year** (second Wednesday of January, April, July, October). A
daily copy is a paid plan ($10,000/year, Member+). The API, though, is up to date every day, and a free API
key allows $1 of use per day: 1 million paper records. This pipeline uses it to add the papers OpenAlex
created since the base snapshot, every day, until the next snapshot replaces the base.

```
srl-updates.timer (01:00)        pipeline/updates/daily.sh
  fetch.py   API -> content/<run>-<n>.parquet (new works only), raw/ (the API's records), facts/ (junk-model facts)
  corpus.py  all content -> corpus/<run>/ (parts, works.sqlite, keywords, map-meta, quality); --gc
  embed.py   works without a vector -> shards beside the backfill's (year=Y/rg-u<run>/part-<date>-<start>)
srl-release-nightly.timer (03:30) release.py: rows/index/smr/tiles/map from base + updates corpus, publish
```

Everything is under `/mnt/fast/scholarsreadinglist/updates/<base id>/` (`common.py`). The daily shards live in
the backfill's shard tree, so every reader of shards (rows.py, positions.py, the search API, citewalk) takes
them as they are.

```sh
pipeline/updates/daily.sh                 # the whole daily run (what the timer runs)
pipeline/updates/daily.sh status          # papers held, published dates, last runs, daily shards
uv run pipeline/updates/fetch.py --dry-run          # what would be downloaded, and its cost
uv run pipeline/updates/check_api.py                # live check: API text == snapshot text
journalctl --user -u srl-updates
```

**API key.** Put it in `~/.config/openalex/api_key` (one line, `chmod 600`), or `OPENALEX_API_KEY`.
Without one the budget is $0.10/day (about 80,000 papers); with the free key $1/day. Make an account at
openalex.org (free), copy the key from Settings → API key.

## What is fetched, and how

- **Selection:** language `en`, a non-empty abstract, a publication type (article, review, preprint, … the
  types releases keep), OpenAlex's "expansion pack" included (`include_xpac=true`) like the snapshot.
- **Which papers:** publication dates from 30 days before the base snapshot to 60 days ahead of today. The
  free API cannot select "created since" (`from_created_date` is a paid filter), so for every date it asks
  how many works match (1 call), then how they split over the 26 OpenAlex fields (1 call), and downloads
  again only the (date, field) cells whose count changed since they were last downloaded.
- **Same columns, same text:** API records are turned into the snapshot's raw format and flattened by
  `pipeline/openalex_content.py`'s own `flatten`. `check_api.py`: for records OpenAlex did not change since
  the snapshot, the embedded text is identical (16/16 on two samples, 2026-10-11). Vectors made by
  `embed.py` for backfill works: cosine ≥ 0.9995 to the stored ones, each its own nearest (512/512).
- **Same vectors:** `embed.py` uses the backfill's encoder and shard writer with the backfill manifest's
  settings, so a daily shard carries the backfill's `plan_id` (one vector space); `task.source` says
  `openalex-api-daily`.
- **Junk-model facts:** search ranks by similarity minus 0.3 × P(junk); the junk model reads record facts
  (venue type, references, indexes, …). Without them a new paper counts as an "unknown record" and was
  ranked lower: median P(junk) 0.10 vs 0.02 for comparable base articles. `fetch.py` saves them
  (`facts/`), `corpus.py` puts them in `corpus/<run>/quality/`: median 0.018 then. Records saved before
  this existed are re-read once, 100 per call (`refresh_records`).
- **Budget:** each run spends at most $0.80 and leaves $0.10 of the day's budget (free key) for other uses
  of the key; when it runs out it stops, and the next run continues where it stopped. Measured: ~60,000
  papers in 2.5 minutes for $0.07; embedding ~280 papers/s on GPU 1 (3 GiB), the CPU when no GPU has room.

## What is not updated daily (until the next snapshot)

- Works OpenAlex adds now with an old publication date (older than 30 days before the base).
- Changes to works the base already holds: new or corrected abstracts, titles, labels, citation counts,
  deletions and merges. Daily works keep their labels and counts as of the day they were fetched.
- The citation walk (citewalk's citation index) and the readable-on-site flag (PMC mirror).
- Labels differ in vintage: OpenAlex re-ran its topic classifier after the 2026-09-23 snapshot (about a
  quarter of publications changed field), so daily papers carry the new labels, base papers the old ones.

## When the next snapshot comes (2026-10-14)

The updates belong to one base (`common.BASE_ID`). After a new base is built (content, embeddings of new
or changed texts, corpus, citations, quality), point `common.py` at it: a new, empty updates directory
starts, and releases built from the new base no longer read the old updates.
