# Data for sciencemadereadable.com: what exists, how it is updated, how the site reaches it

One **data release** feeds everything the site shows: search (by meaning and by exact words),
"similar papers", paper details, the map of science (positions, tiles, counts) and the citation
walk. Releases are built from what the embedding backfill has committed, checked, and switched
in atomically; the previous release stays on disk for a one-command rollback.

## How the site reaches the data

```
browser (sciencemadereadable.com, GitHub Pages)
   │  https://lambda.tail69222b.ts.net:10000/api/...      Tailscale Funnel (public)
   ▼
smr-server  :8799 public door  (sciencemadereadable/app/server.py)
   │  /api/search            the site's existing search (collection smr, Hit shape)
   │  /api/data/<route>      read-only pass-through, per-address limits (60/min; tiles 1200/min)
   ▼
srl-search  :8810  data API (search/srl_search/app.py), serves release/current
   │  /search /similar/REF /works/REF          203.5M details; semantic over the release index
   │  /map/place /map/regions /map/counts      positions of any paper or question; field labels;
   │                                           exact publications per year x subfield
   │  /map/tiles/<release>/index.json, z/x/y.pts, z/x/y.ids     every paper on the map, by zoom
   │  /walk?id=W...|DOI                        citation walk + map position + "readable" per paper
   ▼
citewalk    :8798  citation engine (80 GB memory-mapped citation index, 3.0B citations)
```

Routes reachable from the internet under `/api/data/`: `search`, `similar/REF`, `works/REF`,
`map/place`, `map/regions`, `map/counts`, `map/tiles/RELEASE/(index.json|z/x/y.pts|z/x/y.ids)`,
`walk`, and the API's documentation: `llms.txt` (guide for programs and agents, with live coverage), `docs`
(interactive reference), `openapi.json` (public routes only). Nothing else of srl-search (no `/health`, no
test page) is exposed.

## What a release contains

`/mnt/fast/scholarsreadinglist/release/`

| Product | Where | Built by | Changes when |
|---|---|---|---|
| rows | `rows/<id>/` | `pipeline/release/rows.py` | every release (append-only order) |
| search index | `index/<id>/` | `pipeline/release/build_index.py` | every release |
| readable collection | `smr/<id>/` | `search/build_smr.py` | every release |
| map positions | `positions/<atlas>/<shard>/xy.npy` | `pipeline/release/positions.py` | once per shard, ever |
| map tiles + position table | `tiles/<id>/` | `pipeline/release/tiles.py` | every release |
| list centres (quantizer) | `quantizer/<id>/` | `pipeline/release/train_quantizer.py` | rarely (see below) |
| paper details, keyword index | `search/corpus-2026-10-05/` | `search/build_corpus.py` | new OpenAlex snapshot |
| citation index | `/mnt/fast/citewalk/<generation>/` | `citewalk/build_index.py` | new OpenAlex snapshot |
| authorship index (who wrote each paper, position; for topic maps' experts) | `/mnt/fast/citewalk/<generation>/authors/` | `citewalk/build_authors.py` (~45 min) | new OpenAlex snapshot |
| each author's papers + per-author figures (author pages) | `/mnt/fast/citewalk/<generation>/authors/` (by_*.npy, a_*.npy) | `citewalk/build_author_works.py` (~20 min, 7.5 GB) | new OpenAlex snapshot, after build_authors |
| institution index (a paper's organisations, an organisation's papers, each author's affiliations) | `/mnt/fast/citewalk/<generation>/institutions/` | `citewalk/build_institutions.py` (~30 min, 8.6 GB) | new OpenAlex snapshot, after build_authors |
| per-authorship affiliations (who signed from which organisation on each paper; a paper's page) | `/mnt/fast/citewalk/<generation>/institutions/authorships/` | `citewalk/build_authorships.py` (~36 min, 44 GB peak, 8.8 GB) | new OpenAlex snapshot, after build_institutions |
| source index (each paper's journal, conference or repository; each source's papers, figures, publisher, open access) | `/mnt/fast/citewalk/<generation>/sources/` | `citewalk/build_sources.py` (~11 min, 3.8 GB) | new OpenAlex snapshot, after build_index |
| names index (authors and institutions by name) | `/mnt/fast/scholarsreadinglist/search/names-index/<generation>[-institutions]/` | `search/build_names_index.py [--institutions]` (~15 min, 2.2 GB) | after the two above |
| topic-map answers (cache) | `/mnt/fast/scholarsreadinglist/search/topics-cache/<release>/` | srl-search on request | every release (old ones deleted) |
| atlas geography | `/mnt/fast/science-atlas/build/<id>/` | `citewalk/science/` | deliberately only (versioned) |
| release manifest | `releases/<id>/manifest.json` | `release.py build` | every release |

`current` → `releases/<id>`. Anything that leaves a release (URLs, the site, the citation walk)
uses **OpenAlex W-numbers**; row numbers are internal to a release.

### Rules shared by search and map

- **Publications only** (`citewalk/science/scope.py` = `search/srl_search/scope.py`, a test
  keeps them equal): articles, reviews, conference papers/abstracts, preprints, books,
  chapters, theses, reports, letters, editorials, data/software papers. Datasets, figures,
  instrument logs, covers etc. stay in the details store and can be asked for with `type=`.
- **One row per distinct text** in the search index: byte-identical title+abstract (BLAKE2b-128
  from the backfill) keeps the smallest W-number. First release: 75.9M publications, 3.6M
  duplicates dropped.
- **Readable** = the site can open it: PMC mirror, CC BY, research article, DOI-matched.

### Map positions

Every embedded paper gets the frozen atlas function's position (10 nearest of 291,556 reference
papers, exact). On the GPU: fp16 scores pick 64 candidates, re-scored in fp32; checked identical
to exact CPU search (max difference 3e-6 map widths). ~60,000 papers/s on an RTX 3090, using
≤ 2 GB. Positions are computed once per embedding shard and reused by every release. When no
GPU has room, papers not placed yet are left out of the tiles only; the API still places any
paper on request from its vector (HNSW, 99.5 % exact, ~1 ms each).

### Map tiles

Every kept paper with a position, nested by zoom (`tiles.py` docstring): level 0 holds 0.25 %
of papers in one tile, each level 3× more papers in 4× more tiles, level 5 the rest. Zooming in
only adds papers (a paper never moves or disappears), points per screen stay roughly constant,
and inside a tile points are in random order, so any prefix is a fair sample. 8 bytes per point
(`.pts`: x, y, year + readable bit, subfield) and 8 per id (`.ids`, fetched only on click).
Served with the release id in the URL: cache forever.

## Updating

Nightly at 03:30: `srl-release-nightly.timer` runs `release.py nightly`: if the backfill
committed new shards since the current release, build a release, check it, publish it.

```sh
cd ~/Projects/scholarsreadinglist
search/.venv/bin/python pipeline/release/release.py status
search/.venv/bin/python pipeline/release/release.py build            # by hand
search/.venv/bin/python pipeline/release/release.py publish <id>     # checks, switches, restarts, smoke test
search/.venv/bin/python pipeline/release/release.py rollback
search/.venv/bin/python pipeline/release/release.py gc --keep 2 [--yes]
systemctl --user start srl-release-nightly      # run the nightly job now
journalctl --user -u srl-release-nightly
```

**Publish** refuses a release whose searchable or readable counts dropped more than 2 %, then
restarts srl-search on it, waits for `/health`, runs a search and a walk, and switches back to
the previous release if any of that fails. The search API is down for ~1–3 minutes during the
restart; the site falls back to Europe PMC meanwhile.

## Daily updates (since 2026-10-11)

OpenAlex's free snapshot comes four times a year; its API every day. `srl-updates.timer` (01:00) runs
`pipeline/updates/daily.sh`: new English publications with an abstract from the API → a small **updates
corpus** (details, keyword index, map facts, junk-model facts) beside the base corpus → vectors in daily
shards beside the backfill's. The nightly release then builds from base + updates: `rows.py` and
`mapdata.py` take both corpora (`--corpus` twice), the manifest names the updates corpus
(`paths.updates`), and the search API reads its details, keywords and facts after the base's. The release
reads shards and corpus under the updates lock, so a daily run never changes them mid-read. Details, costs,
checks and what is not updated daily: `pipeline/updates/README.md`.

## First release: 20261009T120544Z (published 2026-10-09)

143.8M embedded rows (backfill at 2011-12) → 76.2M publications → **72.5M searchable** (3.6M exact
duplicate texts dropped), all on the map; **3.73M readable on the site** (was 3.05M).

| | old index (2024+, 64M) | this release (2012+, 72.5M) |
|---|---|---|
| exact top 10 found, 440 questions, 128 lists | 0.892 | **0.921** |
| hardest set (NFCorpus) | 0.744 | 0.793 |
| index search p50 / p95 | 37 / 70 ms | **19 / 24 ms** |

(`search/experiments/eval-release-20261009T120544Z.json`.) Warm end-to-end search with question
encoding: ~25–105 ms. Better than the old index because the 32,768 lists were trained on an
even sample of every year, not on the newest papers.

Build times on the shared machine at low priority: rows 4 min, index 55 min, readable collection
10 min, tiles 4 min, positions for the whole corpus 48 min on GPU 0 (only new shards later).
Size: ~29 GB per release (rows 2.7, index 7.7, readable 16, tiles 2.2), plus 1.2 GB of positions
shared. Peak memory per step: rows 17 GB, tiles 11 GB, readable collection ≤ 20 GB (DuckDB cap).
Tiles: 9 levels, 5,018 tiles, at most 32,768 points (256 KB) each.

### What changes when

- **New embeddings** (nightly: the daily updates' shards since 2026-10-11): new rows,
  index, readable collection, tiles. Geography and list centres stay.
- **List centres**: trained once on the atlas's uniform 1 % sample of all years (1.28M vectors,
  32,768 lists, 18 min). Retrain when the corpus grows by ~30 % or the recall check drops.
- **New OpenAlex snapshot**: rebuild the details/keyword corpus, the citation index and its authorship index (then
  `build_author_works.py`, `build_institutions.py`, `build_names_index.py` with and without `--institutions`: the
  author and institution pages), then point `CORPUS` / `CITATIONS` in `release.py` at them. Not automated yet.
- **New atlas geography** (new reference set or method): a new atlas id; positions are
  recomputed for every shard (~45 min GPU) and old releases keep the old map. Never silently.
- **PMC mirror** (readable papers): daily NCBI inventory diff, not automated yet
  (`pipeline/pmc/README.md`, "Next: updates").

## Machine services (declared in `~/Projects/os/machines/hm/scholarsreadinglist-data.nix`)

`srl-search` (CPUWeight 1000, MemoryLow 16G: the public service wins), `citewalk`,
`science-atlas` (pilot viewer), `srl-release-nightly` (+ timer; Nice 10, CPUWeight 20). The
embedding backfill and smr-server/worker have their own files next to it.

## Known gaps

- (2026-10-11: the backfill is complete, 203,486,035 rows; search covers every year.) Earlier: semantic search covered what the backfill had embedded (to ~2012 on 2026-10-09; ~1.5 days to
  go). Keyword search and details already cover all 203.5M.
- The site's page does not use `/api/data/` yet: that is the design work in progress.
- **Map positions of new shards need GPU room.** The nightly job uses GPU 1 only if 2 GB are free
  and never waits. Until the backfill ends, GPU 1 is full (qwen3-embed-4b ~12 GB + backfill +
  parakeet), so new papers are searchable but not in the tiles; `/map/place`, `/search?map=true`
  and `/walk` still place them on request. GPU 0 is left to the site's 9B. To place them by hand
  when a GPU is free: `cd pipeline/embeddings && ATLAS_GPU=0 .venv/bin/python ../release/positions.py`
  then rebuild that release's tiles (or wait for the next nightly).
- **Keyword match counts** include non-publication records for broad words ("cancer": 5.68M
  counted, 5.00M publications). Results are publications only. Filtering inside the index costs
  100–500 ms on broad words, so it is used only when the top hits are mostly non-publications.
  Proper fix: a publications flag in the keyword index at its next rebuild.
- A new citation walk takes ~2.3 s in the citewalk engine (repeats are cached).
- No automatic OpenAlex *snapshot* refresh (the quarterly base) or PMC refresh yet (see above). New papers
  arrive daily (pipeline/updates); changes to papers already held wait for the next base.
- The citation walk's own local map (`atlas` in citewalk's answer) is dropped by `/walk`;
  positions come from the atlas instead.

## Map engine data (v2) and the "mode" placement rule (2026-10-09, in progress)

**v2 map data** (`pipeline/release/mapdata.py`, run by `release.py build` after the tiles, best effort):
`tiles/<release>/v2/` = density of every paper (256 px tiles, 9 levels, ~830 MB), studies as points
ordered by citations (most cited first), W-numbers per point, and collision-free names (fields,
subfields, OpenAlex topics, titles of famous papers). The site's explorer draws it with WebGL
(`sciencemadereadable/app/web/src/mapgl.ts`). Needs `CORPUS/map-meta/` (citations and topics per
paper, `pipeline/release/corpus_map_meta.py`, built once per corpus). ~18 min, peak 27 GB.
Built by hand for release 20261009T120544Z.

**New placement rule** (`citewalk/science/placement.py`, "mode"): a paper sits in the densest group
of its 30 nearest reference papers, plus a small fixed per-paper offset, instead of the weighted mean
of its 10 nearest. Held-out check (150,000 papers): neighbours kept 0.123 → 0.209; papers stranded
in empty space 17.7 % → 0.1 %; papers piled exactly on a reference 10.4 % → 1.3 %. Same geography
(references and their coordinates), new atlas id: `20261009T190750-tsne-mode`
(`citewalk/science/derive_mode.py`).

**Switched 2026-10-10** (release `20261010T012946Z`, 87.5M searchable, all on the map with the "mode" rule; the
search API on master 194a89d). A question's map spot is now the densest group of its top 20 results
(`atlas.question_spot`): the atlas rule alone put short questions far from their results ("coral reefs": 0 of 20
results within 0.05 map widths; now 18). The site's page itself (WebGL map, new homepage images) is on the dev
site only until `app/tools/publish.sh`.

`release.py build --resume ID` finishes a build that stopped, keeping its finished products. `positions.py` now
flushes each shard to disk before marking it done: a shard written at the minute of the 2026-10-09 reboot was
left as an empty file with a done marker and stopped the first build of this release.

`map.json` lists which tiles exist as one bit per tile and level (`mapdata.py compact()`, 43 KB instead of
627 KB); the site reads both formats. `--only compact` converts an existing build in place.

**Names split (2026-10-10):** `mapdata.py` also writes `v2/names.json` (fields, subfields, topics; ~110 KB
gzipped, loaded at once) and `v2/t/4/x/y.json` (titles of famous studies by place, loaded for the cells on screen
when zoomed in). `labels.json` (8 MB) stays for older pages. `--only split` converts an existing build. The public
door allows both. The site identifies a tapped study with an 8-byte HTTP range of `i/z/x/y.bin`.

**The public tunnel is the bottleneck for the map:** ~1 MB/s from lambda to the relay (lambda's raw upload is
~3.7 MB/s), one connection that answers in order, shared by every visitor. The site keeps at most 6 tile downloads
in flight so taps are not stuck behind them (a tap: 2-3 s while tiles load, 0.1-0.4 s otherwise). Opening the map
from a cold cache: sharp image 2.5-3.6 s (GitHub Pages), live map ~11 s. The structural fix is to serve the
immutable `v2/` files from somewhere with real bandwidth (the relay's own disk, or a CDN), not through the tunnel.

## Paper identity, rescued abstracts, author identity (live since 2026-10-19, release 20261019T095113Z)

Steps 1-3 of `research/audit/RECOMMENDATIONS.md` (measurements: `research/audit/README.md`). All three are derived
per snapshot; `release.py build --identity` uses them and the nightly does so since 2026-10-19 (the API reads them
from the release manifest: `versions`, `rescue`, `identity`). Rebuild all three for a new OpenAlex snapshot.

| Product | Where | Built by | What it changes |
|---|---|---|---|
| paper versions | `/mnt/fast/scholarsreadinglist/identity/<snapshot>/versions/` | `pipeline/identity/build_versions.py` + `versions_lookup.py` (~35 min) | rows.py `--versions`: one row per paper in the semantic index; API: a paper listed once with its `versions` |
| authorships, flat | `.../identity/<snapshot>/authorships/` (1.32B rows, 33 GB) | `pipeline/identity/extract_authorships.py` (~50 min, HDD) | input of author identity |
| author identity | `.../identity/<snapshot>/authors/` (records, moves, eval, scores) | `pipeline/identity/authors.py` (~15 min), `evaluate.py` | API (`SRL_IDENTITY` / manifest `identity`): a person's ORCID records as one page, moved and attached authorships, non-people out of people search |
| PubMed, parsed | `/mnt/bulk/pubmed/` (baseline + updates, 70 GB; `pubmed.parquet`: 41.3M PMIDs, 29.5M abstracts) | `pipeline/pubmed/download.py`, `parse.py` | input of the rescue |
| rescued abstracts | `/mnt/fast/scholarsreadinglist/rescue/<snapshot>/<run>/` (`corpus/` like the updates corpus; `sources.parquet`) | `pipeline/rescue/build.py` | a third corpus (`--corpus`), its vectors in shard groups `rescue-<run>` (`pipeline/rescue/embed.py`, GPU 1, 3 GiB) taken only with rows.py `--extra-groups` |

Rescue shards live in the backfill's shard tree under `year=*/rescue-<run>/`: every other reader globs
`year=*/rg-*/` and does not see them. The rescue corpus *yields* (rows.py `--yielding-corpus`): once the daily updates
hold a work (OpenAlex now has its abstract), the rescued copy steps aside. Only the 2M most cited rescued papers are
embedded so far (`pipeline/rescue/embed.py --limit`); the other 5.8M are searchable by words, not yet by meaning
(~5.5 h of GPU 1 to finish: run embed.py again, then positions.py --shards for the new rescue shards).
Publishing checks: `publish` refuses a release whose searchable count fell > 2 %; one with versions falls ~3.5 % by
design (duplicates removed), so the check compares searchable + versions_dropped (release.py check).
