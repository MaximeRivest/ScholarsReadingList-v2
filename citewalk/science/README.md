# Atlas of science — pilot

Open **https://lambda.tail69222b.ts.net:18806/** on the tailnet.

One fixed map of English-language research (OpenAlex works with an abstract), drawn from
DenseOn title+abstract vectors. The year controls only choose which dots are lit; **no
position ever changes with the year**. This is the pilot proposed in Chattering record
01a10666 (2026-10-08): a historically balanced sample, nonlinear projections measured
against the PCA baseline, one frozen projection, a year-sliced viewer.

## What is on screen

- **A dot is one publication**, from a fixed 1 % sample of every year 1800–2026. Brightness
  therefore follows real volume. "Equalize brightness" adapts the exposure so that early years
  stay visible; the timeline always shows the exact per-year corpus counts (`count_types.py`).
- **Publications only** (`scope.py`): article, review, conference paper/abstract, preprint, book,
  chapter, dissertation, report, letter, editorial, data/software paper. In the embedding plan,
  `dataset` and other non-publication records are 36 % of all works and 62–79 % of 2024–2026,
  mostly automatic NIFS instrument records ("LHD FMP #184398.1"). A first build that kept them
  showed a fake explosion and large islands of near-identical records; it was discarded
  (`fit-alltypes/`). The atlas covers 128.0 M of the plan's 201.5 M dated works.
- **Modes:** everything up to a year; papers of that year; last 5 or 10 years (older ones dimmer).
  Default is the last 5 years, so playing shows current activity, not a pile-up of everything.
  Territory and growth views need a finite window and switch "everything" to 5 years.
  The newest year is tinted white. Play animates year by year from 1900.
- **Labels** sit at the densest spot of each OpenAlex field/subfield (computed once over all
  years) and appear only once that label has enough dots in the selected years. Colours are
  OpenAlex's topic labels; papers without one are grey.
- **Click a dot:** details, its 10 most similar papers by the original vectors (gold rings),
  and how many of those are no farther on the map than its 10th nearest dot, next to the map's
  average (1.6 of 10; about 4 of 10 within the 100 nearest dots).
- Label rule: a field label shows once it holds ≥ 0.5 % of the dots shown (subfields ≥ 0.12 %),
  so a few classifier hits (e.g. "Computer Science" in the 1920s) only give a faint label.
- **Search:** the text is encoded with DenseOn's `query:` prompt, placed by the frozen map
  function, and its 20 most similar dots are listed.

## Seeing macro shifts (option A: fixed ground, moving activity)

All computed in the browser from the same frozen positions; no position ever moves.

- **Show: growth vs earlier.** For every place on screen, its share of all research in the selected
  years against the same window 5, 10 or 20 years earlier (grey bars on the timeline). Shares, not
  counts, because science grows ~5 %/year overall. Smoothing widens when dots are sparse (about 40
  dots per neighbourhood). A place is coloured only if the difference is statistically clear
  (Poisson |z| > 2, full colour at 4), so 1 % sampling noise stays grey. Red = growing share, blue =
  shrinking, saturating at 4× and ¼×.
- **Continuous time.** Time is a real number, not a year counter. Each dot gets its own moment
  inside its publication year (dots are stored in random order within a year, so moments are
  spread evenly); it fades in over ~5 months after that moment and fades out as it leaves the
  window. This is a display device: many OpenAlex dates are only a year. Exposure, label
  visibility and the timeline playhead follow the same continuous counts. Territory and growth
  maps are computed per whole year in a Web Worker (`static/layers-worker.js`, same code as
  `static/layers.js`) and cross-faded between consecutive years; field labels glide between
  positions (or cross-fade if their territory moved far). Arrow keys and timeline clicks glide to
  the new year instead of jumping. Check: stepping one year in tenths, the picture changes by an
  even amount at every step (largest step ≤ 1.1 × the average, for dots, territories and growth).
  The growth map now uses a fixed 160×160 world grid (was per-screen), so it is the same at any
  zoom and can be precomputed; close up it is coarser than before.
- **Largest subfields** (bar race, right; territory and growth views): the 16 subfields with the
  largest share of all publications in the selected years (total includes unlabelled papers).
  Bars are in field colours and scaled to the current leader, as in a bar chart race; dashed
  gridlines every 1 % move with the scale so years stay comparable. A chip shows the change in
  share vs the comparison period (estimate shrunk towards "no change", prior 10 dots, hidden below
  40 dots; "=" within ±15 %). Measured: 0.56 subfields enter the top 16 per year (1960–2026), vs
  2.2 for the earlier growth race. Leaders: History / Economics (1900s–1940s, few papers),
  Molecular Biology (1970s–2000s), Electrical Engineering (2010s), Sociology and Political Science
  (2020s). The leader's share fell from ~7 % to ~3.5 %: research is spread over more subfields.
  Rows persist and slide/stretch (0.45 s per year when the chart is on screen). Each subfield has
  a hand-picked emoji (`static/icons.js`); hover marks its home on the map, click flies there.
  An earlier version ranked by growth instead; it was replaced because rankings by growth change
  much faster than by size.
- **Field territories** (default view, last 5 years). Every place on a fixed 256×256 grid takes
  the colour of the field with the most papers there in the selected years, like countries.
  A place is coloured firmly only if the winner's lead over the runner-up is beyond sampling luck
  (z ≥ 2.5); contested places are barely tinted, places with too few papers are empty. Field names
  sit at the densest point of each field's largest current territory, so they follow it.
  Checks: splitting the dots into two random halves gives 0 % disagreement on firm places
  (14–26 % before the significance rule, which is why that rule exists). Between decades, firm
  places almost never change owner (< 0.1 %): **fields grow, shrink and appear; they do not
  migrate.** Share of firm territory, 1970 → 1990 → 2010 → 2026: Computer Science 0 → 3 → 8.5 → 7 %,
  Environmental Science 0 → 2 → 3 → 4 %, Medicine 35 → 36 → 28 → 28 %, Chemistry 9.5 → 4.5 → 2.4 → 0 %
  (its places become contested, mostly with materials science and biochemistry). Firm area also
  depends on how many papers there are, so early years show less of it.
- **Field drift** (centre-of-mass trails, first version) was removed: each field's centre moved
  only ~3 px per year on screen while its papers spread ~110–150 px around it, so trails looked
  static and only the arrow heads were noticeable.
- Caveats: OpenAlex's subfield labels come from a recent classifier applied to old papers too,
  so a ×27 jump (e.g. Health Informatics) partly reflects how the classifier and the vocabulary
  of abstracts changed, not only research effort. The record caveats above (abstract coverage,
  indexing) apply to every comparison across decades.

## Citation walk on the atlas

Start a walk by typing an OpenAlex ID (`W…`) or a DOI in the search box, or with **Citation walk
from this paper** in a dot's panel. The walk is the citewalk algorithm unchanged (`citewalk/server.py`,
full citation snapshot, all years): X, the papers it cites, the 30 papers sharing the most references
with it, what those cite that X does not, and the red answer. It is drawn on top of the atlas; nothing
on the map moves.

- **Positions:** an atlas dot keeps its exact position; a fitted reference paper its fitted position;
  any other paper with a DenseOn vector (backfill or the atlas sample) is placed by the atlas's own
  frozen function (`placement.py`). Papers with no vector are listed in the panel, never guessed.
  So a paper is always at the same place, whichever walk it appears in (`citewalk/atlas_overlay.py`,
  `citewalk/test_atlas_overlay.py`).
- **Lines:** X's references (faint white) and the citations into the answer (red) by default; hover or
  click a marker for all of its lines; "show every citation line" draws all of them.
- **Walking:** click a marker, then "Walk here". The trail at the top of the panel steps back. The walk is
  in the address (`&wk=W…`), so a link or reload reopens it.
- **Years:** the walk is computed on all years; the year slider only dims walk papers published later.
  A walk restricted to what was known in a given year is not built.
- The atlas server forwards `/api/walk` to the citewalk server on port 8798, which must be running.
- Browser check: `check_walk_browser.py` (real clicks, positions, walking, reload, phone).

## How it was made

| Step | Script | Result |
|---|---|---|
| Sample | `sample.py` | Works with a fixed hash `u = splitmix64(W-number)`. Display = `u < 1%` every year. Extra papers for 1880–1979 up to 3,000 per year shape the geography only. 2,231,809 works. |
| Vectors | `embed_sample.py` | 1,161,000 copied from the backfill; 1,070,809 older ones encoded with the backfill's own encoder and manifest (same model revision, prompt, 512-token cap, BF16, normalization, float16). Check on 2,000 works both sources have: identical input text for all, cosine median 0.99995, minimum 0.998. |
| Types | `count_types.py`, `scope.py` | Exact eligible counts per year and type (201,527,004, equal to the plan). |
| Fit + measure | `fit.py`, `fit_more.py` | Publications only. 291,556 references (per-year quota ∝ √volume) and 22,500 held-out (1,500 per decade). PCA, UMAP, t-SNE compared. |
| Build | `build.py` | t-SNE layout of the references, frozen. Every other paper placed by `placement.py`. |
| Viewer | `server.py`, `static/` | WebGL2, density accumulated in a float buffer, then tone-mapped. |

**Frozen map function** (`placement.py`): a paper lands at the softmax(similarity/0.01)-weighted
mean of the map positions of its 10 most similar references (exact cosine). It depends only on
the paper's own vector and the frozen references, so it applies to any of the 203.5 M backfill
papers, and positions never depend on what else is shown. References keep their fitted positions.

## Measured quality (publications the map was not fitted on)

291,556 references, 22,500 held-out publications (1,500 per decade, 1880s–2020s).

| Method | 10-NN kept | true 10-NN within 100 nearest dots | same subfield | global rank corr. |
|---|---|---|---|---|
| PCA (current citewalk approach) | 0.6 % | 2.5 % | 3.6 % | 0.41 |
| UMAP + placement | 8.8 % | 26.4 % | 13.3 % | 0.34 |
| **t-SNE + placement (chosen)** | **16.3 %** | **40.1 %** | **18.3 %** | **0.35** |

- Subfield agreement of the true 768-D neighbours is 24.9 %; that is the ceiling for that column.
- The libraries' own out-of-sample transforms were worse than the simple placement
  (t-SNE 3.8 %, UMAP 3.0 %), which is why the placement function is used.
- t-SNE 10-NN kept per decade: 13–23 % from the 1880s to the 2020s. Older decades are not
  mapped worse (the √-volume reference quota does its job).
- Placement sweep (first, all-types fit; t-SNE, K × τ): sharper weighting helps (K=3, τ=0.01
  reached 18.5 % vs 12.5 % at the initial τ=0.03); K=10, τ=0.01 was kept for smoother placement.
- **Frozen map and new science:** t-SNE refitted with no 2020+ references, then 2020s papers
  placed on it: 12.2 % 10-NN kept (vs 13.9 % with the 2020s in the fit); 29.5 % vs 34.5 % within
  100 nearest. New years land on a frozen map with about one seventh less precision.
- **Does a new field show up?** Share of dots within 0.012 of where a query lands, per year:
  - COVID-19: 0.3–0.4 ‰ (2015–2019) → 9.7 ‰ (2020), 10.6 ‰ (2021), fading to 2.3 ‰ (2026). Very clear.
  - Large language models: 0.3–0.4 ‰ (2015–2022) → 1.0 (2023), 1.8 (2024), 3.0 ‰ (2025). Clear.
  - Deep learning: ~1 ‰ until 2016 → 2.9 ‰ (2024). Gradual.
  - CRISPR, HIV: no visible signal. Topics that grow *inside* an existing area (its text is
    close to earlier work) do not form new territory on this map; a 1 % sample also thins them.

These scores compare neighbours within the 314k evaluation pool, not the full corpus.

## Limits to keep in mind

- English works with an abstract only. Older papers lack abstracts far more often, and OpenAlex
  coverage changed over time, so part of any "growth" is better records, not more science.
- **Retrospective map:** a 1950 paper is placed by how today's model reads it, on a geography
  fitted with knowledge of later research.
- A label appearing is not the birth of a field: labels are a classifier's output, thresholds
  are fixed counts of sampled dots.
- 2026 is a partial year. Dates are the OpenAlex publication dates used by the backfill plan.
- No citation overlay yet (citations after the selected year would need to be excluded).
- The vector job ran on GPU 0 (0.9 GB, about 55 min) because the `qwen3-embed-4b` service had
  filled GPU 1; it was capped and refused to start without headroom.

## Files

```
/mnt/fast/science-atlas/
  sample/year=*.parquet, summary.json    the sample, eligibility counts, rates
  vectors/work.npy, vectors.npy, source.npy, check.json, chunks/
  sample/type_counts.json                  exact eligible counts per year and type
  fit/R.npy, E.npy, tsne.npz, umap.npz, report.json, preview-*.png
  fit-alltypes/                            the discarded first fit (all record types)
  build/<id>/ points.f32 year.u16 field.u8 subfield.u16 meta.json info.parquet vectors.npy reference.npz
```

## Run

```sh
cd citewalk/science
uv run python -m unittest test_science -v
CHROMIUM_BIN=/nix/store/33pxss8h71cl7vmfpy21bidsw0lj1g8q-chromium-152.0.7977.82/bin/chromium \
  uv run --with playwright==1.55.0 --with pillow python check_browser.py --base http://127.0.0.1:8806/

# transient service (stops at reboot), uses the pipeline/embeddings environment for search
cd ../../pipeline/embeddings
systemd-run --user --unit=science-atlas --working-directory=$PWD -E HF_HUB_OFFLINE=1 \
  $PWD/.venv/bin/python ../../citewalk/science/server.py --port 8806
tailscale serve --bg --https=18806 http://127.0.0.1:8806
```
