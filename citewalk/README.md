# Citation walk — clean semantic-map test

Open **https://lambda.tail69222b.ts.net:18798/** on the tailnet.

The default page is now a flat diagnostic map, replacing the city/river drawing.
The citation recommendation calculation is unchanged.

## What the new view does

- Fits two PCA directions **once** to the genuine vectors available around the
  starting paper. Up to 256 deterministic anchors are used for very large views.
- Saves that projection and applies it to later papers without refitting.
- Saves each title-group's first vector/coordinates. A different representative
  record or newly completed embedding cannot move an existing point.
- Keeps the camera fixed while walking. **Fit current papers** changes only the
  camera; **Start new map** explicitly creates a new local projection.
- Keeps starting/visited landmarks faintly visible, with an optional toggle.
- Shows equal-size points, without collision pushing, cluster attraction, clipping
  distant coordinates, anisotropic stretching, fake terrain, or citation forces.
- Shows actual supplied citation links only around the selected/hovered paper.
  Solid arrows go to references; dashed arrows arrive from citing papers. These
  are connections in the current analysis, not a complete citation neighborhood.
- Lists missing/invalid vectors separately. They remain in the citation result;
  no centroid, citation-neighbor, or other synthetic position is assigned.
- Offers the Papers list for keyboard/touch selection and choosing unplaced papers.
  Clicking overlapping points offers a choice rather than moving them apart.
- Shows a five-neighbor agreement check against cosine neighbors in the original
  vectors, plus optional exact-neighbor rings and similarity values on selection.

The year control is a **display filter only**; it does not recompute historical
citation rankings. Unknown years remain visible. Coordinates do not depend on
this filter, citation counts, or visual role.

## What this test does NOT establish

This is a **frozen local PCA chart**, not a global atlas of research. PCA provides
an interpretable baseline with a fixed out-of-sample transform, but can lose much
of the local neighborhood structure. The first three example maps retained only
about 29–45% of their original five-nearest-neighbor relationships. They captured
about 19–27% of anchor variance in two dimensions. Those measurements are exposed
rather than hidden behind attractive geography.

The check compares neighbors within the current evaluated result (up to 512
points), not against the whole corpus. Exact cosine similarity is a property of
the model vectors, not independent evidence of scientific relevance.

A reliable larger atlas is still future work: choose a broader fixed reference
sample, evaluate neighborhood-preserving projections, and version the resulting
geography. Do not assume completion of the embedding backfill alone fixes a poor
projection. The app also does not repair bad abstracts or title-based identity
mistakes in the underlying citation model.

## Persistence and source integrity

Saved maps live under:

```text
/mnt/fast/citewalk/<citation-generation>/local-maps/<atlas-id>/
  projection.npz    # fixed mean, axes, anchor IDs and metadata; no pickle
  points.sqlite    # genuine normalized vectors and immutable first placements
```

The URL includes `atlas=<id>` so reloads reuse the same projection. The map is
pinned to a citation generation and embedding plan. A different plan/generation
requires an explicit new map. Camera state is kept per map in browser session
storage. Source embeddings are read-only; the backfill is not changed.

Only published `part-*` shards are read, not the writer's `.part-*.tmp-*`
directories. Plan IDs, dimensions, declared row counts, and duplicate work IDs are
checked before an index refresh is installed. A bounded, locked memory-map cache
replaces the old unsynchronized cache. Full vector-file checksums are not reread
on every lookup; the reader relies on the producer's committed, immutable shards.

The server picks up additional committed shards every ten minutes. Old points
keep their saved coordinates; new points can be added to the same axes. Source
vectors come from the existing English title+abstract DenseOn backfill and its
pinned input snapshot, which may differ from the newest displayed metadata.

## Main files

| File | Purpose |
|---|---|
| `server.py` | Existing citation analysis/API and read-only embedding lookup |
| `atlas.py` | Frozen projection, persistence, honest missingness and diagnostics |
| `static/atlas.html` | Current default page |
| `static/atlas.js` | Flat canvas, fixed camera/coordinates, selection and arrows |
| `static/atlas_app.js` | API calls, navigation, details, quality and missing-paper lists |
| `test_citewalk.py` | Citation/outlier/duplicate-grouping reference tests |
| `test_atlas.py` | Coordinate invariance, persistence, refresh and missingness tests |
| `check_atlas_browser.py` | Actual click/pan/zoom/walk/reload and phone-layout checks |

`static/index.html`, `map.js`, and `iso.js` retain the earlier experiments; they
are not the default page. The retired per-query projection functions in
`server.py` are not called by the HTTP analysis route.

## Test and run

```sh
OPENBLAS_NUM_THREADS=1 uv run --with-requirements citewalk/requirements.txt \
  python -m unittest discover -s citewalk -p 'test_*.py' -v

# Use an isolated standalone Chromium, never the shared user profile.
CHROMIUM_BIN=/path/to/standalone/chromium uv run --with playwright==1.55.0 \
  python citewalk/check_atlas_browser.py --base http://127.0.0.1:8798/
```

The browser check uses fresh profiles and writes screenshots/results under
`/tmp/citewalk-atlas-browser/`. It tests actual pointer selection, panning, wheel
zoom, walking, coordinate invariance, reload, filtering, the Papers list, and the
explicit new-map action. No screenshots alone are treated as interaction tests.

The app remains a transient user service:

```sh
systemd-run --user --unit=citewalk --working-directory=$HOME/Projects/scholarsreadinglist \
  $(which uv) run --with-requirements citewalk/requirements.txt python citewalk/server.py --port 8798
tailscale serve --bg --https=18798 http://127.0.0.1:8798
```

The saved map files survive a restart; the transient service itself stops at
reboot. No OS configuration or embedding worker policy was changed.

## Existing citation computation and limits

For paper X, find papers sharing its references, keep the top k (default 30),
then tally what those papers cite that X does not. Outliers default to more than
300 listed references or more than 500 received citations. X can be an outlier
because it is the chosen starting point. These rules and the title-based grouping
heuristic are unchanged by the clean map.

- Identical normalized titles of at least 20 characters are grouped by the
  existing algorithm. This can merge distinct publications or miss renamed ones.
- The current algorithm may sum citation counts across such grouped records;
  that is not a deduplicated count of citing publications.
- The index contains only citations between works present in its snapshot;
  citations to absent targets are excluded.
- Titles are capped at 600 bytes in the citation index. DOI resolution uses a
  64-bit hash and is not a full DOI-string equality index.
- The citation index is about 80 GB, memory-mapped. Startup also builds the
  committed-embedding ID lookup, which currently takes roughly twenty seconds.
- Build a new citation/topic index after publishing a new tidy generation using
  `build_index.py` and `build_topics.py`. Old saved maps remain tied to their
  original citation generation.
