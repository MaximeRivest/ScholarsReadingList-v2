# Status and next steps

## Done (October 6, 2026)

1. Row contracts and a plain-Python reference extractor (schema v3).
2. Native Polars extraction, identical to the reference on every test.
3. Metadata coverage: core content and publisher-page fields, signatures,
   people, institutions, funders, publishers, topics and their hierarchy,
   keywords, sources, MeSH (descriptors, supplementary concepts, headings),
   SDGs, awards, citations.
4. One current version per work across files; one row per entity,
   classification and vocabulary edge by a documented order. Separate
   extraction plus reduction is proven equal to one extraction.
5. Resumable runner: content-addressed jobs, RAM-guarded concurrency, atomic
   generation publication, gc. End-to-end and incremental-update tests.
6. Calibration on real files; full run started as `openalex-tidy.service`.

Details: [README.md](README.md) (operation), [SCHEMA.md](SCHEMA.md) (model),
`research/openalex/tidy-pilot/` (earlier pilots).

## Next

- **Inspect the first generation**: row counts, note counts (invalid IDs,
  malformed abstracts, MeSH oddities), a sample of papers end to end.
- **Shortcuts and metrics**: `links_content_entities` derived from signatures
  (method `derived`), then counts, citation sums, h-index per entity and
  classification over distinct content, with explicit filters and time windows.
- **Retire the old flat content table** once the new one is accepted.
- Optional: organization lineage (`links_entities_entities`), deprecated
  concepts, official author/institution registries for fuller profiles.

## Known limits

- Peak RAM per job grows with file density; the runner learns it, but a file
  far denser than any seen could still need splitting (not implemented).
- Any change to a transform file invalidates every extraction job (by design:
  correctness over reuse). Changing only `run.py`, `dataset.py` or docs does not.
- Calibrated throughput depends on other users' load on lambda.
