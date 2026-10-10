# Real-data example: content, signatures, clusters, links, metrics

**Historical example — superseded design.** The current agreement is in
[SCHEMA.md](../../../pipeline/SCHEMA.md): signatures are author–affiliation
pairs, unit tables are `content`, `signatures`, `entities`, `classifications`,
and link tables start with `links_`. This example still uses author-entry
signatures, venue signatures, and the old cluster/membership names. The new
pair-based example is available as ordinary files in
[tidy-row-example](../tidy-row-example/EXAMPLE.md), generated and tested by
`pipeline/tidy/`. See [PLAN.md](../../../pipeline/PLAN.md) for next steps.

Open `index.html` in the artifact panel. Use the cluster selector to follow
membership through to the distinct papers and their metrics. IDs in tables
link to the corresponding rows. Each table has a CSV download.

## Rebuild

From the repository root:

```sh
uv run python research/openalex/schema-example/build.py
```

The script reads only the two records in
`research/openalex/schema-review-examples.parquet`. It does not scan the full
corpus, call an external API, or change production tables.

## Modeling choices to evaluate

- Four separate core tables: content, signatures, signature_clusters,
  content_clusters. No universal entity table.
- One author signature per author entry: raw name plus contextual affiliation
  text. One such signature may be assigned to both a person and institutions.
- Institution memberships preserve the exact source field paths supporting
  the match. Unmatched affiliations are not silently assigned.
- Venue attribution is demonstrated as a second signature kind, resolving to
  a venue in signature_clusters. Only the primary location is included.
- Both membership tables have identical columns, but different member types.
- One common metrics schema works over distinct content reached by either path.
- Joined affiliation text is a display field, not a reversible list encoding.
  Exact original fields remain in the fixture. Choosing a production scalar
  encoding is intentionally outside this example.

## Scope and calculations

All metrics are limited to these two articles. Incoming citation counts (71
and 39) are values in the saved source records, not counts inferred from the
248 exported outgoing references. Neither article cites the other in this
fixture.

For UCLA, four signatures lead to just two distinct papers. The citation sum
is therefore 110, not 252. Mean citations is 55 and h-index is 2. These are
example-only statistics, not UCLA's or any author's complete profile.

Only the primary topic and two shared keywords are used as content clusters.
The original records have additional classifications and metadata. Funding
is not represented in this example; no raw funding statement was fabricated.

## Validation

`build.py` checks scalar cells, foreign keys, membership uniqueness, shared
membership schemas, exact example sizes, and expected deduplicated metrics.
The source mappings are imported OpenAlex assertions; confidence is left
unknown where the source does not supply it. Topic and keyword scores are
source scores, not probabilities of author or institution identity.
