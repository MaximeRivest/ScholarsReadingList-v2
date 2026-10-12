---
license: {license}
pretty_name: OpenAlex paper embeddings (DenseOn, English titles and abstracts)
language:
- en
size_categories:
- 100M<n<1B
task_categories:
- feature-extraction
- sentence-similarity
- text-retrieval
tags:
- openalex
- scientific-papers
- scholarly
- embeddings
- semantic-search
- dense-retrieval
- science-map
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - data/*/*.parquet
    - data/updates/*/*.parquet
---

# OpenAlex paper embeddings: {rows_m} million English titles + abstracts, ready to search

One row per work in [OpenAlex](https://openalex.org) that has an English abstract:
its OpenAlex id, DOI, title, date, type, venue, field, a 2D position on a map of science, and a
768-number **embedding** of its title and abstract made with
[lightonai/DenseOn](https://huggingface.co/lightonai/DenseOn). Papers about the same thing get
nearby vectors, so you can search by meaning, find similar papers, cluster, or map all of science.

These are the vectors behind the search on [sciencemadereadable.com](https://sciencemadereadable.com)
and its [free search API for people and AI agents](https://api.sciencemadereadable.com/api/data/docs).

| | |
|---|---|
| Rows (embedded works) | **{rows:,}** |
| …of which in the search index | {searchable:,} publications, one per distinct text ([`in_search_index`](#columns)) |
| Size | {gb:.0f} GB in {files} Parquet files ({file_gb:.2f} GB each at most), split by publication year |
| Text embedded | `title + " " + abstract`, English only, first 512 tokens |
| Vector | 768 × float16, L2-normalised (cosine similarity = dot product) |
| Metadata as of | OpenAlex snapshot {snapshot} + daily OpenAlex API additions to {updates_until} |
| Data release | `{release}` |

## Pick your path

| You have | Do this |
|---|---|
| Nothing to download | Use the hosted API: `curl https://ask.sciencemadereadable.com/coral+reef+recovery` ([docs](https://api.sciencemadereadable.com/api/data/docs)) |
| A laptop | Query the metadata remotely with DuckDB (reads only the columns you ask for), or download one year |
| ~20 GB of RAM, ~16 GB of disk | Download `index/` and search **all {searchable_m} million** papers by meaning ([below](#search-everything-on-your-own-machine)) |
| ~350 GB of SSD | Download everything: exact re-scoring, your own index, clustering, maps |

### Look without downloading (DuckDB)

Parquet lets DuckDB fetch only the columns and row groups a query needs, straight from the Hub:

```sql
-- duckdb
SELECT year, count(*) AS papers
FROM 'hf://datasets/{repo}/data/**/*.parquet'
WHERE year BETWEEN 2015 AND 2026
GROUP BY year ORDER BY year;
```

### Download part of it

Files are grouped by publication year (`data/2025/`, `data/1999/`, …, plus `data/before-1900/`,
`data/after-2026/`, `data/no-year/` and the daily additions in `data/updates/`):

```sh
pip install -U huggingface_hub
hf download {repo} --repo-type dataset --include "data/2025/*" --local-dir openalex-denseon
hf download {repo} --repo-type dataset --local-dir openalex-denseon          # everything, ~{gb:.0f} GB
```

### Use the vectors

```python
import numpy as np, pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

t = pq.read_table("openalex-denseon/data/2025/part-00000.parquet")
E = t["embedding"].combine_chunks().values.to_numpy().reshape(-1, 768)   # float16, one row per paper

model = SentenceTransformer("lightonai/DenseOn", revision="{revision}")
q = model.encode("coral reef recovery after bleaching", prompt_name="query", normalize_embeddings=True)
scores = E.astype(np.float32) @ q
for i in np.argsort(-scores)[:5]:
    print(f"{{scores[i]:.3f}}", t["title"][i], "https://openalex.org/" + str(t["openalex_id"][i]))
```

Questions are encoded with `prompt_name="query"` (`"query: "` in front), papers with
`prompt_name="document"` (`"document: "`). To put **your own texts** in the same space, encode
`title + " " + abstract` with the document prompt: re-encoding papers on a CPU in float32 gives a
cosine ≥ 0.999 with the stored vectors (we checked).

With 🤗 Datasets, stream rather than load: a plain `load_dataset` converts all {gb:.0f} GB into its
cache first.

```python
from datasets import load_dataset
ds = load_dataset("{repo}", split="train", streaming=True)
print(next(iter(ds))["title"])
```

### Search everything on your own machine

`index/` holds the search index the live site uses for this release: FAISS `{index_factory}`
over the {searchable:,} rows with `in_search_index = true` ({index_gb:.1f} GB), and
`index/work_ids.npy` (row number → OpenAlex W-number, {ids_gb:.1f} GB). Its ids are the `row` column.

```sh
hf download {repo} --repo-type dataset --include "index/*" --include "*.py" --local-dir openalex-denseon
cd openalex-denseon
uv run search.py "coral reef recovery after bleaching"          # needs ~20 GB of RAM
```

| What you downloaded | Quality of the top 10 | Time per question |
|---|---|---|
| `index/` only | ~{recall_index:.0%} of an exact search's top 10 (approximate scores) | ~5–30 ms, after a ~20 s load |
| `index/` + all of `data/` + `uv run make_vectors.py` (builds a {vectors_gb:.0f} GB `vectors.npy`; use an SSD) | ~{recall_rerank:.0%} (300 candidates re-scored exactly) | ~10–50 ms |
| `data/` for some years, no index | exact, over those years | ~1 minute per recent year from an SSD (each is 15–40 GB) |

(Measured on 200 paper-vector questions against a deeper exact re-scoring, release `{release}`;
not a relevance judgement. The index was written by `faiss-cpu==1.15.1`, which `search.py` pins.)

The live site ranks a little differently: it also demotes likely junk records (spam, paratext),
which this dataset does not include.

## Columns

| Column | Type | What it is |
|---|---|---|
| `row` | int64 | Row number, 0 … {rows_minus_1:,}. The index's ids. New versions add rows at the end ([Versions](#versions)) |
| `openalex_id` | string | `W…` id; `https://openalex.org/<id>` is the record, `https://api.openalex.org/works/<id>` its full metadata |
| `doi` | string | DOI without `https://doi.org/`, null if none |
| `title` | string | Title as embedded |
| `publication_date`, `year` | date, int16 | From OpenAlex. A few are wrong in OpenAlex (years like 1007 or 2050) |
| `type` | string | OpenAlex type: `article`, `book-chapter`, `preprint`, `review`, `paratext`, `erratum`, … |
| `venue` | string | Journal, conference or repository |
| `field`, `subfield`, `topic` | string | OpenAlex primary topic and its subfield and field |
| `cited_by_count` | int32 | Citations counted by OpenAlex at that time |
| `is_retracted`, `is_open_access`, `license` | bool, bool, string | From OpenAlex (`license` is the work's licence, e.g. `cc-by`) |
| `in_search_index` | bool | In the search index: a publication type (article, review, preprint, book, chapter, conference paper, thesis, report, letter, editorial, …) and the smallest W-number among works with byte-identical title + abstract |
| `atlas_x`, `atlas_y` | float32 | Position (0–1) on the [map of science](https://sciencemadereadable.com): openTSNE layout of a reference sample, every paper placed from its 10 nearest reference papers. Atlas `{atlas}`. Projections distort distances: use it to look, not to measure |
| `text_hash` | binary(16) | BLAKE2b-128 of the UTF-8 text embedded; equal hash = identical text |
| `embedding` | float16[768] | The vector |

Every Parquet file also carries the model, revision, prompts and input format in its metadata.

## How it was made

- **Selection.** All OpenAlex works whose language is `en` and whose abstract is not empty, with no
  filter on type, open access or completeness ({rows:,} rows: {base:,} from the OpenAlex snapshot
  of {snapshot}, {updates:,} new papers from the daily API run since).
- **Text.** `((title or "") + " " + abstract).strip()`, abstract rebuilt from OpenAlex's inverted
  index (words put back in order). DenseOn reads the first 512 tokens.
- **Model.** `lightonai/DenseOn` at revision `{revision}` (149M parameters, Apache-2.0),
  sentence-transformers {st_version}, bfloat16 on an RTX 3090, `document: ` prompt, then L2-normalised
  in float32 and stored as float16.
- **Checks.** For every year folder, the title + abstract of a full shard of rows was hashed again
  from the OpenAlex records and compared with the hash stored next to each vector: all matched.
  Every row was joined to exactly one OpenAlex record.

## Limits, said plainly

- **Not all of science.** Only works OpenAlex labels English *and* gives an abstract. Many papers
  (older ones, many books, much non-English research) have no abstract in OpenAlex, and its language
  labels make mistakes in both directions.
- **Not only papers.** {non_pub:,} rows ({non_pub_share:.0%}) are not publications by the definition
  above ({top_non_pub}, …), and {dups:,} publications repeat another record's exact title and abstract.
  `in_search_index` removes both; `type` and `text_hash` let you choose differently.
- **Metadata ages.** Titles, citation counts, topics and retractions are as of {snapshot} (daily
  additions: as of the day they were fetched). OpenAlex re-ran its topic classifier after that
  snapshot, so recent additions carry newer topic labels than older rows.
- **No abstracts here.** Abstracts belong to their publishers, and this dataset does not redistribute
  them. Get them from OpenAlex (free snapshot or API) by `openalex_id`, or through our
  [API](https://api.sciencemadereadable.com/api/data/docs).
- **Search quality** of the vectors is DenseOn's (BEIR nDCG@10 56.2 per its authors). We have not
  published a relevance evaluation of this release.

## Versions

New versions add files for new papers and may refresh metadata columns; an existing row keeps its
`row` number and vector. A change that would renumber rows or change vectors (a new model, a new
way to build the text) will be a separate dataset, not a silent update. Each version is a tagged
revision of this repository named after its data release; `export.json` describes it.

## Licence

{license_text}

The metadata comes from [OpenAlex](https://openalex.org) (CC0). DenseOn is Apache-2.0.

## Cite

```bibtex
@misc{{rivest_openalex_denseon_{year},
  author = {{Rivest, Maxime}},
  title  = {{OpenAlex paper embeddings (DenseOn, English titles and abstracts)}},
  year   = {{{year}}},
  publisher = {{Hugging Face}},
  url    = {{https://huggingface.co/datasets/{repo}}}
}}
```

Also cite OpenAlex (Priem, Piwowar & Orr 2022, [arXiv:2205.01833](https://arxiv.org/abs/2205.01833)) and DenseOn.
Made by [ScienceMadeReadable](https://sciencemadereadable.com); code: [srl-search](https://github.com/MaximeRivest/srl-search),
[pipeline](https://github.com/MaximeRivest/ScholarsReadingList-v2/tree/main/pipeline).
