# Publishing the embeddings on Hugging Face

The vectors of a data release, with enough metadata to use them alone, as a public dataset:
<https://huggingface.co/datasets/maximerivest/openalex-denseon-embeddings>.

```sh
cd ~/Projects/scholarsreadinglist
OUT=/mnt/bulk/hf-export/openalex-denseon

uv run pipeline/hf/export.py --out $OUT          # release -> data/<year>/part-NNNNN.parquet (~45 min, resumable)
uv run pipeline/hf/verify.py --out $OUT          # random rows of every file == the shards and the OpenAlex records
uv run pipeline/hf/card.py --out $OUT --repo maximerivest/openalex-denseon-embeddings   # README, scripts, index copy
pipeline/hf/upload.sh start                      # resumable upload, capped at 70 Mbit/s; status | stop | uncap
```

| File | What it does |
|---|---|
| `export.py` | Two stages: per year, joins every vector to its OpenAlex record and checks text hashes (`work/meta/`); then writes the files in parallel. `row` = the release's row number (the search index's ids). |
| `verify.py` | Every row exactly once; sampled vectors bit-identical to the shards; sampled text hashes recomputed from the records. |
| `card.py` | Renders `repo/README.template.md` with the release's numbers, copies `search.py`, `make_vectors.py` and the release's FAISS index + `work_ids.npy` into the folder. |
| `upload.sh` | `hf upload` as the `hf-upload` user service. The cap marks that service's packets (iptables cgroup match) and limits them with a tc class, so the public site keeps the rest of the uplink. |
| `repo/` | What users get besides the data: the card template, `search.py` (index or exact scan), `make_vectors.py` (one memory-mappable `vectors.npy`). |

Next version: run the same commands with the newer release. Rows are append-only, so existing files
keep their rows; Xet storage only uploads chunks that changed. Tag each upload with its release id
(`hf repos tag create ... --repo-type dataset`).

Not in the dataset, on purpose: abstracts (publishers' text; users fetch them from OpenAlex by id)
and the junk model's scores (they belong to the search service and change with it).
