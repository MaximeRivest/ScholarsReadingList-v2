# Title + abstract retrieval benchmark

> **October 5 update:** See [compression-results.md](compression-results.md) for the
> compression experiment and corrected storage accounting. The first speed sweep
> stopped early; its low-memory timings are invalid. WARP's released 1-bit indexes
> cannot be searched. A local mmap-loader fix and its validation are documented in
> [patches/README.md](patches/README.md); revised results are kept separately.

Compares search models and engines for Scholars Reading List, using titles and
abstracts only (no full text).

- Code: this folder. Data, vectors, indexes, results: `/mnt/fast/scholarsreadinglist/retrieval-bench/`.
- Test sets: BEIR SciFact, NFCorpus, SciDocs, TREC-COVID (MD5-checked downloads).
- Scale test: those four corpora + 1,000,000 random OpenAlex works with abstracts
  (main collection, deduplicated by title against the test sets) = 1,205,228 papers.
- Metric: nDCG@10 (checked against `pytrec_eval` in `tests/`).

## Results (2026-10-04)

### Exact search, test set alone

| Model | SciFact | NFCorpus | SciDocs | TREC-COVID |
|---|---:|---:|---:|---:|
| BM25 keywords | 0.686 | 0.321 | 0.158 | 0.623 |
| DenseOn (1 vector) | 0.760 | **0.391** | **0.223** | 0.822 |
| LateOn | 0.761 | 0.381 | 0.218 | **0.833** |
| LateOn, 512-token docs | **0.765** | 0.382 | 0.218 | 0.833 |
| mLateOn | 0.764 | 0.378 | 0.204 | 0.829 |
| mxbai-edge-colbert 32M | 0.745 | 0.361 | 0.165 | 0.767 |

LateOn reproduces its published scores (0.766 / 0.378 / 0.219 / 0.836).

### 1.2M papers: exact vs WARP index (4 CPU cores)

| Method | SciFact | NFCorpus | SciDocs | TREC-COVID | Search p50 / p95 |
|---|---:|---:|---:|---:|---|
| BM25 | 0.511 | 0.120 | 0.086 | 0.618 | 5–45 / ~60 ms (1 core) |
| DenseOn exact | 0.603 | **0.172** | **0.141** | **0.805** | not timed (no index yet) |
| LateOn exact | **0.612** | 0.164 | 0.139 | 0.802 | — |
| LateOn + WARP | 0.555 | 0.134 | 0.123 | 0.628 | 26–62 / 50–215 ms |
| mxbai 32M exact | 0.591 | 0.151 | 0.090 | 0.726 | — |
| mxbai 32M + WARP | 0.558 | 0.145 | 0.092 | 0.680 | 35–87 / 90–160 ms |

OpenAlex papers are unjudged and count as wrong. The change from "alone" to
"1.2M" cannot establish the true quality change or a bound on it. WARP ran with
its automatic settings.

Total build directories (WARP, 4-bit): LateOn 25.7 GB per million papers, mxbai
16.6 GB. These include duplicate build chunks: read-only serving files are about
half that size. See the compression report for measured serving-file sizes.
Query encoding on 4 cores (PyTorch, float32): LateOn 55 ms, DenseOn 50 ms, mxbai 12 ms.

FastPLAID on 4 CPU cores: 0.35–1.6 s per query already at 5,000 papers
(`probes/fastplaid_cpu.py`), so it was dropped from the scale test.

### DenseOn speed at 1.2M papers (4 CPU cores, `probes/dense_cpu.py`)

| Index | SciFact | NFCorpus | SciDocs | TREC-COVID | Search p50 / p95 |
|---|---:|---:|---:|---:|---|
| Flat (exact) | 0.603 | 0.173 | 0.141 | 0.805 | ~147 / ~165 ms |
| HNSW M=32, ef=64 | 0.590 | 0.161 | 0.131 | 0.808 | 0.35–0.56 / 0.5–0.8 ms |
| HNSW M=32, ef=128 | 0.601 | 0.169 | 0.135 | 0.806 | 0.65–1.0 / 1.0–1.4 ms |
| HNSW M=32, ef=256 | 0.604 | 0.171 | 0.138 | 0.804 | 1.0–1.8 / 1.4–2.4 ms |

HNSW build: 573 s on 32 threads. It holds full float32 vectors in RAM (~3.4 GB
per million papers); quantized variants were not tested.

## Known issues

- DenseOn must be encoded in bfloat16 or float32; float16 overflows (NaN vectors).
- NixOS: set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` for index builds.
- `gpu1_services.sh stop|start` pauses/resumes the family services on GPU 1.

## Next

1. Tune WARP (more probes) to recover TREC-COVID quality; measure the speed cost.
2. Test quantized DenseOn indexes (8-bit HNSW, IVF-PQ) for RAM at 100M+ scale; faster query encoding (ONNX int8).
3. Judge a sample of the OpenAlex papers that reach the top 10, to measure the true drop.
