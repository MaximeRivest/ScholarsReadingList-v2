# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2", "pyarrow>=21", "sentence-transformers>=5.1", "faiss-cpu==1.15.1"]
# ///
"""Search the downloaded dataset by meaning, on your own machine.

    uv run search.py "coral reef bleaching recovery"                 # from inside the download folder
    uv run search.py "coral reef bleaching recovery" --data ~/openalex-denseon -k 20

Two ways, picked from what you downloaded:

  index/ivf.faiss present   fast search over every paper of the search index (~16 GB of RAM;
                            loading it takes ~20 s, so for many questions load it once in Python):
                            the index proposes --depth candidates, then
                              * with vectors.npy (make_vectors.py) they are re-scored exactly: ~99 % of
                                an exact search's top 10, tens of milliseconds;
                              * without it the index's own approximate scores are used: ~75 %.
                            Rows whose Parquet file you did not download still get their OpenAlex
                            id (index/work_ids.npy), just no title.
  no index                  every row of the Parquet files you downloaded is compared with the question
                            (exact; time grows with the files: one recent year is ~40 GB to read).
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

MODEL = "lightonai/DenseOn"
REVISION = "cb9947ebccb33862d24e3c7ca2edb25e51acd887"  # the exact weights the vectors were made with
SHOW = ["openalex_id", "title", "year", "doi", "type"]


def encode(question: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    return model.encode(question, prompt_name="query", normalize_embeddings=True).astype(np.float32)


class Files:
    """The Parquet files on disk, and which rows each one holds (from the files' own statistics)."""

    def __init__(self, data: Path):
        self.paths, self.starts, self.ends = [], [], []
        for p in sorted(data.glob("data/**/part-*.parquet")):
            md = pq.ParquetFile(p).metadata
            col = md.schema.to_arrow_schema().get_field_index("row")
            lo = md.row_group(0).column(col).statistics.min
            hi = md.row_group(md.num_row_groups - 1).column(col).statistics.max
            self.paths.append(p); self.starts.append(lo); self.ends.append(hi)
        order = np.argsort(self.starts)
        self.paths = [self.paths[i] for i in order]
        self.starts = [self.starts[i] for i in order]
        self.ends = [self.ends[i] for i in order]
        if not self.paths:
            raise SystemExit(f"no data/**/part-*.parquet under {data}")

    def details(self, rows: list[int]) -> dict[int, dict]:
        """Row number -> the displayed columns, reading only the row groups that hold them."""
        out = {}
        for r in rows:
            i = bisect.bisect_right(self.starts, r) - 1
            if i < 0 or r > self.ends[i]:
                continue  # that file was not downloaded
            f = pq.ParquetFile(self.paths[i])
            rg, first = 0, self.starts[i]
            while first + f.metadata.row_group(rg).num_rows <= r:
                first += f.metadata.row_group(rg).num_rows
                rg += 1
            t = f.read_row_group(rg, columns=SHOW)
            out[r] = t.slice(r - first, 1).to_pylist()[0]
        return out

    def scan(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Exact: compare the question with every downloaded row, one row group at a time."""
        best_rows, best_scores = np.empty(0, np.int64), np.empty(0, np.float32)
        for p in self.paths:
            f = pq.ParquetFile(p)
            for rg in range(f.metadata.num_row_groups):
                t = f.read_row_group(rg, columns=["row", "embedding"])
                e = t["embedding"].combine_chunks().values.to_numpy().reshape(-1, q.shape[0])
                s = e.astype(np.float32) @ q
                best_rows = np.concatenate([best_rows, t["row"].to_numpy()])
                best_scores = np.concatenate([best_scores, s])
                top = np.argsort(-best_scores)[:k]
                best_rows, best_scores = best_rows[top], best_scores[top]
        return list(zip(best_rows.tolist(), best_scores.tolist()))


def index_search(data: Path, q: np.ndarray, k: int, depth: int, nprobe: int) -> tuple[list, str]:
    import faiss
    index = faiss.read_index(str(data / "index" / "ivf.faiss"))
    params = faiss.SearchParametersIVF(nprobe=nprobe)
    scores, cand = index.search(q.reshape(1, -1), max(depth, k), params=params)
    cand, scores = cand[0][cand[0] >= 0], scores[0][cand[0] >= 0]
    vec_path = data / "vectors.npy"
    if not vec_path.exists():
        top = np.argsort(-scores)[:k]
        return list(zip(cand[top].tolist(), scores[top].tolist())), "index only (approximate scores)"
    vectors = np.load(vec_path, mmap_mode="r")
    order = np.argsort(cand)  # sorted reads are kinder to the disk
    exact = np.empty(len(cand), np.float32)
    exact[order] = vectors[cand[order]].astype(np.float32) @ q
    top = np.argsort(-exact)[:k]
    return list(zip(cand[top].tolist(), exact[top].tolist())), "index + exact re-scoring"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--data", type=Path, default=Path("."), help="the downloaded dataset folder")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--depth", type=int, default=300, help="index candidates re-scored (with vectors.npy)")
    ap.add_argument("--nprobe", type=int, default=128, help="index lists scanned: more = slower, more complete")
    ap.add_argument("--scan", action="store_true", help="ignore the index; compare every downloaded row")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    q = encode(args.question)
    files = Files(args.data)
    if (args.data / "index" / "ivf.faiss").exists() and not args.scan:
        hits, how = index_search(args.data, q, args.k, args.depth, args.nprobe)
    else:
        hits, how = files.scan(q, args.k), "exact scan of the downloaded files"
    info = files.details([r for r, _ in hits])
    ids_path = args.data / "index" / "work_ids.npy"  # row -> OpenAlex W-number, for rows not downloaded
    wids = np.load(ids_path, mmap_mode="r") if ids_path.exists() else None
    results = []
    for r, s in hits:
        h = {"row": r, "score": round(s, 4), **info.get(r, {})}
        if "openalex_id" not in h and wids is not None:
            h["openalex_id"] = f"W{int(wids[r])}"
        results.append(h)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return
    print(f"{how}, {time.time() - t0:.1f} s\n")
    for i, h in enumerate(results, 1):
        doi = f"  https://doi.org/{h['doi']}" if h.get("doi") else ""
        print(f"{i:2}. {h['score']:.3f}  {h.get('year', '')}  {h.get('title') or '(row not downloaded)'}")
        print(f"            https://openalex.org/{h.get('openalex_id', '')}{doi}")


if __name__ == "__main__":
    main()
