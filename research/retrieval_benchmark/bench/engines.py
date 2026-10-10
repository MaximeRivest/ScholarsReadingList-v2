"""Approximate multi-vector indexes: FastPLAID and WARP (xtr-warp-rs)."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ENGINES = ("fastplaid", "warp")


def _doc_tensors(docs: np.ndarray, doclens: np.ndarray) -> list[torch.Tensor]:
    offsets = np.concatenate([[0], np.cumsum(doclens, dtype=np.int64)])
    flat = torch.from_numpy(np.ascontiguousarray(docs))
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(doclens))]


def du_bytes(path: Path) -> int:
    return int(subprocess.check_output(["du", "-sb", str(path)]).split()[0])


def build(engine: str, path: Path, docs: np.ndarray, doclens: np.ndarray, device: str,
          nbits: int = 4) -> dict:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = _doc_tensors(docs, doclens)
    t = time.perf_counter()
    if engine == "fastplaid":
        from fast_plaid import search
        index = search.FastPlaid(index=str(path), device=device)
        index.create(documents_embeddings=tensors, nbits=nbits, batch_size=10_000, seed=42)
    elif engine == "warp":
        from xtr_warp import XTRWarp
        XTRWarp(index=str(path)).create(embeddings_source=tensors, device=device, nbits=nbits,
                                        seed=42, show_progress=False)
    else:
        raise ValueError(engine)
    return {"build_seconds": time.perf_counter() - t, "index_bytes": du_bytes(path),
            "build_device": device, "nbits": nbits}


class Searcher:
    """Loads an index on CPU and answers one query at a time."""

    def __init__(self, engine: str, path: Path, threads: int):
        self.engine, self.threads = engine, threads
        if engine == "fastplaid":
            from fast_plaid import search
            self.index = search.FastPlaid(index=str(path), device="cpu")
        else:
            from xtr_warp import XTRWarp
            self.index = XTRWarp(index=str(path)).load(device="cpu", mmap=True)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = torch.from_numpy(np.ascontiguousarray(query)).float().unsqueeze(0)
        if self.engine == "fastplaid":
            return self.index.search(queries_embeddings=q, top_k=k, show_progress=False)[0]
        return self.index.search(queries_embeddings=q, top_k=k, num_threads=self.threads)[0]
