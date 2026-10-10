"""Encode documents and queries once per (model, corpus); cache to NVMe.

Layout under DATA/emb/<model>/<corpus>/:
  late:  docs.f16 (all token vectors, concatenated), doclens.npy, queries.npy (padded),
         qlens.npy, meta.json
  dense: docs.npy (N x dim, L2-normalised float16), queries.npy, meta.json
Document order is the order given by the caller (saved as doc_ids.json).
"""
from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch

from .common import DATA, write_json
from .models import MODELS, Model

CHUNK = 20_000


def cache_dir(model: str, corpus: str):
    return DATA / "emb" / model / corpus


def load_model(m: Model, device: str):
    # Late-interaction models: float16 (verified NaN-free, scores match published ones).
    # DenseOn overflows float16 on ~10% of documents (NaN vectors), so dense models use
    # bfloat16, which has float32's range.
    dtype = torch.float16 if m.kind == "late" else torch.bfloat16
    kwargs = {"model_kwargs": {"dtype": dtype}} if device.startswith("cuda") else {}
    if m.kind == "late":
        from pylate import models
        if m.document_length:
            kwargs["document_length"] = m.document_length
        return models.ColBERT(model_name_or_path=m.hf_id, device=device, **kwargs)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(m.hf_id, device=device, **kwargs)


def _encode_late(model, texts, is_query, batch_size):
    out = model.encode(texts, is_query=is_query, batch_size=batch_size,
                       show_progress_bar=False, convert_to_numpy=True)
    embs = [np.asarray(e, dtype=np.float16) for e in out]
    for e in embs:
        _check_finite(e)
    return embs


def _encode_dense(model, texts, is_query, batch_size):
    out = model.encode(texts, prompt_name="query" if is_query else "document",
                       batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    return _check_finite(np.asarray(out, dtype=np.float16))


def _check_finite(a: np.ndarray) -> np.ndarray:
    if not np.isfinite(a).all():
        raise FloatingPointError("encoder produced NaN/inf vectors; use a wider dtype")
    return a


def truncated_fraction(model, texts) -> float | None:
    """Approximate share of documents longer than the model's document length."""
    limit = getattr(model, "document_length", None)
    if not limit:
        limit = getattr(model, "max_seq_length", None)
    if not limit:
        return None
    sample = texts[:: max(1, len(texts) // 5000)]
    lengths = [len(x) for x in model.tokenizer(sample, add_special_tokens=False)["input_ids"]]
    # +3: document marker and the two special tokens.
    return float(np.mean([n + 3 > limit for n in lengths]))


def encode(model_slug: str, corpus: str, doc_ids: list[str], doc_texts: list[str],
           query_ids: list[str], query_texts: list[str], device: str = "cuda:1") -> dict:
    folder = cache_dir(model_slug, corpus)
    meta_path = folder / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["n_docs"] == len(doc_ids) and meta["n_queries"] == len(query_ids):
            return meta
    m = MODELS[model_slug]
    if m.pooled_from:
        raise ValueError(f"{model_slug} is derived by bench.compress, not encoded")
    folder.mkdir(parents=True, exist_ok=True)
    model = load_model(m, device)
    t0 = time.perf_counter()
    meta = {"model": m.hf_id, "document_length": getattr(model, "document_length", None), "corpus": corpus, "n_docs": len(doc_ids),
            "n_queries": len(query_ids), "device": device,
            "truncated_fraction": truncated_fraction(model, doc_texts)}
    if m.kind == "late":
        lens = []
        with open(folder / "docs.f16.part", "wb") as fh:
            for i in range(0, len(doc_texts), CHUNK):
                embs = _encode_late(model, doc_texts[i:i + CHUNK], False, m.batch_size)
                for e in embs:
                    fh.write(e.tobytes())
                    lens.append(len(e))
                print(f"  {model_slug}/{corpus}: {i + len(embs):,}/{len(doc_texts):,} docs", flush=True)
        dim = embs[0].shape[1]
        (folder / "docs.f16.part").replace(folder / "docs.f16")
        np.save(folder / "doclens.npy", np.asarray(lens, dtype=np.int32))
        meta["doc_seconds"] = time.perf_counter() - t0
        qs = _encode_late(model, query_texts, True, m.batch_size)
        qmax = max(len(q) for q in qs)
        padded = np.zeros((len(qs), qmax, dim), dtype=np.float16)  # zero rows add exactly 0 to MaxSim
        for i, q in enumerate(qs):
            padded[i, : len(q)] = q
        np.save(folder / "queries.npy", padded)
        np.save(folder / "qlens.npy", np.asarray([len(q) for q in qs], dtype=np.int32))
        meta |= {"dim": int(dim), "n_vectors": int(sum(lens)),
                 "mean_vectors_per_doc": float(np.mean(lens))}
    else:
        docs = np.concatenate([_encode_dense(model, doc_texts[i:i + CHUNK], False, m.batch_size)
                               for i in range(0, len(doc_texts), CHUNK)])
        meta["dtype"] = "bfloat16" if device.startswith("cuda") else "float32"
        meta["doc_seconds"] = time.perf_counter() - t0
        np.save(folder / "docs.npy", docs)
        np.save(folder / "queries.npy", _encode_dense(model, query_texts, True, m.batch_size))
        meta |= {"dim": int(docs.shape[1])}
    meta["docs_per_second_gpu"] = len(doc_ids) / meta["doc_seconds"]
    write_json(folder / "doc_ids.json", doc_ids)
    write_json(folder / "query_ids.json", query_ids)
    write_json(meta_path, meta)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def load_late(model_slug: str, corpus: str):
    folder = cache_dir(model_slug, corpus)
    meta = json.loads((folder / "meta.json").read_text())
    docs = np.memmap(folder / "docs.f16", dtype=np.float16, mode="r").reshape(-1, meta["dim"])
    return (docs, np.load(folder / "doclens.npy"), np.load(folder / "queries.npy"),
            np.load(folder / "qlens.npy"), json.loads((folder / "doc_ids.json").read_text()),
            json.loads((folder / "query_ids.json").read_text()), meta)


def load_dense(model_slug: str, corpus: str):
    folder = cache_dir(model_slug, corpus)
    return (np.load(folder / "docs.npy"), np.load(folder / "queries.npy"),
            json.loads((folder / "doc_ids.json").read_text()),
            json.loads((folder / "query_ids.json").read_text()),
            json.loads((folder / "meta.json").read_text()))
