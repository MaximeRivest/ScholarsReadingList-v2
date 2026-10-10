"""Shared paths and helpers."""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

DATA = Path(os.environ.get("BENCH_DATA", "/mnt/fast/scholarsreadinglist/retrieval-bench"))
OPENALEX_WORKS = Path("/mnt/bulk/openalex/works")
DATASETS = ("scifact", "nfcorpus", "scidocs", "trec-covid")

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_title(title: str | None) -> str:
    """Comparable form of a title: accents stripped, lowercase, alphanumerics only."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return _NON_ALNUM.sub(" ", text).strip()


def abstract_from_inverted_index(raw: str | None) -> str:
    """Rebuild an OpenAlex abstract from its JSON inverted index."""
    if not raw:
        return ""
    index = json.loads(raw)
    if not index:
        return ""
    positions = [(pos, word) for word, poss in index.items() for pos in poss]
    positions.sort()
    return " ".join(word for _, word in positions)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


@contextmanager
def timer(store: dict, key: str):
    start = time.perf_counter()
    yield
    store[key] = time.perf_counter() - start
