"""Paths and rules shared by the daily update steps (standard library only).

The daily updates extend one *base*: the OpenAlex snapshot that the big corpus, the embedding backfill and the
citation index were built from. Everything daily lives under UPDATES / <base id>, so a new base (the next
quarterly OpenAlex release) starts a fresh, empty set of updates and never mixes with the old one.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------- the base these updates extend
BASE_ID = "openalex-2026-09-23"
BASE_SNAPSHOT_DATE = "2026-09-23"                       # manifest date of the public OpenAlex snapshot
BASE_CORPUS = Path("/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05")
BASE_EMBEDDINGS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts")   # manifest.json, shards/
# where daily shards are written (tests point this elsewhere; production: beside the backfill's own shards)
SHARDS_OUT = Path(os.environ.get("SRL_UPDATES_EMBEDDINGS", BASE_EMBEDDINGS))
RAW_SCHEMA_SOURCE = Path("/mnt/bulk/openalex/works/updated_date=2026-09-23/part_0000.parquet")

UPDATES = Path(os.environ.get("SRL_UPDATES", "/mnt/fast/scholarsreadinglist/updates")) / BASE_ID
CONTENT = UPDATES / "content"            # <run>-<n>.parquet: new works, content table columns (openalex_content.py)
RAW = UPDATES / "raw"                    # <run>-<n>.jsonl.gz: the same works as the API sent them
FACTS = UPDATES / "facts"                # <run>-<n>.parquet: per work, the record facts the search's junk model reads
CORPUS = UPDATES / "corpus"              # <run>/: parts, works.sqlite, keywords, map-meta (build_corpus layout)
STATE = UPDATES / "state.json"           # per (date, field) cell: the API count already fetched
LOCK = UPDATES.parent / ".lock"          # held while corpus/shards change and while a release reads them

# The selection, identical to the backfill's (embed_content.py, policy has_english_abstract) restricted to the
# publication types that releases keep (citewalk/science/scope.py): datasets etc. would only be dropped later.
SCHOLARLY_TYPES = ("article", "review", "conference-paper", "conference-abstract", "preprint", "book",
                   "book-chapter", "dissertation", "report", "letter", "editorial", "data-paper", "software-paper")

# Shard directories of daily updates: year=Y/rg-u<run>/part-<date>-<start>; the "rg-*" and "part-*" globs used
# by every consumer of the backfill's shards (rows.py, positions.py, vectors.py, citewalk) match them as is.
SHARD_GROUP_PREFIX = "rg-u"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


@contextmanager
def locked(what: str, wait_s: float = 4 * 3600):
    """The updates lock: corpus swaps and shard commits on one side, a release reading them on the other."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as f:
        t = time.time()
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t > wait_s:
                    raise TimeoutError(f"{LOCK} held for over {wait_s:.0f}s; {what} gave up")
                time.sleep(10)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def latest_corpus() -> Path | None:
    """The newest finished updates corpus (DONE.json written last), or None."""
    done = sorted(p.parent for p in CORPUS.glob("*/DONE.json"))
    return done[-1] if done else None


def content_files() -> list[Path]:
    return sorted(CONTENT.glob("*.parquet"))
