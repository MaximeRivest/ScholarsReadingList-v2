"""Download and load BEIR scientific test sets (corpus = title + abstract)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from .common import DATA, DATASETS

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
MD5 = {  # Published in the BEIR README.
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "scidocs": "38121350fc3a4d2f48850f6aff52e4a9",
    "trec-covid": "ce62140cb23feb9becf6270d0d1fe6d1",
}
ROOT = DATA / "beir"


def download(name: str) -> Path:
    folder = ROOT / name
    if (folder / "corpus.jsonl").exists():
        return folder
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / f"{name}.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(URL.format(name=name), tmp)
        tmp.replace(archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5[name]:
        raise RuntimeError(f"{name}: md5 {digest} != published {MD5[name]}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(ROOT)
    return folder


def load_qrels(name: str, split: str = "test") -> dict[str, dict[str, int]]:
    """Judgments only (no corpus): qid -> {doc_id: grade}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(download(name) / "qrels" / f"{split}.tsv") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for qid, did, grade in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(grade)
    return qrels


def load(name: str, split: str = "test"):
    """Return (docs, queries, qrels).

    docs: list of (doc_id, title, text_for_models) where text = f"{title} {body}",
    the same formatting PyLate uses for BEIR.
    queries: dict qid -> text, restricted to queries that have judgments.
    qrels: dict qid -> {doc_id: grade}; only grades > 0 count as relevant.
    """
    folder = download(name)
    docs = []
    with open(folder / "corpus.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            title = d.get("title") or ""
            text = f"{title} {d.get('text') or ''}".strip()
            docs.append((str(d["_id"]), title, text))
    qrels: dict[str, dict[str, int]] = {}
    with open(folder / "qrels" / f"{split}.tsv") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for qid, did, grade in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(grade)
    queries = {}
    with open(folder / "queries.jsonl") as fh:
        for line in fh:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    missing = set(qrels) - set(queries)
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} judged queries have no text")
    return docs, queries, qrels


if __name__ == "__main__":
    for n in sys.argv[1:] or DATASETS:
        docs, queries, qrels = load(n)
        rel = sum(1 for q in qrels.values() for g in q.values() if g > 0)
        print(f"{n}: {len(docs):,} docs, {len(queries):,} queries, {rel:,} relevant judgments")
