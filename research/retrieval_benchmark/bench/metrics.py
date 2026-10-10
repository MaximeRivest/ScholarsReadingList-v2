"""Retrieval metrics with the same conventions as BEIR (pytrec_eval).

- nDCG@k uses linear gain (gain = relevance grade) and log2(rank + 1) discount;
  the ideal ranking is built from *all* judged documents with grade > 0.
- Recall@k counts documents with grade > 0.
- Ties in score are broken by document id (descending), like trec_eval, so results
  are deterministic. Unjudged retrieved documents count as non-relevant.
"""
from __future__ import annotations

import math

Run = dict[str, dict[str, float]]      # qid -> {doc_id: score}
Qrels = dict[str, dict[str, int]]      # qid -> {doc_id: grade}


def _ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], _neg_key(x[0])))]


def _neg_key(doc_id: str):
    # trec_eval breaks ties by docno in descending lexicographic order.
    return tuple(-ord(c) for c in doc_id)


def ndcg_at_k(scores: dict[str, float], judged: dict[str, int], k: int) -> float:
    ideal = sorted((g for g in judged.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    ranked = _ranked(scores)[:k]
    dcg = sum(max(judged.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ranked))
    return dcg / idcg


def recall_at_k(scores: dict[str, float], judged: dict[str, int], k: int) -> float:
    relevant = {d for d, g in judged.items() if g > 0}
    if not relevant:
        return 0.0
    ranked = _ranked(scores)[:k]
    return len(relevant.intersection(ranked)) / len(relevant)


def evaluate(run: Run, qrels: Qrels) -> dict[str, float]:
    """Mean metrics over judged queries that have at least one relevant document.

    A judged query with no results scores 0 (it is not dropped), so a method cannot
    look better by failing to answer.
    """
    qids = [q for q, j in qrels.items() if any(g > 0 for g in j.values())]
    out = {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "recall@1000": 0.0}
    for q in qids:
        scores = run.get(q, {})
        out["ndcg@10"] += ndcg_at_k(scores, qrels[q], 10)
        for k in (10, 100, 1000):
            out[f"recall@{k}"] += recall_at_k(scores, qrels[q], k)
    return {m: v / len(qids) for m, v in out.items()} | {"queries": len(qids)}
