import random

import pytest
import pytrec_eval

from bench.metrics import evaluate, ndcg_at_k, recall_at_k


def test_hand_computed_ndcg():
    # Relevant docs: a (grade 2), b (grade 1). Ranking: x, a, b.
    scores = {"x": 3.0, "a": 2.0, "b": 1.0}
    judged = {"a": 2, "b": 1, "x": 0}
    import math
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(scores, judged, 10) == pytest.approx(dcg / idcg)
    assert recall_at_k(scores, judged, 1) == 0
    assert recall_at_k(scores, judged, 2) == 0.5


def test_missing_query_counts_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": {"a": 1.0}}
    assert evaluate(run, qrels)["ndcg@10"] == pytest.approx(0.5)


def test_matches_pytrec_eval_on_random_runs():
    rng = random.Random(0)
    docs = [f"d{i}" for i in range(300)]
    qrels, run = {}, {}
    for q in range(40):
        qid = f"q{q}"
        qrels[qid] = {d: rng.choice([0, 1, 1, 2]) for d in rng.sample(docs, 25)}
        # Include deliberate score ties to exercise tie-breaking.
        run[qid] = {d: float(rng.randint(0, 30)) for d in rng.sample(docs, 200)}
    ours = evaluate(run, qrels)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10", "recall.100"})
    ref = ev.evaluate(run)
    for ours_key, ref_key in (("ndcg@10", "ndcg_cut_10"), ("recall@10", "recall_10"),
                              ("recall@100", "recall_100")):
        expected = sum(r[ref_key] for r in ref.values()) / len(ref)
        assert ours[ours_key] == pytest.approx(expected, abs=1e-9), ours_key
