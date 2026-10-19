"""Where the round-3 field classifiers (0.8B, 2B) are weaker than Sol, Jev and Qwen: the numbers behind NEXT_ROUNDS.md.

CPU only. Reads saved results, never runs a model:
  - ~/Projects/jevarticle/results/per_article.csv          the 200-article expert test, every method's pick
  - /mnt/fast/scholarsreadinglist/fieldjudge/v3/evaluation/ round-3 acceptance runs (rows + per-option probabilities)
  - /mnt/fast/scholarsreadinglist/fieldjudge/v3/training/   what the models were trained on

    cd search-dev/experiments/fieldjudge/v3 && ../../../.venv/bin/python ~/Projects/scholarsreadinglist/research/fieldjudge/weak_spots.py
(run from v3 so eval3.build() can rebuild the row tags: the saved rows carry no tags)
"""
import collections, csv, json, random, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, ".")
import eval3 as E  # noqa: E402  (rebuilds the evaluation rows with their tags; no GPU)

JA = Path.home() / "Projects/jevarticle"
EV = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v3/evaluation")
OURS, O8 = "srl3-2b-abs", "srl3-08-abs"
OTHERS = ["flat-abs", "sol61-abs", "qwen-direct-abs", "qwen-think-abs"]


def section(t):
    print(f"\n{'=' * 8} {t}")


def expert_test():
    rows = list(csv.DictReader(open(JA / "results/per_article.csv")))
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["run"]][r["scopus_id"]] = r
    tax = {t["subfield"]: t["field"] for t in csv.DictReader(open(JA / "data/taxonomy.csv"))}
    ids = list(by[OURS])

    def acc(run, f=lambda r: True):
        xs = [int(by[run][i]["match_mode"]) for i in ids if f(by[run][i])]
        return (sum(xs) / len(xs) if xs else float("nan")), len(xs)

    section("expert test: accuracy by the experts' field (n>=5): 2B / 0.8B / Jev / Sol / Qwen direct")
    fields = collections.Counter(tax.get(by[OURS][i]["expert_modes"].split("|")[0].strip(), "?") for i in ids)
    for fl, n in fields.most_common():
        if n < 5:
            continue
        f = lambda r, fl=fl: tax.get(r["expert_modes"].split("|")[0].strip(), "?") == fl
        print(f"{fl:42s} n={n:3d}", *[f"{acc(x, f)[0] * 100:5.0f}" for x in [OURS, O8] + OTHERS[:3]])

    section("expert test: errors that cross a field boundary (same field / different field)")
    for run in [OURS, O8] + OTHERS:
        same = cross = 0
        for i in ids:
            r = by[run][i]
            if r["match_mode"] == "0":
                gf = {tax.get(g.strip()) for g in r["expert_modes"].split("|")}
                same, cross = (same + 1, cross) if tax.get(r["prediction"]) in gf else (same, cross + 1)
        print(f"{run:18s} same-field {same:3d}  cross-field {cross:3d}")

    section("expert test: 'General ...' subfields (gold n=8): how often each method picks one, and hits")
    for run in [OURS, O8] + OTHERS:
        gold_gen = [i for i in ids if any(g.strip().startswith("General") for g in by[run][i]["expert_modes"].split("|"))]
        pred_gen = [i for i in ids if by[run][i]["prediction"].startswith("General")]
        print(f"{run:18s} picks General {len(pred_gen)}  hits {sum(by[run][i]['match_mode'] == '1' for i in gold_gen)}/{len(gold_gen)}")

    section("expert test: accuracy by confidence bin")
    for run in [OURS, O8, "flat-abs", "sol61-abs"]:
        bins = collections.defaultdict(list)
        for i in ids:
            try:
                bins[min(int(float(by[run][i]["confidence"]) * 5), 4)].append(int(by[run][i]["match_mode"]))
            except ValueError:
                pass
        print(run, {f"{b / 5:.1f}-{(b + 1) / 5:.1f}": (f"{np.mean(v) * 100:.0f}%", len(v)) for b, v in sorted(bins.items())})

    section("expert test: title only vs with abstract")
    for ha in "01":
        f = lambda r, ha=ha: r["has_abstract"] == ha
        print("abstract" if ha == "1" else "title only", *[f"{x}:{acc(x, f)[0] * 100:.0f}" for x in [OURS, O8, "flat-abs", "sol61-abs", "qwen-think-abs"]])

    section("expert test: 2B wrong while Sol AND Jev right")
    for i in ids:
        a, s, j, e = by[OURS][i], by["sol61-abs"][i], by["flat-abs"][i], by[O8][i]
        if a["match_mode"] == "0" and s["match_mode"] == "1" and j["match_mode"] == "1":
            print(f"- {a['title'][:88]} [{a['subset'][:4]}, abs={a['has_abstract']}]\n    experts={a['expert_modes']} | 2B={a['prediction']} ({a['confidence']}) | 0.8B={e['prediction']}")


def round3_runs():
    rows = E.build(False)
    S = {n: json.load(open(EV / f"{n}-scores.json")) for n in ["round2-0.8B", "v3-0.8B", "v3-2B"]}
    idx = collections.defaultdict(list)
    for k, r in enumerate(rows):
        idx[r["tag"][0]].append(k)
    subs, fields, field_of = E.subs, E.fields, E.field_of
    votes = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(JA / "data/gold_standard_long.csv")):
        votes[r["scopus_id"]][r["evaluator"]].append(r["subfield"])

    def modes(xs):
        c = collections.Counter(xs); m = max(c.values()); return {s for s, n in c.items() if n == m}

    gold = [modes([v[0] for v in votes[a["scopus_id"]].values()]) for a in E.arts]
    gold_f = [modes([field_of[v[0]] for v in votes[a["scopus_id"]].values()]) for a in E.arts]

    for n in ["v3-0.8B", "v3-2B"]:
        P = S[n]
        flat = {rows[k]["tag"][1]: np.array(P[k]) for k in idx["flat"]}
        hits = [subs[int(flat[i].argmax())] in gold[i] for i in range(200)]
        section(f"{n}: option-order flips by confidence (2 shuffles x 200 articles)")
        shuf = collections.defaultdict(list)
        for k in idx["shuffled"]:
            q = np.zeros(176); q[np.array(rows[k]["tag"][2])] = np.array(P[k]); shuf[rows[k]["tag"][1]].append(q)
        fl = np.array([(flat[i].max(), int(q.argmax() != flat[i].argmax()), hits[i], subs[int(q.argmax())] in gold[i])
                       for i in range(200) for q in shuf[i]], float)
        for lo, hi in [(0, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            m = (fl[:, 0] >= lo) & (fl[:, 0] < hi)
            print(f"confidence {lo}-{hi}: n={m.sum():3.0f} flips {fl[m, 1].mean():.2f}  accuracy unshuffled {fl[m, 2].mean():.2f} shuffled {fl[m, 3].mean():.2f}")
        section(f"{n}: the 22-field question, confusions")
        fp = {rows[k]["tag"][1]: np.array(P[k]) for k in idx["field"]}
        conf = collections.Counter()
        for i in range(200):
            pf = fields[int(fp[i].argmax())]
            if pf not in gold_f[i]:
                conf[(tuple(sorted(gold_f[i])), pf)] += 1
        for (g, p), c in conf.most_common(6):
            print(f"  {c:3d} {g} -> {p}")
        print("field accuracy: asked directly", np.mean([fields[int(fp[i].argmax())] in gold_f[i] for i in range(200)]).round(3),
              " read off the subfield pick", np.mean([field_of[subs[int(flat[i].argmax())]] in gold_f[i] for i in range(200)]).round(3))

    section("yes/no, 8 held-out topics: average precision / AUC  (round2-0.8B, v3-0.8B, v3-2B)")
    per = collections.defaultdict(list)
    for k in idx["yesno"]:
        per[rows[k]["tag"][1]].append(k)
    for topic, ks in per.items():
        y = np.array([rows[k]["tag"][2] for k in ks]); w = np.array([rows[k]["tag"][3] for k in ks])
        out = []
        for n in ["round2-0.8B", "v3-0.8B", "v3-2B"]:
            p = np.array([S[n][k][0] for k in ks], float)
            out.append(f"{average_precision_score(y, p, sample_weight=w):.3f}/{roc_auc_score(y, p, sample_weight=w):.3f}")
        print(f"{topic[:50]:50s}", *out)
    section("yes/no: the 2B's most confident mistakes on its weakest topics (reference labels, not human)")
    P, P8 = S["v3-2B"], S["v3-0.8B"]
    for topic in ["difference-in-differences with staggered treatment", "medieval manuscript stemmatics",
                  "sulfide solid electrolytes for lithium batteries"]:
        ks = per[topic]
        print(f"-- {topic}")
        for k in sorted([k for k in ks if rows[k]["tag"][2] == 0], key=lambda k: -P[k][0])[:4]:
            print(f"   says yes p={P[k][0]:.2f} (0.8B {P8[k][0]:.2f}) | {rows[k]['user'][7:110]}")
        for k in sorted([k for k in ks if rows[k]["tag"][2] == 1], key=lambda k: P[k][0])[:4]:
            print(f"   says no  p={P[k][0]:.2f} (0.8B {P8[k][0]:.2f}) | {rows[k]['user'][7:110]}")

    section("unseen schemes vs the teachers: agreement by slice (v3-2B)")
    P = S["v3-2B"]
    recs = []
    for k in idx["unseen"]:
        r, p = rows[k], np.array(P[k])
        for role, d in r["teachers"].items():
            t = np.array([d.get(x, 0.0) for x in r["nodes"]])
            if t.sum() == 0:
                continue
            t /= t.sum()
            recs.append(dict(group=r["tag"][1], unit=r["tag"][2], n=len(r["answers"]), hit=int(p.argmax() == t.argmax()),
                             tconf=t.max(), title_only="Abstract:" not in r["user"].split("## Classification")[0]))

    def agg(key):
        g = collections.defaultdict(list)
        for x in recs:
            g[key(x)].append(x["hit"])
        return {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(g.items(), key=lambda z: str(z[0]))}

    print("by group:", agg(lambda x: x["group"]))
    print("by options:", agg(lambda x: min(b for b in [5, 10, 20, 40, 80, 200, 600] if x["n"] <= b)))
    print("by teacher's own confidence:", agg(lambda x: min(b for b in [.5, .7, .9, 1.0] if x["tconf"] <= b)))
    print("title only:", agg(lambda x: x["title_only"]))


def training_data():
    section("round-3 training data")
    n = collections.Counter(); tconf = []; nopt = []; none_off = 0; ys = []; broad = 0
    with open("/mnt/fast/scholarsreadinglist/fieldjudge/v3/training/train.jsonl") as fh:
        for line in fh:
            r = json.loads(line); n[r["kind"]] += 1
            if r["kind"] == "yesno":
                ys.append(float(r["y"])); continue
            t = np.array(list(r["target"].values()) if isinstance(r["target"], dict) else r["target"], float)
            tconf.append(t.max() / t.sum()); nopt.append(len(r["options"]))
            none_off += any("None of these" in o["name"] for o in r["options"])
            broad += "field" in r["question"].lower() or "domain" in r["question"].lower()
    tconf, nopt = np.array(tconf), np.array(nopt)
    print(dict(n), "teacher sure (100% on one option):", (tconf == 1).mean().round(3), " options<=3:", (nopt <= 3).sum(),
          " options>100:", (nopt > 100).sum(), " 'None of these' offered:", none_off, " broad-level questions:", broad,
          " yes/no share of yes:", round(float(np.mean(ys)), 3), " soft yes/no labels:", sum(0 < y < 1 for y in ys))


if __name__ == "__main__":
    expert_test()
    round3_runs()
    training_data()
