# Field classifier (0.8B / 2B): where it is weaker than Sol, Jev and Qwen, and the next training rounds

Written 2026-10-18, from the Comparison conversation (Chattering 01a14577), PrecRecall (01a13c29), Scimeetr
(01a13514), the READMEs in `search-dev/experiments/fieldjudge/` and the saved evaluation files. The numbers in
part 2 come from `weak_spots.py` beside this file (CPU only; output in `weak_spots.out`). Nothing here touched
the round-4 run (GPU 0 and 1, `fj4-train-08` / `fj4-train-2b`) or its files.

## 1. How the models were trained so far

| Round | Starting point | Data | Format | Result |
|---|---|---|---|---|
| 1 (2026-10-15) | Qwen3.5-0.8B / 2B base | 436 topics × 90 papers, Kimi yes/no ($47) | yes/no, per-topic question | AUC 0.966 / 0.976 on 40 unseen topics; as good as per-topic deciders, better at classics; too lenient at 0.5 |
| 2 (2026-10-15/16) | base again | 85k topic–paper pairs, Sol + Opus (25 per request, split, not double-judged), 27% yes | yes/no, hard labels | stricter: precision 43→64%, recall 87→63% on 8 new held-out topics; ranking unchanged (AP 0.68 / 0.75); went live (0.8B) |
| 3 (2026-10-17/18) | round-2 models, LoRA r=32 + answer-token output rows, 1 epoch, lr 1e-4 | 46,179 pick questions (Sol/Opus, 1–3 options with shares, 321 schemes: 7 real + 28 of our maps + 284 invented) + 40,000 round-2 yes/no replayed | paper first, then list with two-letter codes; one-token answer, no generation | expert test 64/53% (0.8B), 64/63% (2B); yes/no AP 0.73 / 0.80; unseen schemes 78–87% vs teachers (ceiling 90–92%) |
| 4 (running now) | round-3 models | same data as round 3 | **list first, then paper** (prefix reuse, ~3–5× faster); parent label dropped 30% of the time | not finished |

Teachers' own score on the expert test: Sol 66/69%, Opus 65/69%. Jev 64/61% flat, 68/65% field-then-subfield.
Qwen 27B 57/64% direct, 64/67% thinking. One human expert 63/57%. (specialist / general journals, 100 articles each.)

Two things decided along the way that matter for what comes next:
- Round 2's lesson: the mix decides the model's *style* more than its judgment (27% yes made a strict model;
  2.5× more data did not improve ranking). Round 3's soft targets from the teachers were the first calibrated signal.
- In the rewriter project Maxime found LoRA on the 0.8B "no longer learning enough" and moved to full-weight
  training (conversation 01a0f6b0, 2026-10-05). The field judge has only ever used LoRA r=32.

## 2. Where our models are weaker: the evidence

Every test we have is either 200 articles (±10 points) or labels from a model, not a human. The gains proposed in
part 3 are 2–6 points each: **the current tests cannot see them**. That is why proposal A comes first.

### 2a. Pick-from-list, the expert test (Science-Metrix, 176 subfields, never trained on)

**Who beats us where.** Against the experts' majority pick:

| | 2B | 0.8B | Jev | Sol | Qwen think |
|---|---:|---:|---:|---:|---:|
| Specialist journals | 62 | 64 | 64 | 66 | 64 |
| General journals | 62 | **53** | 61 | 69 | 67 |
| Title only (41 articles) | 59 | **41** | 49 | **71** | **71** |
| Clinical Medicine (n=52) | 56 | 58 | 60 | 65 | — |
| Physics & Astronomy (n=29) | 69 | 66 | 79 | 79 | — |
| Enabling & Strategic Tech. (n=21) | 57 | **33** | 43 | 62 | — |
| Engineering (n=7) | **14** | 57 | 71 | 71 | — |

Head to head on the same articles: the 2B is wrong where Sol is right on 28 articles, and right where Sol is wrong on 17.

**The kind of mistake we make, not just how many.**

1. **We cross field boundaries more.** Of the 2B's 76 errors, 47 land in a different *field* (0.8B: 57 of 83);
   Jev 39 of 75, Sol 39 of 65. Our models' errors are less "a neighbouring subfield" and more "the wrong science".
2. **The object studied beats the discipline.** The cleanest examples where both Sol and Jev are right and we are wrong:
   - *Measuring milk fat content by random laser emission* → we say Dairy & Animal Science; experts: Optics.
   - *A giant thunderstorm on Saturn* → Meteorology; experts: Astronomy & Astrophysics.
   - *Prevalence of Toxoplasma gondii in Canadian market-age pigs* → Veterinary Sciences; experts: Mycology & Parasitology.
   - *Genomic variants revealed by missing genotypes in Nelore cattle* → Dairy & Animal Science; experts: Genetics & Heredity.
   - *Paper-based CRP monitoring devices* → Analytical Chemistry (0.85 confident); experts: Biomedical Engineering.
   - *Hypoxia-induced modulation of apoptosis and BCL-2 in cancer cells* → Biochemistry; experts: Oncology.
   The teachers' labels on real schemes (OpenAlex, MeSH…) often *do* follow the object; Science-Metrix's experts
   follow the discipline of the journal/method. The model has not learned to read which convention a scheme uses
   from its description. (All 14 such cases: `weak_spots.out`.)
3. **"General …" categories are avoided.** 8 articles have a "General Physics / General Chemistry / General &
   Internal Medicine" majority. The 2B picks a General category twice (hits 2), the 0.8B never (hits 1); Sol picks
   one 6 times (hits 4), Qwen-thinking 5 hits. We always prefer the most specific-sounding option.
4. **Low margins, not bad calibration.** The 2B puts >0.8 on only 60 of 200 picks (Jev: 111; Sol: 106). In the
   0.4–0.6 band it is right 52% of the time, which is honest (ECE 0.053), but it means the model often cannot
   separate two or three neighbours. Calibration is *not* the problem; discrimination is.
5. **"Order sensitivity" is the same thing seen from the side.** Shuffling the options changes the pick in 24%
   of cases overall, but 38–60% when confidence is <0.6 and 0–2% when it is >0.8. Shuffled accuracy equals
   unshuffled accuracy. So the goal "same pick ≥90%" will not be met by an order-consistency loss alone; it needs
   sharper margins. A consistency loss is still cheap insurance (see C4); literature confirms small models'
   selection bias on option IDs [1][2].
6. **Broad levels are a specific confusion, not general weakness.** Asked for one of 22 fields directly: 2B 50%,
   0.8B 41% (reading the field off the subfield pick: 79% / 75%). 54 of the 2B's 99 errors are two confusions:
   Clinical Medicine → Biomedical Research (33) and Biomedical Research → Biology (21). At the broad level the
   model slides to the "more basic" science. Training had 9k broad-ish questions but almost none with
   Science-Metrix-style field names, and 2,248 rows with ≤3 options.

### 2b. Pick-from-list, unseen schemes (held-out JEL, invented, held-out papers), against the teachers

2B agrees with the teachers' top pick 80–87%; teachers agree with each other 90–92%. The gap concentrates in:

| Slice | 2B agreement | n |
|---|---:|---:|
| Teacher itself ≤50% sure of its top option | **47%** | 417 |
| Teacher 50–70% sure | 64% | 1,012 |
| Teacher 100% sure | 97% | 2,818 |
| Lists of 41–80 options | 78% | 440 |
| Lists of 81–200 options (JEL level 2 has 139) | **74%** | 420 |
| Lists ≤20 options | 85–87% | 3,740 |
| Title-only papers | 82% (vs 84%) | 740 |

Worst units: social-science and humanities schemes (a critical-pedagogy map 47–55%, organisational-downsizing
frameworks 50–57%, a social-science thematic map 60%), JEL at 139 options (47–60%), ACM CCS at 85 options (50–60%).
Our 28 maps and the 7 real schemes are mostly natural science and medicine; so is round 2's topic list.

One unit is a data defect worth fixing in the generator: "Clinical and mycological outcomes… — adverse effects"
offered 4 harm categories to papers that are general treatment reviews. Neither option fits; Sol and Opus spread
their shares arbitrarily and the student "disagrees" 80% of the time. Tree-level questions where the paper does
not belong under the parent are noise in training too (`fmt.NONE` is offered in 33% of training rows but the
eval units never offer it, so we do not measure abstaining at all).

### 2c. Yes/no (the maps' judge), 8 held-out topics (reference-model labels) and 2 development topics (Jev)

| Topic | 0.8B AP | 2B AP | What goes wrong |
|---|---:|---:|---|
| difference-in-differences with staggered treatment | 0.34 | **0.40** | definition says "not causal inference in general"; the model says **yes (p=1.00)** to applied studies that merely *use* DiD and **no (p=0.01–0.03)** to the methodological papers (*Structural mean models for instrumented DiD*, *Semiparametric DiD*, *A simple forward DiD method*). Keyword in the title wins; the "what it covers" clause is ignored. |
| medieval manuscript stemmatics | 0.62 | 0.74 | yes (0.98) to codicology / fragmentology / scribal networks (manuscripts, but no stemma); no (0.05–0.10) to textual-criticism work on manuscript sources. Neighbouring subject counted in. |
| sulfide solid electrolytes for lithium batteries | 0.59 | 0.76 | yes (0.95–0.98) to lithium–**sulfur** batteries, iron-sulfide molten-salt cells, LAGP (an oxide); no (0.00) to **argyrodite** / LPSC papers, which *are* sulfide electrolytes. A vocabulary gap: the small model does not know argyrodite = Li₆PS₅Cl. |
| retrieval-augmented generation attribution | 0.72 | 0.83 | generic RAG papers ≈0.5; attribution papers without the word missed. |
| CAR-T exhaustion, inrush currents, subglacial drainage, white-nose | 0.80–0.99 | 0.85–1.00 | fine |

On the development topics with Jev as comparison: fragmentation AUC 0.987 vs Jev 0.990 (fine); **LLM post-training
precision 0.55 vs Jev 0.84** — any fine-tuning of a language model (adapters, BERT-era, intermediate-task training)
counted as post-training; "LLM" the law degree once at 0.94 despite the definition. Same family of error as DiD:
the topic's boundary clause is not used.

In short, the yes/no weakness is **boundary reading**: (i) the definition's exclusions, (ii) "studies the method"
vs "uses the method", (iii) neighbour subjects sharing the vocabulary, (iv) synonyms the small model does not know.
Round 2's 85k pairs were 25-paper batches with hard labels; no pair was built to teach (i)–(iii) on purpose, and the
yes/no rows replayed in round 3 are hard 0/1 (soft labels: 0 of 40,000).

### 2d. What the training data itself says

- 46k pick rows, teacher 100% sure on 54% of them: little soft signal where it matters (the uncertain half of 2b).
- Paper text is cut to ≤700 tokens in training (`paper_budget`); 15% of rows are title-only.
- One epoch; LoRA rank 32; batch 32; the 2B learned 1–3 points more than the 0.8B at every step.
- No row is built against the model's own mistakes: nothing was mined from where round 2/3 models were unsure or
  disagreed (round 2's plan had this; it was cut for speed).
- "None of these" is offered in 33% of rows, the top target in 5%; never evaluated.
- Schemes: 7 real (natural-science/medicine heavy), 28 maps, 284 invented; held out: Science-Metrix, JEL, 18 invented.

## 3. Proposed next rounds (all on the list-first format of round 4)

Costs: "subscriptions" = Sol/Opus through the plans (no cash); GPU = our two 3090s, yielding to live maps as the
round-3/4 scripts do; Kimi only to measure, and only where said. Nothing below spends more than ~$3 without asking.

### A. First, tests that can see a 3-point gain (no money, CPU, ~half a day) — **built 2026-10-18**

Done, CPU only, no model run yet: `search-dev/experiments/fieldjudge/goldtests/` (README there). 2,980 arXiv papers
(20 per category, 149 options, 5 variants) and 2,500 MeSH papers (9,826 yes/no pairs with minor / sibling / branch
negatives, 2,051 short-list picks), none seen in any training round; scorer with paired bootstrap. Resolution
±1.4 points. Baseline runs (round-3 0.8B and 2B) wait for round 4 to free a GPU. The plan as written:

The 200-article test has ±10-point intervals; the held-out units are teacher labels. Before training anything, build
human-labelled tests big enough to measure the proposals:

1. **arXiv primary categories** (author-assigned, ~150 options): `/mnt/bulk/arxiv/openalex_vs_arxiv.parquet` already
   joins arXiv's categories to our works. 3,000 papers stratified by category and year, excluding anything in
   training. Pick-from-list with the real arXiv list; also the 8 archive-level broad question.
2. **MeSH major descriptors** (NLM indexers): the `mesh` column in our works. Yes/no "is X a central subject?" on
   2,000 (paper, descriptor) pairs with hard negatives = descriptors present but not major, and siblings in the
   MeSH tree. This is the only human-made yes/no test we can get for free, and it matches the maps' question.
3. **Keep the expert test as is**; add the option-order and 22-field variants to the new tests.
4. Report by slice: title-only, list size, broad vs leaf, field (social sciences / humanities separately).

Gain: measurement, and a shared yardstick for every proposal below. Risk: arXiv and MeSH were *teacher-labelled*
training schemes in round 3, so held-out papers are needed; the author label is also a different convention from the
teacher's (that difference is itself informative).

### B. Round 5 data: a targeted labelling run (~1 day; subscriptions ~2–3 h; GPU ~2 h for mining)

One labelling run that fixes the five data gaps, instead of "more of the same":

| B# | What | Teaches | Size |
|---|---|---|---|
| B1 | **Mined hard examples**: once round 4 is done, run its 0.8B and 2B over ~60k unlabelled (paper, scheme) questions on the existing schemes; keep the ones where (a) the top-2 margin is <0.2, (b) the two sizes disagree, or (c) two option orders disagree. Teachers label them with shares. | the near-ties of 2a-4/5 and 2b | 15–20k pick rows |
| B2 | **Broad and short lists**: for every scheme, questions at its top two levels; plus invented schemes whose options are *levels of generality* (basic science / translational / clinical; theory / method / application) with the scheme text saying how to read them. | 2a-6 | 8–10k |
| B3 | **Convention-aware schemes**: pairs of schemes over the same papers, one "by discipline/method", one "by object/application", each with a one-line rule in its description; the teacher labels both. Include catch-all "General X" and "None of these" options and have the teacher use them. | 2a-2, 2a-3 | 6–8k |
| B4 | **Boundary yes/no with soft labels**: topics with a "what it covers / not" clause, built in triplets: on-topic; *uses the method but does not study it*; *neighbour subject with the same vocabulary*; plus synonym/jargon variants (the teacher lists 5 synonyms, we retrieve papers by them). Both teachers answer with a probability, kept soft. Oversample social sciences, humanities, economics, statistics/methods. | 2c | 15k pairs |
| B5 | **Relabel the uncertain**: the 4,200 pick rows where the single teacher was ≤70% sure get the second teacher; target = average. Drop or add "None of these" to tree-level units whose papers do not fit the parent (detect: teacher spread flat across options). | 2b | 4k relabels |

Expected gain (honest guess, to be measured on A): expert test +2–4, unseen-scheme agreement +3–5 (towards the
90% ceiling), broad-field question 50 → 70+, yes/no AP on DiD/stemmatics-type topics +0.1–0.2, no loss elsewhere
because round-3/4 data is kept. Literature: uncertainty-targeted data is the standard way to spend a labelling
budget [3]; soft targets from several annotators beat hard majority labels when annotators disagree [4]; noisy-student
style self-training with the model's own uncertain cases improves students without new teacher calls [5].

### C. Round 5 training recipe (0.8B ~6 h GPU, 2B ~14 h; run after B, on round-4 models)

| C# | Change | Why | Cost |
|---|---|---|---|
| C1 | **Full-weight fine-tune for the 0.8B** (keep LoRA for the 2B; it fits in ~21 GB) | the 0.8B is 10 points behind on general journals and 18 on title-only; LoRA learns less than full training on large targets [6]; Maxime already saw this on the rewriter 0.8B | same time, 21 GB instead of 9 (needs GPU 0 or a quiet GPU 1) |
| C2 | **2 epochs**, round-3/4 data + B | round 3 ran 1 epoch and was still improving at the end (79 → 81% between steps 1,200 and 1,600) | 2× time |
| C3 | **Paper budget 700 → 1,100 tokens** | list-first makes the paper the only per-paper cost, so this is affordable now; abstracts are cut today | +15% time |
| C4 | **Two orders per pick question in the same batch, symmetric-KL consistency term (weight 0.1)** on top of the soft cross-entropy | insurance against selection bias [1][2]; it also acts as the "3-order ensemble" at training time | +40% time on pick rows |
| C5 | **Loss weight by teacher agreement** (B5 rows where teachers disagree count half) | less noise from arbitrary spreads [4] | free |

Measured on A + the round-3 suite. Pilot first: C1 alone on the 0.8B with round-4 data (5 h on GPU 1) tells
whether full fine-tuning is worth it before B is even labelled.

### D. Pilots that may or may not pay (ask before running; cheap)

| D# | Idea | Why it might help | Cost | Go/no-go measure |
|---|---|---|---|---|
| D1 | **Human-labelled training rows at zero cost**: 20k arXiv-category rows and 20k MeSH-major yes/no rows mixed in (author/indexer labels, not teacher) | real human signal; big; removes the 90% teacher ceiling on two schemes | CPU to build; the training time above | A-tests on held-out arXiv/MeSH papers *and* the expert test (to check it does not skew the model to those two conventions) |
| D2 | **Rationale multi-task** ("Distilling step-by-step" [7]): for 5k rows the teacher writes one line of reason; training adds a second objective of generating it; inference unchanged (no generation) | the paper reports small students beating larger ones with less data; our failures are "reasoning about conventions", exactly what rationales carry | subscriptions ~1 h, GPU +30% | expert test and A; drop if <+2 |
| D3 | **References as input**: append the titles of up to 5 references to the paper text (OpenAlex has them for most non-arXiv works; the maps pipeline has them anyway) | citation context is what SPECTER-style models gain from [8]; the 0.8B's title-only score (41%) is the biggest single gap | +100 tokens/paper at inference; retrain with the field present 50% of the time | title-only slice of A and the expert test |
| D4 | **A 4B student** (Qwen3.5-4B, LoRA) on the same data | the 2B gained +10 on general journals over the 0.8B; the curve may continue | ~30 h GPU, 2× slower than the 2B in use | only if Maxime wants a "quality" tier; measure cost/quality on the Pareto chart |
| D5 | **Test-time 2-order averaging for the 0.8B only** | measured already: helps the 2B (+6 specialist), not the 0.8B; re-check on round 5 | none | A |

### E. What I would run, in order

1. **A** now (CPU, no GPU, no money): the arXiv and MeSH gold tests. Half a day.
2. When round 4 finishes and its models are tested: **C1 pilot** (full-weight 0.8B, same data), 5 h on GPU 1, and
   the **B1 mining** pass (~2 h). Both only need round-4's merged models, read-only.
3. **B** labelling (Sol + Opus, ~3 h of subscription time, ~3,000 requests; last round's run reached 43% of Sol's
   5-hour window and 51% of Opus's weekly one, so this fits but should be checked first).
4. **Round 5** with C (0.8B ~6 h, 2B ~14 h), evaluated on A + the round-3 suite + the two-topic replay.
5. D1–D3 as separate pilots on the 0.8B afterwards, one variable at a time, each ~5 h GPU.

Everything above costs subscription time and our own GPUs; the only cash is Kimi for reading-list scoring in the
replay (~$1–3). D4 is the only item with a real time cost (30 GPU hours) and should be a deliberate choice.

## References

- [1] https://doi.org/10.48550/arxiv.2309.03882 — Zheng 2023, Large language models are not robust multiple choice selectors
- [2] https://doi.org/10.18653/v1/2024.findings-naacl.130 — Pezeshkpour & Hruschka 2024, LLM sensitivity to the order of options
- [3] https://openalex.org/W3180304956 — Schröder 2021, Uncertainty-based query strategies for active learning with transformers
- [4] https://doi.org/10.1109/cvpr.2019.01150 — Tanno 2019, Learning from noisy labels by regularized estimation of annotator confusion
- [5] https://doi.org/10.48550/arxiv.1911.04252 — Xie 2019, Self-training with Noisy Student
- [6] https://doi.org/10.48550/arxiv.2405.09673 — Biderman 2024, LoRA learns less and forgets less
- [7] https://doi.org/10.48550/arxiv.2305.02301 — Hsieh 2023, Distilling step-by-step
- [8] https://doi.org/10.18653/v1/2020.acl-main.207 — Cohan 2020, SPECTER: citation-informed transformers
- [9] https://doi.org/10.1371/journal.pone.0251493 — Rivest 2021, the 200-article expert benchmark
