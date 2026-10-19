# Steps 1–3 of the recommendations: what was built and measured (2026-10-19)

Nothing below is live yet. Everything runs on a private copy of the search API (`search-identity/identity-dev.sh`,
127.0.0.1:8812, branch `identity` of `search/`) and a test release under `/mnt/fast/scholarsreadinglist/release-identity`.
Details and rules: `pipeline/identity/README.md`, DATA.md "Paper identity, rescued abstracts, author identity".

## 1. One paper, one entry (paper versions)

- 14.0M papers have 2+ OpenAlex records; 16.3M surplus records (3.5M preprint + published pairs, 0.33M mirror copies).
- Read by hand: 144 of 145 random groups are truly one paper.
- Semantic index: 4.36M extra versions leave it (−3.5 %), readable papers are kept first (no loss of readable ones).
- Results: a paper appears once, as its readable or published version, with `versions` and `versions_count`.
  Example, keyword search `"inverse participation ratio" Anderson transition`: the PRL paper, its Regensburg
  repository copy and its arXiv copy were 3 of the top 10; now one result ("3 versions"), and journal versions
  replace arXiv copies through the list.

## 2. Missing abstracts (rescue)

- PubMed baseline + updates downloaded and parsed (free): 41.3M records, 29.5M abstracts.
- Probe first (1,000 abstract-less articles, Europe PMC): 30 % have an abstract there (Elsevier 36 %, Springer 37 %,
  Nature 81 %); 97 % of those through a PMID OpenAlex already holds, hence a bulk join by PMID, not API calls.
- Rescued: **7.79M English publications** OpenAlex holds without an abstract: 7.06M from PubMed (PMID or DOI),
  729k from another version of the same paper. PubMed's title must match OpenAlex's (98.9 % agree ≥ 0.9; 2,365 with
  a wrong PMID dropped).
- On the 49 expert-test articles the site's corpus lacked: 27 now have their abstract (live: 0).
- Vectors: the 2M most cited embedded first (GPU 1, 3 GiB, ~2 h); 5.8M left (~5.5 h of GPU 1).
- 9.8M PubMed abstracts carry a publisher copyright line (Elsevier, Springer...). Using them to find papers is one
  thing; showing their text on the site and through the API is another: decision needed (see below).

## 3. Authors: the certain, ORCID-based fixes, and the yardstick for the model

- 1.52M OpenAlex author records join the person their ORCID names (names must agree; the ORCID evidence must cover
  the record, so OpenAlex's own namesake mixtures do not join whole).
- 6.74M paper authorships that had no author record are attached to the person whose ORCID the publisher printed;
  939k authorships move from the wrong record to the ORCID's person.
- 397k records flagged as not people ("Anonymous", ":unav", universities, lists of names): out of people search.
- Held-out test (9.67M authorships, 1.06M ORCIDs): a person's page holds **83.2 % → 98.0 %** of their papers;
  wrong papers on a page stay at 0.04–0.05 %. Prolific people 73 % → 96 %; 2020s papers 76 % → 97 %.
- Limit, stated plainly: every fix rests on ORCIDs. For people whose papers carry no ORCID, nothing changes (the
  "fair" score equals OpenAlex's). That is the model's job (step 4); `authors/eval.parquet` is its training and test
  set (87.2M + 9.67M authorships, name blocks).
- Hand checks of the rules: 40 merges only the lenient name test allows: ~38 right; 25 moves: 25 right; 40 records
  flagged as not people: ~36 right (catalogue names with links were the misses, fixed since).

## Mistake corrected

The audit said one paper (Vaccine 2023, 10.1016/j.vaccine.2023.02.054) was only findable through its mirror copy.
Wrong: it is in the corpus with its abstract; the check was a careless search. README.md is corrected.

## Decisions for Maxime

1. Publish? Order proposed: versions first (lowest risk), then author identity, then the rescue.
2. PubMed abstracts: find-only (search, maps, lists use them; pages show "abstract on PubMed" with a link), or shown in
   full with the source named. Publisher copyright applies to 9.8M of them.
3. The other 5.8M rescued papers' vectors: ~5.5 h of GPU 1, when convenient.
4. The site (its own repository) does not show `versions` or the identity note yet; that is a small UI change.
