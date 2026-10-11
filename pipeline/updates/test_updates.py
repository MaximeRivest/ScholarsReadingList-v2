"""Tests of the daily updates' own logic (no network).

    uv run --with pytest --with 'polars>=1.30' --with 'pyarrow>=17' --with 'numpy>=2' python -m pytest -q pipeline/updates

check_api.py is the live check against OpenAlex (the embedded text of real works is identical to the snapshot's).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import fetch as F  # noqa: E402


def test_window_order_and_bounds():
    w = F.window(dt.date(2026, 10, 11))
    base = dt.date.fromisoformat(C.BASE_SNAPSHOT_DATE)
    assert w[0] == "2026-10-11" and w[len(range((dt.date(2026, 10, 11) - base).days))] == C.BASE_SNAPSHOT_DATE
    assert "2026-12-10" in w and "2026-12-11" not in w                    # 60 days ahead
    assert (base - dt.timedelta(30)).isoformat() in w and (base - dt.timedelta(31)).isoformat() not in w
    assert len(w) == len(set(w))


def record(**over):
    r = {"id": "https://openalex.org/W7000000001", "doi": "https://doi.org/10.1/x",
         "title": "A study of sleep", "display_name": "A study of sleep", "type": "article", "language": "en",
         "publication_date": "2026-10-08", "publication_year": 2026,
         "authorships": [{"author": {"display_name": "Ada Lovelace", "orcid": None, "id": "A1"},
                          "is_corresponding": True, "institutions": [], "countries": ["GB"],
                          "raw_author_name": "Ada Lovelace"},
                         {"author": {"display_name": "Bo Li", "orcid": None, "id": "A2"},
                          "is_corresponding": False, "institutions": [], "countries": [], "raw_author_name": "Bo Li"}],
         "primary_location": {"landing_page_url": "https://x", "pdf_url": None, "license": "cc-by",
                              "source": {"id": "https://openalex.org/S42", "display_name": "J. Sleep",
                                         "type": "journal", "is_core": True, "is_in_doaj": False,
                                         "host_organization_name": "Pub", "issn_l": "1234-5678"}},
         "best_oa_location": None, "biblio": {"volume": "3", "issue": None, "first_page": "1", "last_page": "9"},
         "primary_topic": {"display_name": "Sleep", "subfield": {"display_name": "Neurology"},
                           "field": {"display_name": "Medicine"}, "domain": {"display_name": "Health Sciences"}},
         "keywords": [{"display_name": "sleep"}], "mesh": [], "open_access": {"is_oa": True, "oa_status": "gold"},
         "funders": [], "awards": [], "cited_by_count": 0, "referenced_works_count": 12,
         "referenced_works": ["https://openalex.org/W1"], "is_retracted": False,
         "updated_date": "2026-10-10T09:11:00.788812", "created_date": "2026-10-09T00:00:00",
         "abstract_inverted_index": {"Sleep": [0], "matters,": [1], "\u00e9t\u00e9": [3], "really": [2]},
         "is_xpac": False, "is_paratext": False, "indexed_in": ["crossref", "doaj"], "locations_count": 2,
         "fwci": None, "citation_normalized_percentile": None}
    r.update(over)
    return r


def test_facts_encoding():
    t = F.facts([record(), record(id="https://openalex.org/W7000000002", fwci=1.0)
                 | {"is_paratext": True}, {k: v for k, v in record().items() if k != "fwci"}]).to_pylist()
    assert len(t) == 2                                # the record without every field is left out
    a = t[0]
    assert a["id"] == 7000000001 and a["source_type"] == 1 and a["source"] == 42 and a["source_is_core"]
    assert a["indexed_in"] == 0b101 and a["refs"] == 12 and a["authors"] == 2 and a["locations"] == 2
    assert a["citation_pct"] is None and t[1]["is_paratext"]


@pytest.mark.skipif(not (C.UPDATES / "raw-schema.arrow").exists() and not C.RAW_SCHEMA_SOURCE.exists(),
                    reason="needs the snapshot's raw schema")
def test_api_record_to_content_row():
    schema = F.raw_schema()
    rows = F.to_content([record(), record(id="https://openalex.org/W7000000003", language="fr"),
                         record(id="https://openalex.org/W7000000004", abstract_inverted_index=None),
                         record(id="https://openalex.org/W7000000005", type="dataset")], schema).to_dicts()
    assert [r["id"] for r in rows] == ["https://openalex.org/W7000000001"]     # French, no abstract, dataset: out
    r = rows[0]
    assert r["abstract"] == "Sleep matters, really \u00e9t\u00e9"                    # words back in position order
    assert r["authors"] == "Ada Lovelace | Bo Li" and r["venue"] == "J. Sleep" and r["field"] == "Medicine"
    assert r["pages"] == "1\u20139" and str(r["date"]) == "2026-10-08" and r["year"] == 2026


def test_known_ids(tmp_path, monkeypatch):
    (tmp_path / "map-meta").mkdir()
    np.save(tmp_path / "map-meta" / "id.npy", np.array([5, 9, 7000000001], np.int64))
    monkeypatch.setattr(C, "BASE_CORPUS", tmp_path)
    monkeypatch.setattr(C, "CONTENT", tmp_path / "content")
    k = F.Known()
    assert 9 in k and 7000000001 in k and 8 not in k and 10**10 not in k
    k.extra.add(8)
    assert 8 in k
