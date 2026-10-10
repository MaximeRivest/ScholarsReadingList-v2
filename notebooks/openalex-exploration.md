---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../r_ports_to_py/dpyr"
      - "-e ../functai/python"
      - "-e ../lmcc/python"
---

```python
import gc, html, json, re, types
from dpyr import DFrame

_TABLE_JS = r"""
const root = document.currentScript.previousElementSibling;
const {cols, rows, total} = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
const URL_RE = /(https?:\/\/[^\s<>"']+)/g, LONG = 300;
let page = 0, size = 25;
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify = s => esc(s).replace(URL_RE, u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
function cell(v) {
  if (v === null) return '<span class="na">null</span>';
  const nested = typeof v === 'object';
  const s = nested ? JSON.stringify(v) : String(v);
  const cls = nested ? ' class="nested"' : '';
  if (s.length <= LONG) return `<div${cls}>${linkify(s)}</div>`;
  return `<details${cls}><summary>${linkify(s.slice(0, LONG))}…</summary>${linkify(s)}</details>`;
}
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages - 1);
  const slice = rows.slice(page * size, page * size + size);
  root.querySelector('tbody').innerHTML = slice.map(r => '<tr>' + r.map(v => `<td>${cell(v)}</td>`).join('') + '</tr>').join('');
  const first = rows.length ? page * size + 1 : 0, last = page * size + slice.length;
  root.querySelector('.info').textContent =
    `rows ${first}–${last} of ${rows.length.toLocaleString()} loaded` + (total === null ? ' (total unknown)' : ` · ${total.toLocaleString()} in table`);
  root.querySelector('.pg').textContent = `page ${page + 1} / ${pages}`;
  root.querySelector('.prev').disabled = page === 0;
  root.querySelector('.next').disabled = page >= pages - 1;
  root.querySelector('.wrap').scrollTop = 0;
}
root.querySelector('thead tr').innerHTML = cols.map(([n, short, full]) =>
  `<th title="${esc(n + ': ' + full)}">${esc(n)}<div class="t">${esc(short)}</div></th>`).join('');
root.querySelector('.prev').onclick = () => { page--; render(); };
root.querySelector('.next').onclick = () => { page++; render(); };
root.querySelector('.size').onchange = e => { size = +e.target.value; page = 0; render(); };
render();
"""

_TABLE_CSS = """
.dp{font:13px system-ui,sans-serif;color:inherit}
.dp .bar{display:flex;gap:8px;align-items:center;margin:2px 0 4px;flex-wrap:wrap}
.dp .bar .info{opacity:.7;margin-right:auto}
.dp button,.dp select{font:inherit;padding:1px 8px}
.dp .wrap{max-height:560px;overflow:auto;border:1px solid #8884}
.dp table{border-collapse:collapse}
.dp th,.dp td{border-bottom:1px solid #8883;padding:4px 8px;vertical-align:top;text-align:left;
  min-width:120px;max-width:380px;white-space:pre-wrap;overflow-wrap:anywhere}
.dp th{position:sticky;top:0;z-index:1;background:Canvas;white-space:normal}
.dp .t{font-weight:normal;opacity:.55;font-size:11px}
.dp .na{opacity:.4}
.dp .nested{font:11px ui-monospace,monospace}
.dp details summary{cursor:pointer;list-style:none}
.dp details[open] summary{display:none}
.dp a{color:#3b82f6}
"""

def _short_type(t):
    """'List(Struct(id: Str, ...))' -> 'list<struct>': short enough for a header."""
    s = repr(t)
    m = re.match(r"(\w+)\((\w+)", s)
    if not m:
        return s.lower()
    return f"{m.group(1).lower()}<{m.group(2).lower()}>" if m.group(1) in ("List", "Array") else m.group(1).lower()

def table_html(df, rows=200):
    """A dpyr table as HTML: loads up to `rows` rows, shows them 25 per page."""
    from dpyr.materialize import preview
    shown, total = preview(df.plan, rows)
    data = {
        "cols": [[c, _short_type(t), repr(t)] for c, t in df.schema.items()],
        "rows": [list(r) for r in shown.iter_rows()],
        "total": total,
    }
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return f"""<style>{_TABLE_CSS}</style><div class="dp">
<div class="bar"><span class="info"></span>
<button class="prev">‹ prev</button><span class="pg"></span><button class="next">next ›</button>
<select class="size"><option>10</option><option selected>25</option><option>50</option><option>100</option></select></div>
<div class="wrap"><table><thead><tr></tr></thead><tbody></tbody></table></div>
<script type="application/json">{payload}</script></div><script>{_TABLE_JS}</script>"""

class html_view:
    """Show a dpyr table as HTML. `rows` = how many rows to load (paged 25 at a time)."""
    def __init__(self, df, rows=200): self.df, self.rows = df, rows
    def _repr_html_(self): return table_html(self.df, self.rows)

def _show_tables_as_html():
    """Wrap the kernel's 'show the last value' step so dpyr tables become HTML."""
    show = next(f for f in gc.get_objects()
                if isinstance(f, types.FunctionType) and f.__name__ == "_show_value"
                and f.__globals__.get("__name__") == "__main__")
    original = getattr(show, "_original", show)
    def patched(value):
        if isinstance(value, DFrame) and value._repr_html_() is not None:
            return original(html_view(value))
        return original(value)
    patched._original = original
    show.__globals__["_show_value"] = patched

_show_tables_as_html()
```


# Explore OpenAlex: papers, abstracts, authors, and citations

Run the cells from top to bottom. The data is on **lambda**, in
`/mnt/bulk/openalex/works/`, and is still downloading.

Everything uses **dpyr**: `read`, `filter`, `mutate`, `count`, and friends.
No model is called unless you turn on the last cell.

## 1. Load

```python
import json
from dpyr import read, col, desc, n, vectorize

# Every finished file. Unfinished downloads end in .part, so they are skipped.
# Rerun this cell to include files that finished since.
works = read("/mnt/bulk/openalex/works/*/*.parquet")
works
```

<iframe class="rat-output" src="../_assets/generated/d0ec69f962c3.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

```py
works.columns
```

```output
['id', 'doi', 'title', 'display_name', 'ids', 'indexed_in', 'publication_date', 'publication_year', 'language', 'type', 'authorships', 'authors_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'primary_topic', 'topics', 'keywords', 'concepts', 'locations', 'locations_count', 'primary_location', 'best_oa_location', 'sustainable_development_goals', 'awards', 'funders', 'institutions', 'countries_distinct_count', 'institutions_distinct_count', 'open_access', 'is_paratext', 'is_retracted', 'is_xpac', 'biblio', 'referenced_works', 'referenced_works_count', 'related_works', 'abstract_inverted_index', 'cited_by_count', 'counts_by_year', 'apc_list', 'apc_paid', 'fwci', 'citation_normalized_percentile', 'cited_by_percentile_year', 'mesh', 'has_content', 'has_fulltext', 'created_date', 'updated_date']
```

```py
works.schema
```

```output
{'id': Str, 'doi': Str, 'title': Str, 'display_name': Str, 'ids': List(Struct(key: Str, value: Str)), 'indexed_in': List(Str), 'publication_date': Date, 'publication_year': Int64, 'language': Str, 'type': Str, 'authorships': List(Struct(author: Struct(display_name: Str, id: Str, orcid: Str), author_position: Str, affiliations: List(Struct(institution_ids: List(Str), raw_affiliation_string: Str)), countries: List(Str), raw_author_name: Str, raw_orcid: Str, is_corresponding: Bool, raw_affiliation_strings: List(Str), institutions: List(Struct(country_code: Str, display_name: Str, id: Str, lineage: List(Str), ror: Str, type: Str)))), 'authors_count': Int64, 'corresponding_author_ids': List(Str), 'corresponding_institution_ids': List(Str), 'primary_topic': Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str)), 'topics': List(Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str))), 'keywords': List(Struct(id: Str, display_name: Str, score: Float64)), 'concepts': List(Struct(id: Str, wikidata: Str, display_name: Str, level: Int64, score: Float64)), 'locations': List(Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool)), 'locations_count': Int64, 'primary_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'best_oa_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'sustainable_development_goals': List(Struct(id: Str, display_name: Str, score: Float64)), 'awards': List(Struct(id: Str, display_name: Str, funder_award_id: Str, funder_id: Str, funder_display_name: Str)), 'funders': List(Struct(id: Str, display_name: Str, ror: Str)), 'institutions': List(Struct(id: Str, display_name: Str, ror: Str, country_code: Str, type: Str, lineage: List(Str))), 'countries_distinct_count': Int64, 'institutions_distinct_count': Int64, 'open_access': Struct(is_oa: Bool, oa_status: Str, any_repository_has_fulltext: Bool, oa_url: Str), 'is_paratext': Bool, 'is_retracted': Bool, 'is_xpac': Bool, 'biblio': Struct(volume: Str, issue: Str, first_page: Str, last_page: Str), 'referenced_works': List(Str), 'referenced_works_count': Int64, 'related_works': List(Str), 'abstract_inverted_index': Str, 'cited_by_count': Int64, 'counts_by_year': List(Struct(year: Int64, cited_by_count: Int64)), 'apc_list': Struct(value: Int64, currency: Str, value_usd: Int64), 'apc_paid': Struct(value: Float64, currency: Str, value_usd: Float64), 'fwci': Float64, 'citation_normalized_percentile': Struct(value: Float64, is_in_top_1_percent: Bool, is_in_top_10_percent: Bool), 'cited_by_percentile_year': Struct(min: Int64, max: Int64), 'mesh': List(Struct(descriptor_ui: Str, descriptor_name: Str, qualifier_ui: Str, qualifier_name: Str, is_major_topic: Bool)), 'has_content': Struct(pdf: Bool, grobid_xml: Bool), 'has_fulltext': Bool, 'created_date': Datetime, 'updated_date': Datetime}
```

```py
works.select(col.title, col.doi, col.publication_year)
```

<iframe class="rat-output" src="../_assets/generated/c766f1db2608.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

Show tables as HTML: run this once, and every cell that ends with a dpyr
table shows it as a paged HTML table: 25 rows per page (10/25/50/100), full
text wrapped, long cells folded (click to open), links clickable, scrolls both
ways. It loads the first 200 rows; `html_view(df, rows=1000)` loads more.
Restarting the kernel turns it off; rerun to turn it back on.



## 2. Take a small sample

Searching all works reads every file, which takes a while. To explore, keep the
first 20,000 works in memory. They come from the first files, so this is
**not a random sample**.

```python
# English articles from 2020 on that list their references.
# Filtering ALL works first is too slow (>5 min), so look at the first 200,000 only.
sample = (works.slice_head(200_000)
          .filter(col.type == "article",
                  col.language == "en", col.referenced_works_count > 0)
          .persist())
sample.shape
```

```output
(3401, 49)
```

## 3. A readable table of papers

OpenAlex stores each abstract as words with their positions. This small function
puts the words back in order. `@vectorize` lets dpyr use it on a whole column.

```python
@vectorize
def abstract_text(inverted_index: str) -> str:
    if not inverted_index:
        return None
    index = json.loads(inverted_index)
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)

papers = (
    sample
    .unnest_wider(col.primary_topic, names_sep="_")
    .mutate(abstract=abstract_text(col.abstract_inverted_index))
    .select(col.id, col.title, col.publication_year, col.type,
            col.cited_by_count, col.primary_topic_display_name, col.abstract)
    .rename(year=col.publication_year, citations=col.cited_by_count,
            topic=col.primary_topic_display_name)
)
papers
```

<iframe class="rat-output" src="../_assets/generated/d84cbf30abf1.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 3b. A tidy content table

`content_table(works)` turns raw works into a flat table: one row per
publication, one plain value per cell, no lists or records. It keeps what a
publisher's page shows to a reader without access: title, authors,
affiliations, venue, volume/issue/pages, date, abstract, keywords, topic, open
access and license, funding, citation counts, and the list of references.

Lists (authors, keywords, references…) are joined with ` | `. A `|` inside a
value becomes `¦`, so `text.split(" | ")` always gets the items back.
`author_orcids` lines up with `authors` one for one (`-` when there is no ORCID).

The column names are generic (`title`, `authors`, `date`, `venue`, `abstract`,
`url`…), so books, preprints or blog posts can fill the same table later.

It runs in Python one row at a time: fine for a sample, too slow for all of `works`.

```python
from dataclasses import dataclass
from dpyr import vectorize

SEP = " | "   # between items in a list column; a "|" inside an item becomes "¦"

def _join(items):
    """Several values -> one string. None/empty items are dropped; None if nothing left."""
    out = [str(x).replace("|", "¦").strip() for x in items or [] if x not in (None, "")]
    return SEP.join(out) if out else None

def _get(d, *path):
    """d['a']['b'] that returns None instead of failing on a missing level."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d

@dataclass
class Content:
    url: str | None
    pdf_url: str | None
    venue: str | None
    publisher: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    authors: str | None
    author_orcids: str | None
    corresponding_authors: str | None
    institutions: str | None
    countries: str | None
    abstract: str | None
    keywords: str | None
    topic: str | None
    subfield: str | None
    field: str | None
    domain: str | None
    is_open_access: bool | None
    open_access_status: str | None
    license: str | None
    funders: str | None
    grants: str | None
    mesh_terms: str | None
    references: str | None

@vectorize
def _flatten_work(authorships: list, primary_location: dict, best_oa_location: dict,
                  biblio: dict, primary_topic: dict, keywords: list, open_access: dict,
                  funders: list, awards: list, mesh: list, referenced_works: list,
                  abstract_inverted_index: str) -> Content:
    authorships = authorships or []
    loc = primary_location or {}
    source = loc.get("source") or {}
    oa = best_oa_location or {}
    first, last = _get(biblio, "first_page"), _get(biblio, "last_page")
    pages = f"{first}–{last}" if first and last and first != last else first
    insts = {}
    for a in authorships:
        for inst in a.get("institutions") or []:
            insts.setdefault(inst.get("display_name"), None)
    abstract = None
    if abstract_inverted_index:
        index = json.loads(abstract_inverted_index) or {}
        words = sorted((pos, w) for w, ps in index.items() for pos in ps)
        abstract = " ".join(w for _, w in words) or None
    return Content(
        url=loc.get("landing_page_url"),
        pdf_url=oa.get("pdf_url") or loc.get("pdf_url"),
        venue=source.get("display_name") or loc.get("raw_source_name"),
        publisher=source.get("host_organization_name"),
        issn=source.get("issn_l"),
        volume=_get(biblio, "volume"),
        issue=_get(biblio, "issue"),
        pages=pages,
        authors=_join(_get(a, "author", "display_name") or a.get("raw_author_name") for a in authorships),
        author_orcids=_join(_get(a, "author", "orcid") or "-" for a in authorships),
        corresponding_authors=_join(_get(a, "author", "display_name") for a in authorships if a.get("is_corresponding")),
        institutions=_join(insts),
        countries=_join(dict.fromkeys(c for a in authorships for c in a.get("countries") or [])),
        abstract=abstract,
        keywords=_join(k.get("display_name") for k in keywords or []),
        topic=_get(primary_topic, "display_name"),
        subfield=_get(primary_topic, "subfield", "display_name"),
        field=_get(primary_topic, "field", "display_name"),
        domain=_get(primary_topic, "domain", "display_name"),
        is_open_access=_get(open_access, "is_oa"),
        open_access_status=_get(open_access, "oa_status"),
        license=oa.get("license") or loc.get("license"),
        funders=_join(f.get("display_name") for f in funders or []),
        grants=_join(f"{a.get('funder_display_name')}: {a.get('funder_award_id')}" for a in awards or []
                     if a.get("funder_award_id")),
        mesh_terms=_join(dict.fromkeys(m.get("descriptor_name") for m in mesh or [])),
        references=_join(referenced_works),
    )

def content_table(raw):
    """Raw OpenAlex works -> one flat row per publication (no lists, no records)."""
    return (
        raw
        .mutate(content=_flatten_work(
            col.authorships, col.primary_location, col.best_oa_location, col.biblio,
            col.primary_topic, col.keywords, col.open_access, col.funders, col.awards,
            col.mesh, col.referenced_works, col.abstract_inverted_index))
        .select(col.id, col.doi, col.title, col.type, col.language, col.publication_date,
                col.publication_year, col.cited_by_count, col.referenced_works_count,
                col.is_retracted, col.updated_date, col.content)
        .unnest_wider(col.content)
        .select(col.id, col.doi, col.url, col.pdf_url,
                col.title, col.type, col.language, col.publication_date, col.publication_year,
                col.authors, col.author_orcids, col.corresponding_authors, col.institutions, col.countries,
                col.venue, col.publisher, col.issn, col.volume, col.issue, col.pages,
                col.abstract, col.keywords, col.topic, col.subfield, col.field, col.domain, col.mesh_terms,
                col.is_open_access, col.open_access_status, col.license,
                col.funders, col.grants,
                col.cited_by_count, col.referenced_works_count, col.references,
                col.is_retracted, col.updated_date)
        .rename(date=col.publication_date, year=col.publication_year,
                citations=col.cited_by_count, references_count=col.referenced_works_count)
    )

content = content_table(sample)
content
```

<iframe class="rat-output" src="../_assets/generated/9b973bd3ab1e.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 4. Find papers

Change the words and the year. `(?i)` ignores upper/lower case; `|` means OR.

```python
matches = (
    content
    .filter(col.field.str_detect(r"Agricultural and Biological Sciences"), col.year >= 2000)
    .arrange(desc(col.citations))
)
matches
```

<iframe class="rat-output" src="../_assets/generated/7a1bc4452c5c.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 5. Read one abstract

The most-cited paper in the sample that has an abstract.

```python
paper = papers.filter(~col.abstract.is_na()).slice_max(col.citations, 1)
print(paper.pull(col.title)[0])
print()
print(paper.pull(col.abstract)[0])
```

## 6. Authors

Each paper holds a list of authors. `unnest_longer` gives each author its own
row; `unnest_wider` turns their details into columns.

```python
authors = (
    sample
    .select(col.id, col.authorships)
    .unnest_longer(col.authorships)
    .unnest_wider(col.authorships)
    .unnest_wider(col.author, names_sep="_")
    .select(col.id, col.author_display_name, col.author_position)
)
authors.count(col.author_display_name).arrange(desc(col.n))
```

## 7. Keywords

The same idea. These keywords are added by OpenAlex, not chosen by the authors.

```python
keywords = (
    sample
    .select(col.id, col.keywords)
    .unnest_longer(col.keywords)
    .unnest_wider(col.keywords, names_sep="_")
)
keywords.count(col.keywords_display_name).arrange(desc(col.n))
```

## 8. Citations

One row per link: paper `id` cites paper `cited_id`. Some links are missing in
OpenAlex. A cited paper is often outside the sample, so its title shows as `null`.

```python
citations = (
    sample
    .select(col.id, col.referenced_works)
    .unnest_longer(col.referenced_works)
    .rename(cited_id=col.referenced_works)
)

# Which papers does the sample cite most?
(
    citations
    .count(col.cited_id)
    .left_join(papers.select(col.id, col.title).rename(cited_id=col.id), on=col.cited_id)
    .arrange(desc(col.n))
)
```

## 9. Save a reading list

Writes your search results from section 4 to a CSV file in this project.

```python
matches.select(col.id, col.title, col.year, col.citations).write("research/openalex/reading-list.csv")
```

## 10. Optional: explain papers with functai

A functai function works inside `mutate`, like `abstract_text` above. Each row is
one model call, so this sends **3 abstracts** to the model and may cost money.
It is off until you set `RUN_AI = True` and choose a model.

```python
import functai
from functai import ai

@ai
def explain_paper(title: str, abstract: str) -> str:
    """Explain this paper's question, approach, and findings in three short sentences.
    Use only the title and abstract. Do not invent missing details.
    """
    ...

RUN_AI = False
MODEL = None   # for example "claude-haiku-4-5"

if RUN_AI:
    with functai.configure(lm=MODEL):
        explained = (
            papers
            .filter(~col.abstract.is_na())
            .slice_head(3)
            .mutate(explanation=explain_paper(col.title, col.abstract))
            .select(col.title, col.explanation)
            .collect()
        )
    print(explained)
else:
    print("Skipped. Set RUN_AI = True and MODEL to try it.")
```
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../r_ports_to_py/dpyr"
      - "-e ../functai/python"
      - "-e ../lmcc/python"
---

```python
import gc, html, json, re, types
from dpyr import DFrame

_TABLE_JS = r"""
const root = document.currentScript.previousElementSibling;
const {cols, rows, total} = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
const URL_RE = /(https?:\/\/[^\s<>"']+)/g, LONG = 300;
let page = 0, size = 25;
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify = s => esc(s).replace(URL_RE, u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
function cell(v) {
  if (v === null) return '<span class="na">null</span>';
  const nested = typeof v === 'object';
  const s = nested ? JSON.stringify(v) : String(v);
  const cls = nested ? ' class="nested"' : '';
  if (s.length <= LONG) return `<div${cls}>${linkify(s)}</div>`;
  return `<details${cls}><summary>${linkify(s.slice(0, LONG))}…</summary>${linkify(s)}</details>`;
}
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages - 1);
  const slice = rows.slice(page * size, page * size + size);
  root.querySelector('tbody').innerHTML = slice.map(r => '<tr>' + r.map(v => `<td>${cell(v)}</td>`).join('') + '</tr>').join('');
  const first = rows.length ? page * size + 1 : 0, last = page * size + slice.length;
  root.querySelector('.info').textContent =
    `rows ${first}–${last} of ${rows.length.toLocaleString()} loaded` + (total === null ? ' (total unknown)' : ` · ${total.toLocaleString()} in table`);
  root.querySelector('.pg').textContent = `page ${page + 1} / ${pages}`;
  root.querySelector('.prev').disabled = page === 0;
  root.querySelector('.next').disabled = page >= pages - 1;
  root.querySelector('.wrap').scrollTop = 0;
}
root.querySelector('thead tr').innerHTML = cols.map(([n, short, full]) =>
  `<th title="${esc(n + ': ' + full)}">${esc(n)}<div class="t">${esc(short)}</div></th>`).join('');
root.querySelector('.prev').onclick = () => { page--; render(); };
root.querySelector('.next').onclick = () => { page++; render(); };
root.querySelector('.size').onchange = e => { size = +e.target.value; page = 0; render(); };
render();
"""

_TABLE_CSS = """
.dp{font:13px system-ui,sans-serif;color:inherit}
.dp .bar{display:flex;gap:8px;align-items:center;margin:2px 0 4px;flex-wrap:wrap}
.dp .bar .info{opacity:.7;margin-right:auto}
.dp button,.dp select{font:inherit;padding:1px 8px}
.dp .wrap{max-height:560px;overflow:auto;border:1px solid #8884}
.dp table{border-collapse:collapse}
.dp th,.dp td{border-bottom:1px solid #8883;padding:4px 8px;vertical-align:top;text-align:left;
  min-width:120px;max-width:380px;white-space:pre-wrap;overflow-wrap:anywhere}
.dp th{position:sticky;top:0;z-index:1;background:Canvas;white-space:normal}
.dp .t{font-weight:normal;opacity:.55;font-size:11px}
.dp .na{opacity:.4}
.dp .nested{font:11px ui-monospace,monospace}
.dp details summary{cursor:pointer;list-style:none}
.dp details[open] summary{display:none}
.dp a{color:#3b82f6}
"""

def _short_type(t):
    """'List(Struct(id: Str, ...))' -> 'list<struct>': short enough for a header."""
    s = repr(t)
    m = re.match(r"(\w+)\((\w+)", s)
    if not m:
        return s.lower()
    return f"{m.group(1).lower()}<{m.group(2).lower()}>" if m.group(1) in ("List", "Array") else m.group(1).lower()

def table_html(df, rows=200):
    """A dpyr table as HTML: loads up to `rows` rows, shows them 25 per page."""
    from dpyr.materialize import preview
    shown, total = preview(df.plan, rows)
    data = {
        "cols": [[c, _short_type(t), repr(t)] for c, t in df.schema.items()],
        "rows": [list(r) for r in shown.iter_rows()],
        "total": total,
    }
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return f"""<style>{_TABLE_CSS}</style><div class="dp">
<div class="bar"><span class="info"></span>
<button class="prev">‹ prev</button><span class="pg"></span><button class="next">next ›</button>
<select class="size"><option>10</option><option selected>25</option><option>50</option><option>100</option></select></div>
<div class="wrap"><table><thead><tr></tr></thead><tbody></tbody></table></div>
<script type="application/json">{payload}</script></div><script>{_TABLE_JS}</script>"""

class html_view:
    """Show a dpyr table as HTML. `rows` = how many rows to load (paged 25 at a time)."""
    def __init__(self, df, rows=200): self.df, self.rows = df, rows
    def _repr_html_(self): return table_html(self.df, self.rows)

def _show_tables_as_html():
    """Wrap the kernel's 'show the last value' step so dpyr tables become HTML."""
    show = next(f for f in gc.get_objects()
                if isinstance(f, types.FunctionType) and f.__name__ == "_show_value"
                and f.__globals__.get("__name__") == "__main__")
    original = getattr(show, "_original", show)
    def patched(value):
        if isinstance(value, DFrame) and value._repr_html_() is not None:
            return original(html_view(value))
        return original(value)
    patched._original = original
    show.__globals__["_show_value"] = patched

_show_tables_as_html()
```


# Explore OpenAlex: papers, abstracts, authors, and citations

Run the cells from top to bottom. The data is on **lambda**, in
`/mnt/bulk/openalex/works/`, and is still downloading.

Everything uses **dpyr**: `read`, `filter`, `mutate`, `count`, and friends.
No model is called unless you turn on the last cell.

## 1. Load

```python
import json
from dpyr import read, col, desc, n, vectorize

# Every finished file. Unfinished downloads end in .part, so they are skipped.
# Rerun this cell to include files that finished since.
works = read("/mnt/bulk/openalex/works/*/*.parquet")
works
```

<iframe class="rat-output" src="../_assets/generated/d0ec69f962c3.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

```py
works.columns
```

```output
['id', 'doi', 'title', 'display_name', 'ids', 'indexed_in', 'publication_date', 'publication_year', 'language', 'type', 'authorships', 'authors_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'primary_topic', 'topics', 'keywords', 'concepts', 'locations', 'locations_count', 'primary_location', 'best_oa_location', 'sustainable_development_goals', 'awards', 'funders', 'institutions', 'countries_distinct_count', 'institutions_distinct_count', 'open_access', 'is_paratext', 'is_retracted', 'is_xpac', 'biblio', 'referenced_works', 'referenced_works_count', 'related_works', 'abstract_inverted_index', 'cited_by_count', 'counts_by_year', 'apc_list', 'apc_paid', 'fwci', 'citation_normalized_percentile', 'cited_by_percentile_year', 'mesh', 'has_content', 'has_fulltext', 'created_date', 'updated_date']
```

```py
works.schema
```

```output
{'id': Str, 'doi': Str, 'title': Str, 'display_name': Str, 'ids': List(Struct(key: Str, value: Str)), 'indexed_in': List(Str), 'publication_date': Date, 'publication_year': Int64, 'language': Str, 'type': Str, 'authorships': List(Struct(author: Struct(display_name: Str, id: Str, orcid: Str), author_position: Str, affiliations: List(Struct(institution_ids: List(Str), raw_affiliation_string: Str)), countries: List(Str), raw_author_name: Str, raw_orcid: Str, is_corresponding: Bool, raw_affiliation_strings: List(Str), institutions: List(Struct(country_code: Str, display_name: Str, id: Str, lineage: List(Str), ror: Str, type: Str)))), 'authors_count': Int64, 'corresponding_author_ids': List(Str), 'corresponding_institution_ids': List(Str), 'primary_topic': Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str)), 'topics': List(Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str))), 'keywords': List(Struct(id: Str, display_name: Str, score: Float64)), 'concepts': List(Struct(id: Str, wikidata: Str, display_name: Str, level: Int64, score: Float64)), 'locations': List(Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool)), 'locations_count': Int64, 'primary_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'best_oa_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'sustainable_development_goals': List(Struct(id: Str, display_name: Str, score: Float64)), 'awards': List(Struct(id: Str, display_name: Str, funder_award_id: Str, funder_id: Str, funder_display_name: Str)), 'funders': List(Struct(id: Str, display_name: Str, ror: Str)), 'institutions': List(Struct(id: Str, display_name: Str, ror: Str, country_code: Str, type: Str, lineage: List(Str))), 'countries_distinct_count': Int64, 'institutions_distinct_count': Int64, 'open_access': Struct(is_oa: Bool, oa_status: Str, any_repository_has_fulltext: Bool, oa_url: Str), 'is_paratext': Bool, 'is_retracted': Bool, 'is_xpac': Bool, 'biblio': Struct(volume: Str, issue: Str, first_page: Str, last_page: Str), 'referenced_works': List(Str), 'referenced_works_count': Int64, 'related_works': List(Str), 'abstract_inverted_index': Str, 'cited_by_count': Int64, 'counts_by_year': List(Struct(year: Int64, cited_by_count: Int64)), 'apc_list': Struct(value: Int64, currency: Str, value_usd: Int64), 'apc_paid': Struct(value: Float64, currency: Str, value_usd: Float64), 'fwci': Float64, 'citation_normalized_percentile': Struct(value: Float64, is_in_top_1_percent: Bool, is_in_top_10_percent: Bool), 'cited_by_percentile_year': Struct(min: Int64, max: Int64), 'mesh': List(Struct(descriptor_ui: Str, descriptor_name: Str, qualifier_ui: Str, qualifier_name: Str, is_major_topic: Bool)), 'has_content': Struct(pdf: Bool, grobid_xml: Bool), 'has_fulltext': Bool, 'created_date': Datetime, 'updated_date': Datetime}
```

```py
works.select(col.title, col.doi, col.publication_year)
```

<iframe class="rat-output" src="../_assets/generated/c766f1db2608.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

Show tables as HTML: run this once, and every cell that ends with a dpyr
table shows it as a paged HTML table: 25 rows per page (10/25/50/100), full
text wrapped, long cells folded (click to open), links clickable, scrolls both
ways. It loads the first 200 rows; `html_view(df, rows=1000)` loads more.
Restarting the kernel turns it off; rerun to turn it back on.



## 2. Take a small sample

Searching all works reads every file, which takes a while. To explore, keep the
first 20,000 works in memory. They come from the first files, so this is
**not a random sample**.

```python
# English articles from 2020 on that list their references.
# Filtering ALL works first is too slow (>5 min), so look at the first 200,000 only.
sample = (works.slice_head(200_000)
          .filter(col.type == "article",
                  col.language == "en", col.referenced_works_count > 0)
          .persist())
sample.shape
```

```output
(3401, 49)
```

## 3. A readable table of papers

OpenAlex stores each abstract as words with their positions. This small function
puts the words back in order. `@vectorize` lets dpyr use it on a whole column.

```python
@vectorize
def abstract_text(inverted_index: str) -> str:
    if not inverted_index:
        return None
    index = json.loads(inverted_index)
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)

papers = (
    sample
    .unnest_wider(col.primary_topic, names_sep="_")
    .mutate(abstract=abstract_text(col.abstract_inverted_index))
    .select(col.id, col.title, col.publication_year, col.type,
            col.cited_by_count, col.primary_topic_display_name, col.abstract)
    .rename(year=col.publication_year, citations=col.cited_by_count,
            topic=col.primary_topic_display_name)
)
papers
```

<iframe class="rat-output" src="../_assets/generated/d84cbf30abf1.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 3b. A tidy content table

`content_table(works)` turns raw works into a flat table: one row per
publication, one plain value per cell, no lists or records. It keeps what a
publisher's page shows to a reader without access: title, authors,
affiliations, venue, volume/issue/pages, date, abstract, keywords, topic, open
access and license, funding, citation counts, and the list of references.

Lists (authors, keywords, references…) are joined with ` | `. A `|` inside a
value becomes `¦`, so `text.split(" | ")` always gets the items back.
`author_orcids` lines up with `authors` one for one (`-` when there is no ORCID).

The column names are generic (`title`, `authors`, `date`, `venue`, `abstract`,
`url`…), so books, preprints or blog posts can fill the same table later.

It runs in Python one row at a time: fine for a sample, too slow for all of `works`.

```python
from dataclasses import dataclass
from dpyr import vectorize

SEP = " | "   # between items in a list column; a "|" inside an item becomes "¦"

def _join(items):
    """Several values -> one string. None/empty items are dropped; None if nothing left."""
    out = [str(x).replace("|", "¦").strip() for x in items or [] if x not in (None, "")]
    return SEP.join(out) if out else None

def _get(d, *path):
    """d['a']['b'] that returns None instead of failing on a missing level."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d

@dataclass
class Content:
    url: str | None
    pdf_url: str | None
    venue: str | None
    publisher: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    authors: str | None
    author_orcids: str | None
    corresponding_authors: str | None
    institutions: str | None
    countries: str | None
    abstract: str | None
    keywords: str | None
    topic: str | None
    subfield: str | None
    field: str | None
    domain: str | None
    is_open_access: bool | None
    open_access_status: str | None
    license: str | None
    funders: str | None
    grants: str | None
    mesh_terms: str | None
    references: str | None

@vectorize
def _flatten_work(authorships: list, primary_location: dict, best_oa_location: dict,
                  biblio: dict, primary_topic: dict, keywords: list, open_access: dict,
                  funders: list, awards: list, mesh: list, referenced_works: list,
                  abstract_inverted_index: str) -> Content:
    authorships = authorships or []
    loc = primary_location or {}
    source = loc.get("source") or {}
    oa = best_oa_location or {}
    first, last = _get(biblio, "first_page"), _get(biblio, "last_page")
    pages = f"{first}–{last}" if first and last and first != last else first
    insts = {}
    for a in authorships:
        for inst in a.get("institutions") or []:
            insts.setdefault(inst.get("display_name"), None)
    abstract = None
    if abstract_inverted_index:
        index = json.loads(abstract_inverted_index) or {}
        words = sorted((pos, w) for w, ps in index.items() for pos in ps)
        abstract = " ".join(w for _, w in words) or None
    return Content(
        url=loc.get("landing_page_url"),
        pdf_url=oa.get("pdf_url") or loc.get("pdf_url"),
        venue=source.get("display_name") or loc.get("raw_source_name"),
        publisher=source.get("host_organization_name"),
        issn=source.get("issn_l"),
        volume=_get(biblio, "volume"),
        issue=_get(biblio, "issue"),
        pages=pages,
        authors=_join(_get(a, "author", "display_name") or a.get("raw_author_name") for a in authorships),
        author_orcids=_join(_get(a, "author", "orcid") or "-" for a in authorships),
        corresponding_authors=_join(_get(a, "author", "display_name") for a in authorships if a.get("is_corresponding")),
        institutions=_join(insts),
        countries=_join(dict.fromkeys(c for a in authorships for c in a.get("countries") or [])),
        abstract=abstract,
        keywords=_join(k.get("display_name") for k in keywords or []),
        topic=_get(primary_topic, "display_name"),
        subfield=_get(primary_topic, "subfield", "display_name"),
        field=_get(primary_topic, "field", "display_name"),
        domain=_get(primary_topic, "domain", "display_name"),
        is_open_access=_get(open_access, "is_oa"),
        open_access_status=_get(open_access, "oa_status"),
        license=oa.get("license") or loc.get("license"),
        funders=_join(f.get("display_name") for f in funders or []),
        grants=_join(f"{a.get('funder_display_name')}: {a.get('funder_award_id')}" for a in awards or []
                     if a.get("funder_award_id")),
        mesh_terms=_join(dict.fromkeys(m.get("descriptor_name") for m in mesh or [])),
        references=_join(referenced_works),
    )

def content_table(raw):
    """Raw OpenAlex works -> one flat row per publication (no lists, no records)."""
    return (
        raw
        .mutate(content=_flatten_work(
            col.authorships, col.primary_location, col.best_oa_location, col.biblio,
            col.primary_topic, col.keywords, col.open_access, col.funders, col.awards,
            col.mesh, col.referenced_works, col.abstract_inverted_index))
        .select(col.id, col.doi, col.title, col.type, col.language, col.publication_date,
                col.publication_year, col.cited_by_count, col.referenced_works_count,
                col.is_retracted, col.updated_date, col.content)
        .unnest_wider(col.content)
        .select(col.id, col.doi, col.url, col.pdf_url,
                col.title, col.type, col.language, col.publication_date, col.publication_year,
                col.authors, col.author_orcids, col.corresponding_authors, col.institutions, col.countries,
                col.venue, col.publisher, col.issn, col.volume, col.issue, col.pages,
                col.abstract, col.keywords, col.topic, col.subfield, col.field, col.domain, col.mesh_terms,
                col.is_open_access, col.open_access_status, col.license,
                col.funders, col.grants,
                col.cited_by_count, col.referenced_works_count, col.references,
                col.is_retracted, col.updated_date)
        .rename(date=col.publication_date, year=col.publication_year,
                citations=col.cited_by_count, references_count=col.referenced_works_count)
    )

content = content_table(sample)
content
```

<iframe class="rat-output" src="../_assets/generated/9b973bd3ab1e.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 4. Find papers

Change the words and the year. `(?i)` ignores upper/lower case; `|` means OR.

```python
matches = (
    content
    .filter(col.field.str_detect(r"Agricultural and Biological Sciences"), col.year >= 2000)
    .arrange(desc(col.citations))
)
matches
```

<iframe class="rat-output" src="../_assets/generated/7a1bc4452c5c.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 5. Read one abstract

The most-cited paper in the sample that has an abstract.

```python
paper = papers.filter(~col.abstract.is_na()).slice_max(col.citations, 1)
print(paper.pull(col.title)[0])
print()
print(paper.pull(col.abstract)[0])
```

## 6. Authors

Each paper holds a list of authors. `unnest_longer` gives each author its own
row; `unnest_wider` turns their details into columns.

```python
authors = (
    sample
    .select(col.id, col.authorships)
    .unnest_longer(col.authorships)
    .unnest_wider(col.authorships)
    .unnest_wider(col.author, names_sep="_")
    .select(col.id, col.author_display_name, col.author_position)
)
authors.count(col.author_display_name).arrange(desc(col.n))
```

## 7. Keywords

The same idea. These keywords are added by OpenAlex, not chosen by the authors.

```python
keywords = (
    sample
    .select(col.id, col.keywords)
    .unnest_longer(col.keywords)
    .unnest_wider(col.keywords, names_sep="_")
)
keywords.count(col.keywords_display_name).arrange(desc(col.n))
```

## 8. Citations

One row per link: paper `id` cites paper `cited_id`. Some links are missing in
OpenAlex. A cited paper is often outside the sample, so its title shows as `null`.

```python
citations = (
    sample
    .select(col.id, col.referenced_works)
    .unnest_longer(col.referenced_works)
    .rename(cited_id=col.referenced_works)
)

# Which papers does the sample cite most?
(
    citations
    .count(col.cited_id)
    .left_join(papers.select(col.id, col.title).rename(cited_id=col.id), on=col.cited_id)
    .arrange(desc(col.n))
)
```

## 9. Save a reading list

Writes your search results from section 4 to a CSV file in this project.

```python
matches.select(col.id, col.title, col.year, col.citations).write("research/openalex/reading-list.csv")
```

## 10. Optional: explain papers with functai

A functai function works inside `mutate`, like `abstract_text` above. Each row is
one model call, so this sends **3 abstracts** to the model and may cost money.
It is off until you set `RUN_AI = True` and choose a model.

```python
import functai
from functai import ai

@ai
def explain_paper(title: str, abstract: str) -> str:
    """Explain this paper's question, approach, and findings in three short sentences.
    Use only the title and abstract. Do not invent missing details.
    """
    ...

RUN_AI = False
MODEL = None   # for example "claude-haiku-4-5"

if RUN_AI:
    with functai.configure(lm=MODEL):
        explained = (
            papers
            .filter(~col.abstract.is_na())
            .slice_head(3)
            .mutate(explanation=explain_paper(col.title, col.abstract))
            .select(col.title, col.explanation)
            .collect()
        )
    print(explained)
else:
    print("Skipped. Set RUN_AI = True and MODEL to try it.")
```
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../r_ports_to_py/dpyr"
      - "-e ../functai/python"
      - "-e ../lmcc/python"
---

```python
import gc, html, json, re, types
from dpyr import DFrame

_TABLE_JS = r"""
const root = document.currentScript.previousElementSibling;
const {cols, rows, total} = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
const URL_RE = /(https?:\/\/[^\s<>"']+)/g, LONG = 300;
let page = 0, size = 25;
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify = s => esc(s).replace(URL_RE, u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
function cell(v) {
  if (v === null) return '<span class="na">null</span>';
  const nested = typeof v === 'object';
  const s = nested ? JSON.stringify(v) : String(v);
  const cls = nested ? ' class="nested"' : '';
  if (s.length <= LONG) return `<div${cls}>${linkify(s)}</div>`;
  return `<details${cls}><summary>${linkify(s.slice(0, LONG))}…</summary>${linkify(s)}</details>`;
}
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages - 1);
  const slice = rows.slice(page * size, page * size + size);
  root.querySelector('tbody').innerHTML = slice.map(r => '<tr>' + r.map(v => `<td>${cell(v)}</td>`).join('') + '</tr>').join('');
  const first = rows.length ? page * size + 1 : 0, last = page * size + slice.length;
  root.querySelector('.info').textContent =
    `rows ${first}–${last} of ${rows.length.toLocaleString()} loaded` + (total === null ? ' (total unknown)' : ` · ${total.toLocaleString()} in table`);
  root.querySelector('.pg').textContent = `page ${page + 1} / ${pages}`;
  root.querySelector('.prev').disabled = page === 0;
  root.querySelector('.next').disabled = page >= pages - 1;
  root.querySelector('.wrap').scrollTop = 0;
}
root.querySelector('thead tr').innerHTML = cols.map(([n, short, full]) =>
  `<th title="${esc(n + ': ' + full)}">${esc(n)}<div class="t">${esc(short)}</div></th>`).join('');
root.querySelector('.prev').onclick = () => { page--; render(); };
root.querySelector('.next').onclick = () => { page++; render(); };
root.querySelector('.size').onchange = e => { size = +e.target.value; page = 0; render(); };
render();
"""

_TABLE_CSS = """
.dp{font:13px system-ui,sans-serif;color:inherit}
.dp .bar{display:flex;gap:8px;align-items:center;margin:2px 0 4px;flex-wrap:wrap}
.dp .bar .info{opacity:.7;margin-right:auto}
.dp button,.dp select{font:inherit;padding:1px 8px}
.dp .wrap{max-height:560px;overflow:auto;border:1px solid #8884}
.dp table{border-collapse:collapse}
.dp th,.dp td{border-bottom:1px solid #8883;padding:4px 8px;vertical-align:top;text-align:left;
  min-width:120px;max-width:380px;white-space:pre-wrap;overflow-wrap:anywhere}
.dp th{position:sticky;top:0;z-index:1;background:Canvas;white-space:normal}
.dp .t{font-weight:normal;opacity:.55;font-size:11px}
.dp .na{opacity:.4}
.dp .nested{font:11px ui-monospace,monospace}
.dp details summary{cursor:pointer;list-style:none}
.dp details[open] summary{display:none}
.dp a{color:#3b82f6}
"""

def _short_type(t):
    """'List(Struct(id: Str, ...))' -> 'list<struct>': short enough for a header."""
    s = repr(t)
    m = re.match(r"(\w+)\((\w+)", s)
    if not m:
        return s.lower()
    return f"{m.group(1).lower()}<{m.group(2).lower()}>" if m.group(1) in ("List", "Array") else m.group(1).lower()

def table_html(df, rows=200):
    """A dpyr table as HTML: loads up to `rows` rows, shows them 25 per page."""
    from dpyr.materialize import preview
    shown, total = preview(df.plan, rows)
    data = {
        "cols": [[c, _short_type(t), repr(t)] for c, t in df.schema.items()],
        "rows": [list(r) for r in shown.iter_rows()],
        "total": total,
    }
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return f"""<style>{_TABLE_CSS}</style><div class="dp">
<div class="bar"><span class="info"></span>
<button class="prev">‹ prev</button><span class="pg"></span><button class="next">next ›</button>
<select class="size"><option>10</option><option selected>25</option><option>50</option><option>100</option></select></div>
<div class="wrap"><table><thead><tr></tr></thead><tbody></tbody></table></div>
<script type="application/json">{payload}</script></div><script>{_TABLE_JS}</script>"""

class html_view:
    """Show a dpyr table as HTML. `rows` = how many rows to load (paged 25 at a time)."""
    def __init__(self, df, rows=200): self.df, self.rows = df, rows
    def _repr_html_(self): return table_html(self.df, self.rows)

def _show_tables_as_html():
    """Wrap the kernel's 'show the last value' step so dpyr tables become HTML."""
    show = next(f for f in gc.get_objects()
                if isinstance(f, types.FunctionType) and f.__name__ == "_show_value"
                and f.__globals__.get("__name__") == "__main__")
    original = getattr(show, "_original", show)
    def patched(value):
        if isinstance(value, DFrame) and value._repr_html_() is not None:
            return original(html_view(value))
        return original(value)
    patched._original = original
    show.__globals__["_show_value"] = patched

_show_tables_as_html()
```


# Explore OpenAlex: papers, abstracts, authors, and citations

Run the cells from top to bottom. The data is on **lambda**, in
`/mnt/bulk/openalex/works/`, and is still downloading.

Everything uses **dpyr**: `read`, `filter`, `mutate`, `count`, and friends.
No model is called unless you turn on the last cell.

## 1. Load

```python
import json
from dpyr import read, col, desc, n, vectorize

# Every finished file. Unfinished downloads end in .part, so they are skipped.
# Rerun this cell to include files that finished since.
works = read("/mnt/bulk/openalex/works/*/*.parquet")
works
```

<iframe class="rat-output" src="../_assets/generated/d0ec69f962c3.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

```py
works.columns
```

```output
['id', 'doi', 'title', 'display_name', 'ids', 'indexed_in', 'publication_date', 'publication_year', 'language', 'type', 'authorships', 'authors_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'primary_topic', 'topics', 'keywords', 'concepts', 'locations', 'locations_count', 'primary_location', 'best_oa_location', 'sustainable_development_goals', 'awards', 'funders', 'institutions', 'countries_distinct_count', 'institutions_distinct_count', 'open_access', 'is_paratext', 'is_retracted', 'is_xpac', 'biblio', 'referenced_works', 'referenced_works_count', 'related_works', 'abstract_inverted_index', 'cited_by_count', 'counts_by_year', 'apc_list', 'apc_paid', 'fwci', 'citation_normalized_percentile', 'cited_by_percentile_year', 'mesh', 'has_content', 'has_fulltext', 'created_date', 'updated_date']
```

```py
works.schema
```

```output
{'id': Str, 'doi': Str, 'title': Str, 'display_name': Str, 'ids': List(Struct(key: Str, value: Str)), 'indexed_in': List(Str), 'publication_date': Date, 'publication_year': Int64, 'language': Str, 'type': Str, 'authorships': List(Struct(author: Struct(display_name: Str, id: Str, orcid: Str), author_position: Str, affiliations: List(Struct(institution_ids: List(Str), raw_affiliation_string: Str)), countries: List(Str), raw_author_name: Str, raw_orcid: Str, is_corresponding: Bool, raw_affiliation_strings: List(Str), institutions: List(Struct(country_code: Str, display_name: Str, id: Str, lineage: List(Str), ror: Str, type: Str)))), 'authors_count': Int64, 'corresponding_author_ids': List(Str), 'corresponding_institution_ids': List(Str), 'primary_topic': Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str)), 'topics': List(Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str))), 'keywords': List(Struct(id: Str, display_name: Str, score: Float64)), 'concepts': List(Struct(id: Str, wikidata: Str, display_name: Str, level: Int64, score: Float64)), 'locations': List(Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool)), 'locations_count': Int64, 'primary_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'best_oa_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'sustainable_development_goals': List(Struct(id: Str, display_name: Str, score: Float64)), 'awards': List(Struct(id: Str, display_name: Str, funder_award_id: Str, funder_id: Str, funder_display_name: Str)), 'funders': List(Struct(id: Str, display_name: Str, ror: Str)), 'institutions': List(Struct(id: Str, display_name: Str, ror: Str, country_code: Str, type: Str, lineage: List(Str))), 'countries_distinct_count': Int64, 'institutions_distinct_count': Int64, 'open_access': Struct(is_oa: Bool, oa_status: Str, any_repository_has_fulltext: Bool, oa_url: Str), 'is_paratext': Bool, 'is_retracted': Bool, 'is_xpac': Bool, 'biblio': Struct(volume: Str, issue: Str, first_page: Str, last_page: Str), 'referenced_works': List(Str), 'referenced_works_count': Int64, 'related_works': List(Str), 'abstract_inverted_index': Str, 'cited_by_count': Int64, 'counts_by_year': List(Struct(year: Int64, cited_by_count: Int64)), 'apc_list': Struct(value: Int64, currency: Str, value_usd: Int64), 'apc_paid': Struct(value: Float64, currency: Str, value_usd: Float64), 'fwci': Float64, 'citation_normalized_percentile': Struct(value: Float64, is_in_top_1_percent: Bool, is_in_top_10_percent: Bool), 'cited_by_percentile_year': Struct(min: Int64, max: Int64), 'mesh': List(Struct(descriptor_ui: Str, descriptor_name: Str, qualifier_ui: Str, qualifier_name: Str, is_major_topic: Bool)), 'has_content': Struct(pdf: Bool, grobid_xml: Bool), 'has_fulltext': Bool, 'created_date': Datetime, 'updated_date': Datetime}
```

```py
works.select(col.title, col.doi, col.publication_year)
```

<iframe class="rat-output" src="../_assets/generated/c766f1db2608.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

Show tables as HTML: run this once, and every cell that ends with a dpyr
table shows it as a paged HTML table: 25 rows per page (10/25/50/100), full
text wrapped, long cells folded (click to open), links clickable, scrolls both
ways. It loads the first 200 rows; `html_view(df, rows=1000)` loads more.
Restarting the kernel turns it off; rerun to turn it back on.



## 2. Take a small sample

Searching all works reads every file, which takes a while. To explore, keep the
first 20,000 works in memory. They come from the first files, so this is
**not a random sample**.

```python
# English articles from 2020 on that list their references.
# Filtering ALL works first is too slow (>5 min), so look at the first 200,000 only.
sample = (works.slice_head(200_000)
          .filter(col.type == "article",
                  col.language == "en", col.referenced_works_count > 0)
          .persist())
sample.shape
```

```output
(3401, 49)
```

## 3. A readable table of papers

OpenAlex stores each abstract as words with their positions. This small function
puts the words back in order. `@vectorize` lets dpyr use it on a whole column.

```python
@vectorize
def abstract_text(inverted_index: str) -> str:
    if not inverted_index:
        return None
    index = json.loads(inverted_index)
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)

papers = (
    sample
    .unnest_wider(col.primary_topic, names_sep="_")
    .mutate(abstract=abstract_text(col.abstract_inverted_index))
    .select(col.id, col.title, col.publication_year, col.type,
            col.cited_by_count, col.primary_topic_display_name, col.abstract)
    .rename(year=col.publication_year, citations=col.cited_by_count,
            topic=col.primary_topic_display_name)
)
papers
```

<iframe class="rat-output" src="../_assets/generated/d84cbf30abf1.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 3b. A tidy content table

`content_table(works)` turns raw works into a flat table: one row per
publication, one plain value per cell, no lists or records. It keeps what a
publisher's page shows to a reader without access: title, authors,
affiliations, venue, volume/issue/pages, date, abstract, keywords, topic, open
access and license, funding, citation counts, and the list of references.

Lists (authors, keywords, references…) are joined with ` | `. A `|` inside a
value becomes `¦`, so `text.split(" | ")` always gets the items back.
`author_orcids` lines up with `authors` one for one (`-` when there is no ORCID).

The column names are generic (`title`, `authors`, `date`, `venue`, `abstract`,
`url`…), so books, preprints or blog posts can fill the same table later.

It runs in Python one row at a time: fine for a sample, too slow for all of `works`.

```python
from dataclasses import dataclass
from dpyr import vectorize

SEP = " | "   # between items in a list column; a "|" inside an item becomes "¦"

def _join(items):
    """Several values -> one string. None/empty items are dropped; None if nothing left."""
    out = [str(x).replace("|", "¦").strip() for x in items or [] if x not in (None, "")]
    return SEP.join(out) if out else None

def _get(d, *path):
    """d['a']['b'] that returns None instead of failing on a missing level."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d

@dataclass
class Content:
    url: str | None
    pdf_url: str | None
    venue: str | None
    publisher: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    authors: str | None
    author_orcids: str | None
    corresponding_authors: str | None
    institutions: str | None
    countries: str | None
    abstract: str | None
    keywords: str | None
    topic: str | None
    subfield: str | None
    field: str | None
    domain: str | None
    is_open_access: bool | None
    open_access_status: str | None
    license: str | None
    funders: str | None
    grants: str | None
    mesh_terms: str | None
    references: str | None

@vectorize
def _flatten_work(authorships: list, primary_location: dict, best_oa_location: dict,
                  biblio: dict, primary_topic: dict, keywords: list, open_access: dict,
                  funders: list, awards: list, mesh: list, referenced_works: list,
                  abstract_inverted_index: str) -> Content:
    authorships = authorships or []
    loc = primary_location or {}
    source = loc.get("source") or {}
    oa = best_oa_location or {}
    first, last = _get(biblio, "first_page"), _get(biblio, "last_page")
    pages = f"{first}–{last}" if first and last and first != last else first
    insts = {}
    for a in authorships:
        for inst in a.get("institutions") or []:
            insts.setdefault(inst.get("display_name"), None)
    abstract = None
    if abstract_inverted_index:
        index = json.loads(abstract_inverted_index) or {}
        words = sorted((pos, w) for w, ps in index.items() for pos in ps)
        abstract = " ".join(w for _, w in words) or None
    return Content(
        url=loc.get("landing_page_url"),
        pdf_url=oa.get("pdf_url") or loc.get("pdf_url"),
        venue=source.get("display_name") or loc.get("raw_source_name"),
        publisher=source.get("host_organization_name"),
        issn=source.get("issn_l"),
        volume=_get(biblio, "volume"),
        issue=_get(biblio, "issue"),
        pages=pages,
        authors=_join(_get(a, "author", "display_name") or a.get("raw_author_name") for a in authorships),
        author_orcids=_join(_get(a, "author", "orcid") or "-" for a in authorships),
        corresponding_authors=_join(_get(a, "author", "display_name") for a in authorships if a.get("is_corresponding")),
        institutions=_join(insts),
        countries=_join(dict.fromkeys(c for a in authorships for c in a.get("countries") or [])),
        abstract=abstract,
        keywords=_join(k.get("display_name") for k in keywords or []),
        topic=_get(primary_topic, "display_name"),
        subfield=_get(primary_topic, "subfield", "display_name"),
        field=_get(primary_topic, "field", "display_name"),
        domain=_get(primary_topic, "domain", "display_name"),
        is_open_access=_get(open_access, "is_oa"),
        open_access_status=_get(open_access, "oa_status"),
        license=oa.get("license") or loc.get("license"),
        funders=_join(f.get("display_name") for f in funders or []),
        grants=_join(f"{a.get('funder_display_name')}: {a.get('funder_award_id')}" for a in awards or []
                     if a.get("funder_award_id")),
        mesh_terms=_join(dict.fromkeys(m.get("descriptor_name") for m in mesh or [])),
        references=_join(referenced_works),
    )

def content_table(raw):
    """Raw OpenAlex works -> one flat row per publication (no lists, no records)."""
    return (
        raw
        .mutate(content=_flatten_work(
            col.authorships, col.primary_location, col.best_oa_location, col.biblio,
            col.primary_topic, col.keywords, col.open_access, col.funders, col.awards,
            col.mesh, col.referenced_works, col.abstract_inverted_index))
        .select(col.id, col.doi, col.title, col.type, col.language, col.publication_date,
                col.publication_year, col.cited_by_count, col.referenced_works_count,
                col.is_retracted, col.updated_date, col.content)
        .unnest_wider(col.content)
        .select(col.id, col.doi, col.url, col.pdf_url,
                col.title, col.type, col.language, col.publication_date, col.publication_year,
                col.authors, col.author_orcids, col.corresponding_authors, col.institutions, col.countries,
                col.venue, col.publisher, col.issn, col.volume, col.issue, col.pages,
                col.abstract, col.keywords, col.topic, col.subfield, col.field, col.domain, col.mesh_terms,
                col.is_open_access, col.open_access_status, col.license,
                col.funders, col.grants,
                col.cited_by_count, col.referenced_works_count, col.references,
                col.is_retracted, col.updated_date)
        .rename(date=col.publication_date, year=col.publication_year,
                citations=col.cited_by_count, references_count=col.referenced_works_count)
    )

content = content_table(sample)
content
```

<iframe class="rat-output" src="../_assets/generated/9b973bd3ab1e.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 4. Find papers

Change the words and the year. `(?i)` ignores upper/lower case; `|` means OR.

```python
matches = (
    content
    .filter(col.field.str_detect(r"Agricultural and Biological Sciences"), col.year >= 2000)
    .arrange(desc(col.citations))
)
matches
```

<iframe class="rat-output" src="../_assets/generated/7a1bc4452c5c.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 5. Read one abstract

The most-cited paper in the sample that has an abstract.

```python
paper = papers.filter(~col.abstract.is_na()).slice_max(col.citations, 1)
print(paper.pull(col.title)[0])
print()
print(paper.pull(col.abstract)[0])
```

## 6. Authors

Each paper holds a list of authors. `unnest_longer` gives each author its own
row; `unnest_wider` turns their details into columns.

```python
authors = (
    sample
    .select(col.id, col.authorships)
    .unnest_longer(col.authorships)
    .unnest_wider(col.authorships)
    .unnest_wider(col.author, names_sep="_")
    .select(col.id, col.author_display_name, col.author_position)
)
authors.count(col.author_display_name).arrange(desc(col.n))
```

## 7. Keywords

The same idea. These keywords are added by OpenAlex, not chosen by the authors.

```python
keywords = (
    sample
    .select(col.id, col.keywords)
    .unnest_longer(col.keywords)
    .unnest_wider(col.keywords, names_sep="_")
)
keywords.count(col.keywords_display_name).arrange(desc(col.n))
```

## 8. Citations

One row per link: paper `id` cites paper `cited_id`. Some links are missing in
OpenAlex. A cited paper is often outside the sample, so its title shows as `null`.

```python
citations = (
    sample
    .select(col.id, col.referenced_works)
    .unnest_longer(col.referenced_works)
    .rename(cited_id=col.referenced_works)
)

# Which papers does the sample cite most?
(
    citations
    .count(col.cited_id)
    .left_join(papers.select(col.id, col.title).rename(cited_id=col.id), on=col.cited_id)
    .arrange(desc(col.n))
)
```

## 9. Save a reading list

Writes your search results from section 4 to a CSV file in this project.

```python
matches.select(col.id, col.title, col.year, col.citations).write("research/openalex/reading-list.csv")
```

## 10. Optional: explain papers with functai

A functai function works inside `mutate`, like `abstract_text` above. Each row is
one model call, so this sends **3 abstracts** to the model and may cost money.
It is off until you set `RUN_AI = True` and choose a model.

```python
import functai
from functai import ai

@ai
def explain_paper(title: str, abstract: str) -> str:
    """Explain this paper's question, approach, and findings in three short sentences.
    Use only the title and abstract. Do not invent missing details.
    """
    ...

RUN_AI = False
MODEL = None   # for example "claude-haiku-4-5"

if RUN_AI:
    with functai.configure(lm=MODEL):
        explained = (
            papers
            .filter(~col.abstract.is_na())
            .slice_head(3)
            .mutate(explanation=explain_paper(col.title, col.abstract))
            .select(col.title, col.explanation)
            .collect()
        )
    print(explained)
else:
    print("Skipped. Set RUN_AI = True and MODEL to try it.")
```
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../r_ports_to_py/dpyr"
      - "-e ../functai/python"
      - "-e ../lmcc/python"
---

```python
import gc, html, json, re, types
from dpyr import DFrame

_TABLE_JS = r"""
const root = document.currentScript.previousElementSibling;
const {cols, rows, total} = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
const URL_RE = /(https?:\/\/[^\s<>"']+)/g, LONG = 300;
let page = 0, size = 25;
const esc = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify = s => esc(s).replace(URL_RE, u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
function cell(v) {
  if (v === null) return '<span class="na">null</span>';
  const nested = typeof v === 'object';
  const s = nested ? JSON.stringify(v) : String(v);
  const cls = nested ? ' class="nested"' : '';
  if (s.length <= LONG) return `<div${cls}>${linkify(s)}</div>`;
  return `<details${cls}><summary>${linkify(s.slice(0, LONG))}…</summary>${linkify(s)}</details>`;
}
function render() {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages - 1);
  const slice = rows.slice(page * size, page * size + size);
  root.querySelector('tbody').innerHTML = slice.map(r => '<tr>' + r.map(v => `<td>${cell(v)}</td>`).join('') + '</tr>').join('');
  const first = rows.length ? page * size + 1 : 0, last = page * size + slice.length;
  root.querySelector('.info').textContent =
    `rows ${first}–${last} of ${rows.length.toLocaleString()} loaded` + (total === null ? ' (total unknown)' : ` · ${total.toLocaleString()} in table`);
  root.querySelector('.pg').textContent = `page ${page + 1} / ${pages}`;
  root.querySelector('.prev').disabled = page === 0;
  root.querySelector('.next').disabled = page >= pages - 1;
  root.querySelector('.wrap').scrollTop = 0;
}
root.querySelector('thead tr').innerHTML = cols.map(([n, short, full]) =>
  `<th title="${esc(n + ': ' + full)}">${esc(n)}<div class="t">${esc(short)}</div></th>`).join('');
root.querySelector('.prev').onclick = () => { page--; render(); };
root.querySelector('.next').onclick = () => { page++; render(); };
root.querySelector('.size').onchange = e => { size = +e.target.value; page = 0; render(); };
render();
"""

_TABLE_CSS = """
.dp{font:13px system-ui,sans-serif;color:inherit}
.dp .bar{display:flex;gap:8px;align-items:center;margin:2px 0 4px;flex-wrap:wrap}
.dp .bar .info{opacity:.7;margin-right:auto}
.dp button,.dp select{font:inherit;padding:1px 8px}
.dp .wrap{max-height:560px;overflow:auto;border:1px solid #8884}
.dp table{border-collapse:collapse}
.dp th,.dp td{border-bottom:1px solid #8883;padding:4px 8px;vertical-align:top;text-align:left;
  min-width:120px;max-width:380px;white-space:pre-wrap;overflow-wrap:anywhere}
.dp th{position:sticky;top:0;z-index:1;background:Canvas;white-space:normal}
.dp .t{font-weight:normal;opacity:.55;font-size:11px}
.dp .na{opacity:.4}
.dp .nested{font:11px ui-monospace,monospace}
.dp details summary{cursor:pointer;list-style:none}
.dp details[open] summary{display:none}
.dp a{color:#3b82f6}
"""

def _short_type(t):
    """'List(Struct(id: Str, ...))' -> 'list<struct>': short enough for a header."""
    s = repr(t)
    m = re.match(r"(\w+)\((\w+)", s)
    if not m:
        return s.lower()
    return f"{m.group(1).lower()}<{m.group(2).lower()}>" if m.group(1) in ("List", "Array") else m.group(1).lower()

def table_html(df, rows=200):
    """A dpyr table as HTML: loads up to `rows` rows, shows them 25 per page."""
    from dpyr.materialize import preview
    shown, total = preview(df.plan, rows)
    data = {
        "cols": [[c, _short_type(t), repr(t)] for c, t in df.schema.items()],
        "rows": [list(r) for r in shown.iter_rows()],
        "total": total,
    }
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return f"""<style>{_TABLE_CSS}</style><div class="dp">
<div class="bar"><span class="info"></span>
<button class="prev">‹ prev</button><span class="pg"></span><button class="next">next ›</button>
<select class="size"><option>10</option><option selected>25</option><option>50</option><option>100</option></select></div>
<div class="wrap"><table><thead><tr></tr></thead><tbody></tbody></table></div>
<script type="application/json">{payload}</script></div><script>{_TABLE_JS}</script>"""

class html_view:
    """Show a dpyr table as HTML. `rows` = how many rows to load (paged 25 at a time)."""
    def __init__(self, df, rows=200): self.df, self.rows = df, rows
    def _repr_html_(self): return table_html(self.df, self.rows)

def _show_tables_as_html():
    """Wrap the kernel's 'show the last value' step so dpyr tables become HTML."""
    show = next(f for f in gc.get_objects()
                if isinstance(f, types.FunctionType) and f.__name__ == "_show_value"
                and f.__globals__.get("__name__") == "__main__")
    original = getattr(show, "_original", show)
    def patched(value):
        if isinstance(value, DFrame) and value._repr_html_() is not None:
            return original(html_view(value))
        return original(value)
    patched._original = original
    show.__globals__["_show_value"] = patched

_show_tables_as_html()
```


# Explore OpenAlex: papers, abstracts, authors, and citations

Run the cells from top to bottom. The data is on **lambda**, in
`/mnt/bulk/openalex/works/`, and is still downloading.

Everything uses **dpyr**: `read`, `filter`, `mutate`, `count`, and friends.
No model is called unless you turn on the last cell.

## 1. Load

```python
import json
from dpyr import read, col, desc, n, vectorize

# Every finished file. Unfinished downloads end in .part, so they are skipped.
# Rerun this cell to include files that finished since.
works = read("/mnt/bulk/openalex/works/*/*.parquet")
works
```

<iframe class="rat-output" src="../_assets/generated/d0ec69f962c3.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

```py
works.columns
```

```output
['id', 'doi', 'title', 'display_name', 'ids', 'indexed_in', 'publication_date', 'publication_year', 'language', 'type', 'authorships', 'authors_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'primary_topic', 'topics', 'keywords', 'concepts', 'locations', 'locations_count', 'primary_location', 'best_oa_location', 'sustainable_development_goals', 'awards', 'funders', 'institutions', 'countries_distinct_count', 'institutions_distinct_count', 'open_access', 'is_paratext', 'is_retracted', 'is_xpac', 'biblio', 'referenced_works', 'referenced_works_count', 'related_works', 'abstract_inverted_index', 'cited_by_count', 'counts_by_year', 'apc_list', 'apc_paid', 'fwci', 'citation_normalized_percentile', 'cited_by_percentile_year', 'mesh', 'has_content', 'has_fulltext', 'created_date', 'updated_date']
```

```py
works.schema
```

```output
{'id': Str, 'doi': Str, 'title': Str, 'display_name': Str, 'ids': List(Struct(key: Str, value: Str)), 'indexed_in': List(Str), 'publication_date': Date, 'publication_year': Int64, 'language': Str, 'type': Str, 'authorships': List(Struct(author: Struct(display_name: Str, id: Str, orcid: Str), author_position: Str, affiliations: List(Struct(institution_ids: List(Str), raw_affiliation_string: Str)), countries: List(Str), raw_author_name: Str, raw_orcid: Str, is_corresponding: Bool, raw_affiliation_strings: List(Str), institutions: List(Struct(country_code: Str, display_name: Str, id: Str, lineage: List(Str), ror: Str, type: Str)))), 'authors_count': Int64, 'corresponding_author_ids': List(Str), 'corresponding_institution_ids': List(Str), 'primary_topic': Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str)), 'topics': List(Struct(id: Str, display_name: Str, score: Float64, subfield: Struct(id: Str, display_name: Str), field: Struct(id: Str, display_name: Str), domain: Struct(id: Str, display_name: Str))), 'keywords': List(Struct(id: Str, display_name: Str, score: Float64)), 'concepts': List(Struct(id: Str, wikidata: Str, display_name: Str, level: Int64, score: Float64)), 'locations': List(Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool)), 'locations_count': Int64, 'primary_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'best_oa_location': Struct(id: Str, source: Struct(id: Str, display_name: Str, issn_l: Str, issn: List(Str), is_oa: Bool, is_in_doaj: Bool, is_core: Bool, listed_in: List(Str), host_organization: Str, host_organization_name: Str, host_organization_lineage: List(Str), host_organization_lineage_names: List(Str), type: Str), is_oa: Bool, is_published: Bool, landing_page_url: Str, pdf_url: Str, raw_source_name: Str, raw_type: Str, provenance: Str, license: Str, license_id: Str, version: Str, is_accepted: Bool), 'sustainable_development_goals': List(Struct(id: Str, display_name: Str, score: Float64)), 'awards': List(Struct(id: Str, display_name: Str, funder_award_id: Str, funder_id: Str, funder_display_name: Str)), 'funders': List(Struct(id: Str, display_name: Str, ror: Str)), 'institutions': List(Struct(id: Str, display_name: Str, ror: Str, country_code: Str, type: Str, lineage: List(Str))), 'countries_distinct_count': Int64, 'institutions_distinct_count': Int64, 'open_access': Struct(is_oa: Bool, oa_status: Str, any_repository_has_fulltext: Bool, oa_url: Str), 'is_paratext': Bool, 'is_retracted': Bool, 'is_xpac': Bool, 'biblio': Struct(volume: Str, issue: Str, first_page: Str, last_page: Str), 'referenced_works': List(Str), 'referenced_works_count': Int64, 'related_works': List(Str), 'abstract_inverted_index': Str, 'cited_by_count': Int64, 'counts_by_year': List(Struct(year: Int64, cited_by_count: Int64)), 'apc_list': Struct(value: Int64, currency: Str, value_usd: Int64), 'apc_paid': Struct(value: Float64, currency: Str, value_usd: Float64), 'fwci': Float64, 'citation_normalized_percentile': Struct(value: Float64, is_in_top_1_percent: Bool, is_in_top_10_percent: Bool), 'cited_by_percentile_year': Struct(min: Int64, max: Int64), 'mesh': List(Struct(descriptor_ui: Str, descriptor_name: Str, qualifier_ui: Str, qualifier_name: Str, is_major_topic: Bool)), 'has_content': Struct(pdf: Bool, grobid_xml: Bool), 'has_fulltext': Bool, 'created_date': Datetime, 'updated_date': Datetime}
```

```py
works.select(col.title, col.doi, col.publication_year)
```

<iframe class="rat-output" src="../_assets/generated/c766f1db2608.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

Show tables as HTML: run this once, and every cell that ends with a dpyr
table shows it as a paged HTML table: 25 rows per page (10/25/50/100), full
text wrapped, long cells folded (click to open), links clickable, scrolls both
ways. It loads the first 200 rows; `html_view(df, rows=1000)` loads more.
Restarting the kernel turns it off; rerun to turn it back on.



## 2. Take a small sample

Searching all works reads every file, which takes a while. To explore, keep the
first 20,000 works in memory. They come from the first files, so this is
**not a random sample**.

```python
# English articles from 2020 on that list their references.
# Filtering ALL works first is too slow (>5 min), so look at the first 200,000 only.
sample = (works.slice_head(200_000)
          .filter(col.type == "article",
                  col.language == "en", col.referenced_works_count > 0)
          .persist())
sample.shape
```

```output
(3401, 49)
```

## 3. A readable table of papers

OpenAlex stores each abstract as words with their positions. This small function
puts the words back in order. `@vectorize` lets dpyr use it on a whole column.

```python
@vectorize
def abstract_text(inverted_index: str) -> str:
    if not inverted_index:
        return None
    index = json.loads(inverted_index)
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)

papers = (
    sample
    .unnest_wider(col.primary_topic, names_sep="_")
    .mutate(abstract=abstract_text(col.abstract_inverted_index))
    .select(col.id, col.title, col.publication_year, col.type,
            col.cited_by_count, col.primary_topic_display_name, col.abstract)
    .rename(year=col.publication_year, citations=col.cited_by_count,
            topic=col.primary_topic_display_name)
)
papers
```

<iframe class="rat-output" src="../_assets/generated/d84cbf30abf1.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 3b. A tidy content table

`content_table(works)` turns raw works into a flat table: one row per
publication, one plain value per cell, no lists or records. It keeps what a
publisher's page shows to a reader without access: title, authors,
affiliations, venue, volume/issue/pages, date, abstract, keywords, topic, open
access and license, funding, citation counts, and the list of references.

Lists (authors, keywords, references…) are joined with ` | `. A `|` inside a
value becomes `¦`, so `text.split(" | ")` always gets the items back.
`author_orcids` lines up with `authors` one for one (`-` when there is no ORCID).

The column names are generic (`title`, `authors`, `date`, `venue`, `abstract`,
`url`…), so books, preprints or blog posts can fill the same table later.

It runs in Python one row at a time: fine for a sample, too slow for all of `works`.

```python
from dataclasses import dataclass
from dpyr import vectorize

SEP = " | "   # between items in a list column; a "|" inside an item becomes "¦"

def _join(items):
    """Several values -> one string. None/empty items are dropped; None if nothing left."""
    out = [str(x).replace("|", "¦").strip() for x in items or [] if x not in (None, "")]
    return SEP.join(out) if out else None

def _get(d, *path):
    """d['a']['b'] that returns None instead of failing on a missing level."""
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d

@dataclass
class Content:
    url: str | None
    pdf_url: str | None
    venue: str | None
    publisher: str | None
    issn: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    authors: str | None
    author_orcids: str | None
    corresponding_authors: str | None
    institutions: str | None
    countries: str | None
    abstract: str | None
    keywords: str | None
    topic: str | None
    subfield: str | None
    field: str | None
    domain: str | None
    is_open_access: bool | None
    open_access_status: str | None
    license: str | None
    funders: str | None
    grants: str | None
    mesh_terms: str | None
    references: str | None

@vectorize
def _flatten_work(authorships: list, primary_location: dict, best_oa_location: dict,
                  biblio: dict, primary_topic: dict, keywords: list, open_access: dict,
                  funders: list, awards: list, mesh: list, referenced_works: list,
                  abstract_inverted_index: str) -> Content:
    authorships = authorships or []
    loc = primary_location or {}
    source = loc.get("source") or {}
    oa = best_oa_location or {}
    first, last = _get(biblio, "first_page"), _get(biblio, "last_page")
    pages = f"{first}–{last}" if first and last and first != last else first
    insts = {}
    for a in authorships:
        for inst in a.get("institutions") or []:
            insts.setdefault(inst.get("display_name"), None)
    abstract = None
    if abstract_inverted_index:
        index = json.loads(abstract_inverted_index) or {}
        words = sorted((pos, w) for w, ps in index.items() for pos in ps)
        abstract = " ".join(w for _, w in words) or None
    return Content(
        url=loc.get("landing_page_url"),
        pdf_url=oa.get("pdf_url") or loc.get("pdf_url"),
        venue=source.get("display_name") or loc.get("raw_source_name"),
        publisher=source.get("host_organization_name"),
        issn=source.get("issn_l"),
        volume=_get(biblio, "volume"),
        issue=_get(biblio, "issue"),
        pages=pages,
        authors=_join(_get(a, "author", "display_name") or a.get("raw_author_name") for a in authorships),
        author_orcids=_join(_get(a, "author", "orcid") or "-" for a in authorships),
        corresponding_authors=_join(_get(a, "author", "display_name") for a in authorships if a.get("is_corresponding")),
        institutions=_join(insts),
        countries=_join(dict.fromkeys(c for a in authorships for c in a.get("countries") or [])),
        abstract=abstract,
        keywords=_join(k.get("display_name") for k in keywords or []),
        topic=_get(primary_topic, "display_name"),
        subfield=_get(primary_topic, "subfield", "display_name"),
        field=_get(primary_topic, "field", "display_name"),
        domain=_get(primary_topic, "domain", "display_name"),
        is_open_access=_get(open_access, "is_oa"),
        open_access_status=_get(open_access, "oa_status"),
        license=oa.get("license") or loc.get("license"),
        funders=_join(f.get("display_name") for f in funders or []),
        grants=_join(f"{a.get('funder_display_name')}: {a.get('funder_award_id')}" for a in awards or []
                     if a.get("funder_award_id")),
        mesh_terms=_join(dict.fromkeys(m.get("descriptor_name") for m in mesh or [])),
        references=_join(referenced_works),
    )

def content_table(raw):
    """Raw OpenAlex works -> one flat row per publication (no lists, no records)."""
    return (
        raw
        .mutate(content=_flatten_work(
            col.authorships, col.primary_location, col.best_oa_location, col.biblio,
            col.primary_topic, col.keywords, col.open_access, col.funders, col.awards,
            col.mesh, col.referenced_works, col.abstract_inverted_index))
        .select(col.id, col.doi, col.title, col.type, col.language, col.publication_date,
                col.publication_year, col.cited_by_count, col.referenced_works_count,
                col.is_retracted, col.updated_date, col.content)
        .unnest_wider(col.content)
        .select(col.id, col.doi, col.url, col.pdf_url,
                col.title, col.type, col.language, col.publication_date, col.publication_year,
                col.authors, col.author_orcids, col.corresponding_authors, col.institutions, col.countries,
                col.venue, col.publisher, col.issn, col.volume, col.issue, col.pages,
                col.abstract, col.keywords, col.topic, col.subfield, col.field, col.domain, col.mesh_terms,
                col.is_open_access, col.open_access_status, col.license,
                col.funders, col.grants,
                col.cited_by_count, col.referenced_works_count, col.references,
                col.is_retracted, col.updated_date)
        .rename(date=col.publication_date, year=col.publication_year,
                citations=col.cited_by_count, references_count=col.referenced_works_count)
    )

content = content_table(sample)
content
```

<iframe class="rat-output" src="../_assets/generated/9b973bd3ab1e.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 4. Find papers

Change the words and the year. `(?i)` ignores upper/lower case; `|` means OR.

```python
matches = (
    content
    .filter(col.field.str_detect(r"Agricultural and Biological Sciences"), col.year >= 2000)
    .arrange(desc(col.citations))
)
matches
```

<iframe class="rat-output" src="../_assets/generated/7a1bc4452c5c.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

## 5. Read one abstract

The most-cited paper in the sample that has an abstract.

```python
paper = papers.filter(~col.abstract.is_na()).slice_max(col.citations, 1)
print(paper.pull(col.title)[0])
print()
print(paper.pull(col.abstract)[0])
```

## 6. Authors

Each paper holds a list of authors. `unnest_longer` gives each author its own
row; `unnest_wider` turns their details into columns.

```python
authors = (
    sample
    .select(col.id, col.authorships)
    .unnest_longer(col.authorships)
    .unnest_wider(col.authorships)
    .unnest_wider(col.author, names_sep="_")
    .select(col.id, col.author_display_name, col.author_position)
)
authors.count(col.author_display_name).arrange(desc(col.n))
```

## 7. Keywords

The same idea. These keywords are added by OpenAlex, not chosen by the authors.

```python
keywords = (
    sample
    .select(col.id, col.keywords)
    .unnest_longer(col.keywords)
    .unnest_wider(col.keywords, names_sep="_")
)
keywords.count(col.keywords_display_name).arrange(desc(col.n))
```

## 8. Citations

One row per link: paper `id` cites paper `cited_id`. Some links are missing in
OpenAlex. A cited paper is often outside the sample, so its title shows as `null`.

```python
citations = (
    sample
    .select(col.id, col.referenced_works)
    .unnest_longer(col.referenced_works)
    .rename(cited_id=col.referenced_works)
)

# Which papers does the sample cite most?
(
    citations
    .count(col.cited_id)
    .left_join(papers.select(col.id, col.title).rename(cited_id=col.id), on=col.cited_id)
    .arrange(desc(col.n))
)
```

## 9. Save a reading list

Writes your search results from section 4 to a CSV file in this project.

```python
matches.select(col.id, col.title, col.year, col.citations).write("research/openalex/reading-list.csv")
```

## 10. Optional: explain papers with functai

A functai function works inside `mutate`, like `abstract_text` above. Each row is
one model call, so this sends **3 abstracts** to the model and may cost money.
It is off until you set `RUN_AI = True` and choose a model.

```python
import functai
from functai import ai

@ai
def explain_paper(title: str, abstract: str) -> str:
    """Explain this paper's question, approach, and findings in three short sentences.
    Use only the title and abstract. Do not invent missing details.
    """
    ...

RUN_AI = False
MODEL = None   # for example "claude-haiku-4-5"

if RUN_AI:
    with functai.configure(lm=MODEL):
        explained = (
            papers
            .filter(~col.abstract.is_na())
            .slice_head(3)
            .mutate(explanation=explain_paper(col.title, col.abstract))
            .select(col.title, col.explanation)
            .collect()
        )
    print(explained)
else:
    print("Skipped. Set RUN_AI = True and MODEL to try it.")
```
