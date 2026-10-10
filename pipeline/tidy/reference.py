"""Straightforward, in-memory reference extraction for row tests, not production.

The native partition transforms must produce exactly these logical rows. This
module favors readable correctness over speed and is used on small inputs only.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import blake2b
import json
import math
import re

from .contracts import ContractError, TABLES, REDUCED, frame, validate
from .keys import link_key, signature_key
from .reduction import python_rank, PROPS
from .source_roles import location_relation, KNOWN_VERSIONS

METHOD = 'openalex-import-v3'
_SDG = re.compile(r'https?://metadata\.un\.org/sdg/([0-9]+)')
_MESH_UI = re.compile(r'[A-Z][0-9]+')
_MESH_Q = re.compile(r'Q[0-9]+')
MESH_KINDS = {'D': 'mesh_descriptor', 'C': 'mesh_supplementary_concept'}


def _json(value):
    # Internal oracle fingerprint only, not exported JSON. Parquet may contain
    # nonfinite numbers in unused fields; contracts reject nonfinite outputs.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=True,
                      separators=(',', ':'), default=lambda x: x.isoformat())


def _digest(value):
    return blake2b(_json(value).encode(), digest_size=16).hexdigest()


def oa_id(value):
    """Canonical source-qualified ID, preserving upstream identity rather than name."""
    if not isinstance(value, str) or not value:
        raise ContractError(f'Missing/invalid OpenAlex ID: {value!r}')
    for prefix in ('https://openalex.org/', 'http://openalex.org/', 'openalex:'):
        if value.startswith(prefix) and len(value) > len(prefix):
            return 'openalex:' + value[len(prefix):]
    raise ContractError(f'Unrecognized OpenAlex identifier: {value!r}')


def sdg_id(value):
    m = _SDG.fullmatch(value or '')
    if not m:
        raise ContractError(f'Unrecognized SDG identifier: {value!r}')
    return 'unsdg:' + m.group(1), m.group(1)


def _date(value):
    if value is None or type(value) is date:
        return value
    return date.fromisoformat(value)


def _time(value):
    if value is None:
        return None
    dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    # Old source timestamps may be naive; OpenAlex source times are treated as UTC.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _kleene_any(values):
    if any(v is True for v in values):
        return True
    return None if any(v is None for v in values) else False


def _ids(raw):
    """OpenAlex `ids`: Parquet list of {key, value} or API object. First value wins."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    out = {}
    for item in raw:
        if item and item.get('key') is not None and item['key'] not in out:
            out[item['key']] = item.get('value')
    return out


def abstract_text(raw):
    """Preserve repeated JSON keys and escaped words; fail clearly on malformed data."""
    if raw is None or raw == '':
        return None
    class Pairs(list):
        pass
    pairs = json.loads(raw, object_pairs_hook=Pairs)
    if pairs is None:
        return None
    if not isinstance(pairs, Pairs):
        raise ContractError('Abstract inverted index must be a JSON object or null')
    words = []
    for word, positions in pairs:
        if not isinstance(positions, list) or isinstance(positions, Pairs):
            raise ContractError('Abstract positions must be an array of integers')
        for pos in positions:
            if type(pos) is not int or not 0 <= pos < 2**63:
                raise ContractError('Abstract positions must be nonnegative Int64 integers')
            words.append((pos, word))
    # Stable source occurrence order breaks position ties; do not silently discard words.
    return ' '.join(word for _, word in sorted(words, key=lambda pair: pair[0])) or None


@dataclass
class Result:
    tables: dict
    notes: list[dict]


class _Extract:
    def __init__(self):
        self.rows = {name: [] for name in TABLES if name not in REDUCED}
        self.reduced = {name: {} for name in REDUCED}
        self.variants = {name: {} for name in REDUCED}
        self.notes = {}
        self.link_ids = set()

    def note(self, code, path, detail):
        n = self.notes.setdefault(code, dict(code=code, count=0, detail=detail, examples=[]))
        n['count'] += 1
        if len(n['examples']) < 5:
            n['examples'].append(dict(content_id=self.cid, source_field=path))

    def oa(self, value, path):
        """Canonical OpenAlex ID, or None (noted) when the value is not one."""
        try:
            return oa_id(value)
        except ContractError:
            self.note('invalid_identifier', path, 'Unrecognized identifier dropped')
            return None

    def origin(self, path):
        return dict(source_record=self.cid, source_revision=self.revision, source_field=path)

    def _keep(self, table, key, row):
        self.variants[table].setdefault(key, set()).add(tuple(row[p] for p in PROPS[table]))
        old = self.reduced[table].get(key)
        if old is None or python_rank(table, row) > python_rank(table, old):
            self.reduced[table][key] = row

    def link(self, table, relation, path, **fields):
        from .keys import ENDPOINTS
        if any(fields.get(c) is None for c in ENDPOINTS[table]):
            self.note('invalid_identifier', path, f'{table}: link with an unrecognized identifier dropped')
            return
        row = dict(relation=relation, method=METHOD, **self.origin(path), score=None, score_kind=None)
        row.update(fields)
        row['link_id'] = link_key(table, row)
        if table in REDUCED:
            row['observed_at'] = self.when
            self._keep(table, row['link_id'], row)
        elif row['link_id'] not in self.link_ids:
            self.rows[table].append(row)
            self.link_ids.add(row['link_id'])

    def catalog(self, table, source_id, identifier, path, **fields):
        if identifier is None:
            return None
        row = {TABLES[table].key: identifier, 'source_id': source_id, **fields,
               **self.origin(path), 'observed_at': self.when}
        self._keep(table, identifier, row)
        return identifier

    def entity(self, raw, kind, path):
        return self.catalog('entities', raw['id'], self.oa(raw['id'], path), path, entity_type=kind,
                            name=raw.get('display_name'), orcid=raw.get('orcid'), ror=raw.get('ror'),
                            country_code=raw.get('country_code'), organization_type=raw.get('type'))

    def classification(self, source_id, identifier, kind, scheme, name, path, notation=None):
        return self.catalog('classifications', source_id, identifier, path, classification_type=kind,
                            scheme=scheme, name=name, notation=notation)

    def authors(self):
        for ai, author in enumerate(self.work.get('authorships') or []):
            ap = f'authorships[{ai}]'
            resolved = author.get('author') or {}
            person = None
            if resolved.get('id'):
                person = self.entity({**resolved, 'country_code': None, 'type': None}, 'person', ap + '.author')
            insts = {i['id']: i for i in author.get('institutions') or [] if i.get('id')}
            for ii, institution in enumerate(author.get('institutions') or []):
                if institution.get('id'):
                    self.entity({**institution, 'orcid': None}, 'organization', f'{ap}.institutions[{ii}]')
            pairs = []
            if author.get('affiliations'):
                for fi, aff in enumerate(author['affiliations']):
                    pairs.append((aff.get('raw_affiliation_string'), aff.get('institution_ids') or [],
                                  f'{ap}.affiliations[{fi}].raw_affiliation_string',
                                  f'{ap}.affiliations[{fi}].institution_ids'))
            elif author.get('raw_affiliation_strings'):
                for fi, raw in enumerate(author['raw_affiliation_strings']):
                    pairs.append((raw, [], f'{ap}.raw_affiliation_strings[{fi}]', None))
                    self.note('raw_affiliation_fallback', ap, 'Raw strings retained; no guessed institution matches')
            if not pairs:
                pairs = [(None, [], None, None)]
            for fi, (text, institution_ids, text_path, ids_path) in enumerate(pairs):
                aff_order = fi + 1 if text_path is not None else None
                sid = signature_key(self.cid, self.revision, ap, text_path)
                self.rows['signatures'].append(dict(
                    signature_id=sid, author_text=author.get('raw_author_name'), affiliation_text=text,
                    author_order=ai + 1, affiliation_order=aff_order,
                    author_position=author.get('author_position'), is_corresponding=author.get('is_corresponding'),
                    raw_orcid=author.get('raw_orcid'), source_record=self.cid, source_revision=self.revision,
                    author_source_field=ap + '.raw_author_name', affiliation_source_field=text_path))
                self.link('links_content_signatures', 'has_signature', text_path or ap,
                          content_id=self.cid, signature_id=sid)
                if person:
                    self.link('links_signatures_entities', 'identifies_person', ap + '.author.id',
                              signature_id=sid, entity_id=person)
                for iid in dict.fromkeys(institution_ids):
                    profile = insts.get(iid, {'id': iid})
                    organization = self.entity({**profile, 'orcid': None}, 'organization', ids_path)
                    self.link('links_signatures_entities', 'identifies_institution', ids_path,
                              signature_id=sid, entity_id=organization)
            if insts and not any(p[1] for p in pairs):
                self.note('aggregate_institutions_only', ap + '.institutions',
                          'Aggregate organizations retained, no unsupported signature matches')

    def subjects(self):
        topics = [(t, f'topics[{i}]') for i, t in enumerate(self.work.get('topics') or [])]
        primary = self.work.get('primary_topic') or {}
        if primary.get('id') and not any(t.get('id') == primary['id'] for t, _ in topics):
            topics.append((primary, 'primary_topic'))
        seen = set()
        for topic, path in topics:
            if not topic.get('id') or topic['id'] in seen:
                continue
            seen.add(topic['id'])
            cid = self.classification(topic['id'], self.oa(topic['id'], path), 'topic', 'openalex.topics',
                                      topic.get('display_name'), path)
            score = topic.get('score')
            self.link('links_content_classifications', 'about', path, content_id=self.cid,
                      classification_id=cid, score=score,
                      score_kind='openalex_topic_score' if score is not None else None,
                      is_primary=(topic['id'] == primary['id']) if primary.get('id') else None)
            child = cid
            for level in ('subfield', 'field', 'domain'):
                parent = topic.get(level) or {}
                if not parent.get('id'):
                    child = None  # never invent a direct edge across a missing level
                    continue
                pid = self.classification(parent['id'], self.oa(parent['id'], path + '.' + level), level, 'openalex.topics',
                                          parent.get('display_name'), path + '.' + level)
                if child is not None:
                    self.link('links_classifications_classifications', 'broader', path + '.' + level,
                              from_classification_id=child, to_classification_id=pid)
                child = pid
        for ki, keyword in enumerate(self.work.get('keywords') or []):
            path = f'keywords[{ki}]'
            if not keyword.get('id'):
                self.note('keyword_without_id', path, 'No name-derived ID invented')
                continue
            cid = self.classification(keyword['id'], self.oa(keyword['id'], path), 'keyword', 'openalex.keywords',
                                      keyword.get('display_name'), path)
            score = keyword.get('score')
            self.link('links_content_classifications', 'about', path, content_id=self.cid,
                      classification_id=cid, score=score,
                      score_kind='openalex_keyword_score' if score is not None else None)
        self.mesh()
        seen = set()
        for si, goal in enumerate(self.work.get('sustainable_development_goals') or []):
            path = f'sustainable_development_goals[{si}]'
            if not goal.get('id'):
                self.note('sdg_without_id', path, 'No name-derived ID invented')
                continue
            if goal['id'] in seen:
                continue
            seen.add(goal['id'])
            try:
                identifier, number = sdg_id(goal['id'])
            except ContractError:
                self.note('invalid_identifier', path, 'Unrecognized SDG identifier dropped')
                continue
            cid = self.classification(goal['id'], identifier, 'sdg', 'un.sdg', goal.get('display_name'), path, number)
            score = goal.get('score')
            self.link('links_content_classifications', 'about', path, content_id=self.cid,
                      classification_id=cid, score=score,
                      score_kind='openalex_sdg_score' if score is not None else None)
        if self.work.get('concepts'):
            self.note('deferred_scheme', 'concepts', 'Deprecated OpenAlex concepts not extracted; retained in raw input')

    def mesh(self):
        descriptors, headings, seen_entries = {}, {}, set()
        for mi, m in enumerate(self.work.get('mesh') or []):
            d, q = m.get('descriptor_ui'), m.get('qualifier_ui')
            if not d:
                self.note('mesh_without_descriptor', f'mesh[{mi}]', 'Entry without descriptor ignored')
                continue
            if not _MESH_UI.fullmatch(d) or (q is not None and not _MESH_Q.fullmatch(q)):
                self.note('mesh_malformed_identifier', f'mesh[{mi}]', 'Entry with a malformed MeSH identifier ignored')
                continue
            if d[0] not in MESH_KINDS:
                self.note('mesh_unexpected_record_type', f'mesh[{mi}]', 'Kept as mesh_record; not a descriptor or supplementary concept')
            entry = _json(m)
            if entry in seen_entries:
                self.note('mesh_duplicate_entry', f'mesh[{mi}]', 'Exact duplicate MeSH entry collapsed')
            seen_entries.add(entry)
            descriptors.setdefault(d, dict(index=mi, name=m.get('descriptor_name'), majors=[]))['majors'].append(m.get('is_major_topic'))
            if q is not None:
                headings.setdefault((d, q), dict(index=mi, dname=m.get('descriptor_name'),
                                                 qname=m.get('qualifier_name'), majors=[]))['majors'].append(m.get('is_major_topic'))
        for d, info in descriptors.items():
            path = f"mesh[{info['index']}].descriptor_ui"
            cid = self.classification(d, 'mesh:' + d, MESH_KINDS.get(d[0], 'mesh_record'), 'nlm.mesh', info['name'], path, d)
            self.link('links_content_classifications', 'about', path, content_id=self.cid,
                      classification_id=cid, is_primary=_kleene_any(info['majors']))
        for (d, q), info in headings.items():
            path = f"mesh[{info['index']}]"
            name = f"{info['dname']}/{info['qname']}" if info['dname'] is not None and info['qname'] is not None else None
            cid = self.classification(f'{d}/{q}', f'mesh:{d}/{q}', 'mesh_heading', 'nlm.mesh', name, path, f'{d}/{q}')
            self.link('links_content_classifications', 'about', path, content_id=self.cid,
                      classification_id=cid, is_primary=_kleene_any(info['majors']))
            self.link('links_classifications_classifications', 'broader', path,
                      from_classification_id=cid, to_classification_id='mesh:' + d)

    def funding(self):
        seen = set()
        for fi, funder in enumerate(self.work.get('funders') or []):
            path = f'funders[{fi}]'
            if not funder.get('id'):
                self.note('funder_without_id', path, 'No name-derived ID invented')
                continue
            if funder['id'] in seen:
                continue
            seen.add(funder['id'])
            eid = self.entity({**funder, 'orcid': None, 'country_code': None, 'type': None}, 'organization', path)
            self.link('links_content_entities', 'funded_by', path, content_id=self.cid, entity_id=eid)
        seen = set()
        for ai, award in enumerate(self.work.get('awards') or []):
            path = f'awards[{ai}]'
            if not award.get('id'):
                self.note('award_without_id', path, 'Award without identifier not extracted')
                continue
            if award['id'] in seen:
                continue
            seen.add(award['id'])
            cid = self.classification(award['id'], self.oa(award['id'], path), 'award', 'openalex.awards',
                                      award.get('display_name'), path, award.get('funder_award_id'))
            self.link('links_content_classifications', 'funded_by', path, content_id=self.cid, classification_id=cid)
            if award.get('funder_id'):
                fpath = path + '.funder_id'
                eid = self.entity(dict(id=award['funder_id'], display_name=award.get('funder_display_name')),
                                  'organization', fpath)
                self.link('links_classifications_entities', 'funded_by', fpath, classification_id=cid, entity_id=eid)

    def locations(self):
        primary = self.work.get('primary_location') or {}
        locations = [(x, f'locations[{i}]') for i, x in enumerate(self.work.get('locations') or [])]
        def key(loc):
            return ('id', loc['id']) if loc.get('id') else ('fields', _json(loc))
        if primary and not any(key(loc) == key(primary) for loc, _ in locations):
            locations.append((primary, 'primary_location'))
        for loc, path in locations:
            source = loc.get('source') or {}
            if not source.get('id'):
                self.note('location_without_source_id', path, 'No venue classification invented')
                continue
            kind = source.get('type') or 'source'
            cid = self.classification(source['id'], self.oa(source['id'], path + '.source'), kind, 'openalex.sources',
                                      source.get('display_name'), path + '.source', source.get('issn_l'))
            flag, version = loc.get('is_published'), loc.get('version')
            if flag is not None and version in KNOWN_VERSIONS and flag != (version == 'publishedVersion'):
                self.note('location_version_conflict', path, 'Conflicting publication flags; no publication claim inferred')
            self.link('links_content_classifications', location_relation(kind, loc),
                      path, content_id=self.cid, classification_id=cid,
                      is_primary=(key(loc) == key(primary)) if primary else None,
                      source_location_id=loc.get('id'), raw_source_name=loc.get('raw_source_name'),
                      landing_page_url=loc.get('landing_page_url'), pdf_url=loc.get('pdf_url'),
                      version=loc.get('version'), license=loc.get('license'), is_oa=loc.get('is_oa'))
            if source.get('host_organization'):
                hpath = path + '.source.host_organization'
                eid = self.entity(dict(id=source['host_organization'], display_name=source.get('host_organization_name')),
                                  'organization', hpath)
                self.link('links_classifications_entities', 'host_organization', hpath,
                          classification_id=cid, entity_id=eid)

    def work_rows(self, work, revision=None):
        self.work = work
        self.cid = oa_id(work['id'])
        self.revision = revision or 'record:' + _digest(work)
        self.when = _time(work.get('updated_date'))
        refs = work.get('referenced_works') or []
        try:
            abstract = abstract_text(work.get('abstract_inverted_index'))
        except ValueError:  # ContractError and JSON errors
            self.note('invalid_abstract', 'abstract_inverted_index', 'Malformed inverted index: abstract stored as unknown')
            abstract = None
        biblio, oa, ids = work.get('biblio') or {}, work.get('open_access') or {}, _ids(work.get('ids'))
        fwci = work.get('fwci')
        if isinstance(fwci, float) and not math.isfinite(fwci):
            self.note('nonfinite_fwci', 'fwci', 'Nonfinite FWCI stored as unknown')
            fwci = None
        self.rows['content'].append(dict(
            content_id=self.cid, source_id=work['id'], source_revision=self.revision,
            title=work.get('title'), doi=work.get('doi'), type=work.get('type'), language=work.get('language'),
            date=_date(work.get('publication_date')), year=work.get('publication_year'),
            abstract=abstract, citations=work.get('cited_by_count'),
            references_count=work.get('referenced_works_count'),
            is_complete=work.get('language') == 'en' and abstract is not None and bool(refs),
            is_retracted=work.get('is_retracted'), updated_at=self.when,
            volume=biblio.get('volume'), issue=biblio.get('issue'),
            first_page=biblio.get('first_page'), last_page=biblio.get('last_page'),
            is_oa=oa.get('is_oa'), oa_status=oa.get('oa_status'), oa_url=oa.get('oa_url'),
            fwci=fwci, is_paratext=work.get('is_paratext'),
            pmid=ids.get('pmid'), pmcid=ids.get('pmcid'), mag=ids.get('mag')))
        self.authors()
        self.subjects()
        self.funding()
        self.locations()
        targets = set()
        for ri, raw_id in enumerate(refs):
            target = self.oa(raw_id, f'referenced_works[{ri}]')
            if target is not None and target not in targets:
                self.link('links_content_content', 'cites', f'referenced_works[{ri}]',
                          from_content_id=self.cid, to_content_id=target)
                targets.add(target)
        if work.get('referenced_works_count') is not None and work['referenced_works_count'] != len(targets):
            self.note('reference_count_differs', 'referenced_works',
                      'Reported reference count differs from distinct extracted targets')


def extract(works, source_revisions=None):
    """Extract small already-version-selected input; reject competing revisions.

    An optional content-ID -> source-revision mapping lets parity tests supply
    native source fingerprints without sharing any transformation logic.
    """
    worker = _Extract()
    seen = {}
    for work in works:
        cid, fingerprint = oa_id(work['id']), _digest(work)
        if cid in seen:
            if seen[cid] != fingerprint:
                raise ContractError(f'{cid}: select one source revision before row extraction')
            continue
        seen[cid] = fingerprint
        worker.work_rows(work, source_revisions[cid] if source_revisions is not None else None)
    for table, rows in worker.reduced.items():
        worker.rows[table] = list(rows.values())
        varied = sum(len(v) > 1 for v in worker.variants[table].values())
        if varied and PROPS[table]:
            worker.notes['catalog_variation:' + table] = dict(
                code='catalog_variation', table=table, count=varied,
                detail='Several observed profiles; one chosen by the documented rank')
    tables = {name: frame(name, worker.rows[name]).sort(TABLES[name].key) for name in TABLES}
    validate(tables)
    return Result(tables, sorted(worker.notes.values(), key=lambda n: (n['code'], n.get('table', ''))))
