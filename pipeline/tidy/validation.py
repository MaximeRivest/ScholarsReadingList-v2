"""Columnar integrity checks for in-memory partitions; no Python row materialization."""
import polars as pl
from .contracts import TABLES, RELATIONS, OWNER, REDUCED, ContractError

C = pl.col


def validate(tables):
    if set(tables) != set(TABLES):
        raise ContractError('Expected exactly the nine declared tables')
    ids = {}
    for name, spec in TABLES.items():
        df = tables[name]
        if dict(df.schema) != spec.columns:
            raise ContractError(f'{name}: unexpected schema')
        invalid = df.select(pl.any_horizontal([C(k).is_null() for k in spec.required]).any()).item()
        if invalid or df.filter(C(spec.key) == '').height:
            raise ContractError(f'{name}: missing required values')
        ids[name] = df.select(spec.key)
        if ids[name].n_unique() != df.height:
            raise ContractError(f'{name}: duplicate primary key')
        if name in RELATIONS:
            if df.filter(~C('relation').is_in(RELATIONS[name])).height:
                raise ContractError(f'{name}: unknown relation')
            if df.filter(C('score').is_null() != C('score_kind').is_null()).height:
                raise ContractError(f'{name}: score and score_kind must be supplied together')
            if df.filter(C('score').is_not_null() & ~C('score').is_finite()).height:
                raise ContractError(f'{name}: nonfinite score')
            if df.drop('link_id').n_unique() != df.height:
                raise ContractError(f'{name}: duplicate source assertion')
    fks = {
        'links_content_signatures': {'content_id':'content', 'signature_id':'signatures'},
        'links_signatures_entities': {'signature_id':'signatures', 'entity_id':'entities'},
        'links_content_classifications': {'content_id':'content','classification_id':'classifications'},
        'links_classifications_classifications': {'from_classification_id':'classifications','to_classification_id':'classifications'},
        'links_content_content': {'from_content_id':'content'},
        'links_classifications_entities': {'classification_id':'classifications','entity_id':'entities'},
        'links_content_entities': {'content_id':'content','entity_id':'entities'},
    }
    for name, refs in fks.items():
        for col, target in refs.items():
            absent = tables[name].select(col).unique().join(ids[target], left_on=col, right_on=TABLES[target].key, how='anti')
            if absent.height:
                raise ContractError(f'{name}.{col}: unknown endpoint')
    content = tables['content']
    source_keys = content.select(C('content_id').alias('source_record'), 'source_revision')
    for name in ['entities', 'classifications', *RELATIONS]:
        df = tables[name]
        if df.select('source_record','source_revision').unique().join(source_keys,on=['source_record','source_revision'],how='anti').height:
            raise ContractError(f'{name}: profile/link evidence must identify a current source revision')
        owner = OWNER.get(name)
        if owner and name not in REDUCED and df.filter(C(owner) != C('source_record')).height:
            raise ContractError(f'{name}: work-owned link evidence comes from another content item')
    if content.filter((C('citations') < 0) | (C('references_count') < 0)).height:
        raise ContractError('Citation/reference counts cannot be negative')
    ownership = tables['links_content_signatures'].select('signature_id','content_id')
    if ownership.height != tables['signatures'].height or ownership['signature_id'].n_unique() != ownership.height:
        raise ContractError('Each signature must have exactly one content owner')
    sigs = tables['signatures'].join(ownership,on='signature_id')
    if sigs.filter(C('source_record') != C('content_id')).height:
        raise ContractError('Signature owner differs from source context')
    if sigs.select('source_record','source_revision').unique().join(source_keys,on=['source_record','source_revision'],how='anti').height:
        raise ContractError('Signature must use the current content revision')
    if sigs.select('content_id','author_order','affiliation_order').n_unique() != sigs.height:
        raise ContractError('Duplicate author/affiliation occurrence')
    if sigs.filter((C('author_order') < 1) | (C('affiliation_order') < 1)).height:
        raise ContractError('Source occurrence orders are 1-based')
    if sigs.filter(C('author_position').is_not_null() & ~C('author_position').is_in(['first','middle','last'])).height:
        raise ContractError('Unknown byline position')
    if sigs.filter(C('affiliation_order').is_null() & C('affiliation_text').is_not_null()).height:
        raise ContractError('An affiliation text requires its source order')
    if tables['entities'].filter(~C('entity_type').is_in(['person','organization'])).height:
        raise ContractError('Unknown entity type')
    matches = (tables['links_signatures_entities']
               .join(tables['entities'].select('entity_id','entity_type'), on='entity_id')
               .join(tables['signatures'].select('signature_id',C('source_record').alias('_signature_source')),on='signature_id'))
    if matches.filter(C('source_record') != C('_signature_source')).height:
        raise ContractError('Signature match evidence comes from another content item')
    if matches.filter(((C('relation')=='identifies_person') & (C('entity_type')!='person')) |
                      ((C('relation')=='identifies_institution') & (C('entity_type')!='organization'))).height:
        raise ContractError('Signature match relation has the wrong entity type')
    for name in ('links_classifications_entities', 'links_content_entities'):
        typed = tables[name].join(tables['entities'].select('entity_id', 'entity_type'), on='entity_id')
        if typed.filter(C('entity_type') != 'organization').height:
            raise ContractError('Funding/host links must point to organizations')
    hierarchy = tables['links_classifications_classifications']
    if hierarchy.filter(C('from_classification_id')==C('to_classification_id')).height:
        raise ContractError('A classification cannot be broader than itself')
    refs=tables['links_content_content']
    if refs.select('from_content_id','to_content_id').n_unique()!=refs.height:
        raise ContractError('Duplicate citation pair in this source assignment')
    external=refs.select('to_content_id').unique().join(ids['content'],left_on='to_content_id',right_on='content_id',how='anti')
    return {'external_reference_targets':external.height}
