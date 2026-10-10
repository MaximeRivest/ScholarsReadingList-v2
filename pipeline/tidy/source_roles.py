"""Explicit source-role policy, independent of storage or execution engine."""
from .contracts import ContractError

PUBLICATION_VENUES = frozenset({'journal', 'conference', 'book series'})
KNOWN_VERSIONS = frozenset({'publishedVersion', 'acceptedVersion', 'submittedVersion'})


def location_relation(kind, location):
    flag, version = location.get('is_published'), location.get('version')
    if flag is not None and type(flag) is not bool:
        raise ContractError('Location is_published must be boolean or null')
    confirmed = ((flag is True and version in (None, 'publishedVersion'))
                 or (flag is None and version == 'publishedVersion'))
    return 'published_in' if kind in PUBLICATION_VENUES and confirmed else 'hosted_by'
