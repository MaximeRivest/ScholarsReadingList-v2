"""Name and title normalisation shared by paper versions (build_versions.py) and author identity (authors.py).

Python versions for tests and small jobs; build scripts use the DuckDB SQL twins below (same rules), so a rule
changed here must change in both: test_identity.py checks they agree on a list of cases.
"""
from __future__ import annotations

import re
import unicodedata

# Particles that start a surname ("van der Berg", "de la Cruz", "Al Saud") and are not given names.
PARTICLES = frozenset("van von der den de del della di da dos das du la le al el bin ibn ter ten zu".split())
# Placeholder or corporate words that appear where an author's name should be; never evidence of a shared person.
PLACEHOLDER_TOKENS = frozenset("none available unknown anonymous anon redaktion redaction redacao die the editor editors "
                               "editorial staff board team group collective committee consortium society association "
                               "et al author authors unav na nn various".split())


CYR = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ"
LAT = "abvgdeezziiklmnoprstufhccss_y_euaiieg"
_CYR = str.maketrans(CYR + CYR.upper(), LAT + LAT.upper())
TITLES = re.compile(r"^\s*(prof|dr|mr|mrs|ms|sir)\.?\s+", re.I)


def name_fold(s: str) -> str:
    """For comparing names: titles dropped, Cyrillic written in Latin letters (one letter each, as the SQL twin
    must: 'Линников' -> 'linnikov', 'Кулініч' -> 'kulinic'), then ascii_fold."""
    return ascii_fold(TITLES.sub("", s or "").translate(_CYR)).replace("_", "")


def ascii_fold(s: str) -> str:
    """'Gołka' -> 'golka', 'Østby' -> 'ostby', 'Smaı̈li' -> 'smaili'."""
    s = s.replace("ł", "l").replace("Ł", "L").replace("ø", "o").replace("Ø", "O").replace("ß", "ss") \
         .replace("đ", "d").replace("Đ", "D").replace("ı", "i").replace("æ", "ae").replace("œ", "oe")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def title_key(title: str | None) -> str | None:
    """Letters and digits of the title, folded; None when too short or too few words to identify a paper."""
    if not title:
        return None
    k = re.sub(r"[^a-z0-9]", "", ascii_fold(title))
    if len(k) < 25 or len(title.split()) < 4:
        return None
    return k


ORG_WORDS = ("department|departamento|office|universit\\w*|commission|services|service|library|ministry|council|agency|"
             "institut\\w*|college|school|center|centre|hospital|government|national|division|bureau|society|association|"
             "foundation|company|inc|ltd|gmbh|corporation|program|programme|project|committee|board|laborator\\w*|museum|"
             "archives?|team|group|consortium|network|organi[sz]ation|collaboration|editors|redaktion|anonymous|unknown")


def is_person_name(raw: str) -> bool:
    """False for organisations, placeholders and lists of names printed as one author ('South Carolina Office of
    Resilience', 'Dr A Khan, Dr S Shakir, Dr M Bhutta', 'None Available'). Library-catalogue names stay people
    ('Lubac, Henri de, 1896-1991', 'Gerrard, Martha, editor.'): digits only with a year (a catalogue date)."""
    raw = re.sub(r"https?://\S+|(https?://)?orcid\.org/\S+|\bid_orcid\s*\S+|(?<=[A-Za-z])\d{1,2}\b|\(\s*\d+\s*\)|\b\d{6,}\b|\bNIDN\.?",
                 "", (raw or "")).strip()
    if re.fullmatch(r"[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+", raw):      # 'Melvin S. Center': a person, whatever the word
        return True
    year = re.search(r"\b1[0-9]{3}\b|\b20[0-9]{2}\b", raw)       # a catalogue date: its day numbers are fine too
    if not raw or (re.search(r"[0-9]", raw) and not year) or len(raw.split()) > 12:
        return False
    parts = [p for p in raw.split(",") if p.strip()]
    if len(parts) >= 3 and all(len(p.split()) >= 2 for p in parts):
        return False
    toks = set(re.findall(r"[a-z]+", ascii_fold(raw)))
    if toks and toks <= PLACEHOLDER_TOKENS:         # an empty set (a name in another script) is not a placeholder
        return False
    return not re.search(rf"\b({ORG_WORDS})\b", ascii_fold(raw))


def split_name(raw: str) -> tuple[str, str]:
    """'Surname, Given' or 'Given Surname' -> (given, surname), both folded, particles kept with the surname."""
    raw = TITLES.sub("", re.sub(r"\s+", " ", raw or "").strip())
    if "," in raw:
        sur, given = raw.split(",", 1)
    else:
        parts = raw.split(" ")
        i = len(parts) - 1
        while i > 0 and ascii_fold(parts[i - 1]).strip(".") in PARTICLES:
            i -= 1
        given, sur = " ".join(parts[:i]), " ".join(parts[i:])
    f = lambda x: re.sub(r"[^a-z ]", " ", name_fold(x)).split()
    return " ".join(f(given)), "".join(f(sur))


def surname_tokens(raw: str) -> set[str]:
    """The tokens that could be this person's surname: the surname by the rule above, and, because East-Asian
    names arrive in either order ('Wei Lei' / 'Lei Wei'), the first token too. Initials (one letter) never."""
    given, sur = split_name(raw)
    out = {sur} if len(sur) >= 2 else set()
    g = given.split()
    if g and len(g[0]) >= 2:
        out.add(g[0])
    return out - PLACEHOLDER_TOKENS


def name_key(raw: str) -> str:
    """Block key for author matching: surname + first initial ('Hidekatsu Yanagi', 'Yanagi, H.' -> 'yanagi h')."""
    given, sur = split_name(raw)
    return f"{sur} {given[:1]}".strip()


def given_compatible(a: str, b: str) -> bool:
    """Given names agree as far as both go: 'J. P.' ~ 'John Paul' ~ 'John'; 'John' !~ 'James'; '' ~ anything."""
    ta, tb = a.split(), b.split()
    for x, y in zip(ta, tb):
        if len(x) == 1 or len(y) == 1:
            if x[0] != y[0]:
                return False
        elif x != y:
            return False
    return True


def same_orcid_compatible(a: str, b: str) -> bool:
    """Python twin of the SQL macro ncompat_orcid (difflib's ratio stands in for Jaro-Winkler, in tests only)."""
    import difflib
    words = lambda x: {t for t in re.findall(r"[a-z]+", name_fold(x)) if len(t) >= 3}  # noqa: E731
    (ga, sa), (gb, sb) = split_name(a), split_name(b)
    shared = words(a) & words(b)
    if same_person_name(a, b) or len(shared) >= 2:
        return True
    if not (ga and gb and ga[0] == gb[0]):
        return False
    fa, fb = ga.split()[0], gb.split()[0]
    if not (len(fa) == 1 or len(fb) == 1 or fa.startswith(fb) or fb.startswith(fa)
            or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.8):
        return False
    return bool(shared) or bool(sa and sb and difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.8)


def same_person_name(a: str, b: str) -> bool:
    """Two printed names can be one person: same surname, compatible given names (either order of the two)."""
    ga, sa = split_name(a)
    gb, sb = split_name(b)
    if sa and sa == sb and given_compatible(ga, gb):
        return True
    # East-Asian order swapped: 'Wei Lei' vs 'Lei Wei'
    fa, fb = ga.split(), gb.split()
    return bool(fa and fb and fa[0] == sb and fb[0] == sa and len(fa) == 1 and len(fb) == 1)


# --- DuckDB twins -------------------------------------------------------------------------------------------
# fold(x): DuckDB's strip_accents does not touch ł/ø/ß/ı; the replacements match ascii_fold.
SQL_FOLD = ("lower(strip_accents(replace(replace(replace(replace(replace(replace(replace({x},'ł','l'),'Ł','L'),"
            "'ø','o'),'Ø','O'),'ß','ss'),'ı','i'),'æ','ae')))")


def sql_fold(x: str) -> str:
    return SQL_FOLD.format(x=x)


def sql_title_key(x: str) -> str:
    """NULL when too short; same rule as title_key."""
    k = f"regexp_replace({sql_fold(x)}, '[^a-z0-9]', '', 'g')"
    return (f"CASE WHEN length({k}) >= 25 AND len(string_split(trim({x}), ' ')) >= 4 THEN {k} END")


def sql_is_person(x: str) -> str:
    """SQL twin of is_person_name."""
    x = (f"trim(regexp_replace(regexp_replace(coalesce({x}, ''), 'https?://\\S+|(https?://)?orcid\\.org/\\S+|\\bid_orcid\\s*\\S+|\\(\\s*\\d+\\s*\\)|\\b\\d{{6,}}\\b|\\bNIDN\\.?', '', 'g'), "
         f"'([A-Za-z])\\d{{1,2}}\\b', '\\1', 'g'))")
    f = sql_fold(x)
    parts = f"list_filter(string_split({x}, ','), p -> trim(p) <> '')"
    return (f"(regexp_full_match({x}, '[A-Z][a-z]+ [A-Z]\\. [A-Z][a-z]+') OR (coalesce(trim({x}), '') <> '' "
            f"AND (NOT regexp_matches({x}, '[0-9]') OR regexp_matches({x}, '\\b1[0-9]{{3}}\\b|\\b20[0-9]{{2}}\\b')) "
            f"AND len(string_split(trim({x}), ' ')) <= 12 "
            f"AND NOT (len({parts}) >= 3 AND list_bool_and(list_transform({parts}, p -> len(string_split(trim(p), ' ')) >= 2))) "
            f"AND NOT regexp_matches({f}, '\\b({ORG_WORDS})\\b') "
            f"AND NOT (len(regexp_extract_all({f}, '[a-z]+')) > 0 "
            f"AND list_has_all({list(sorted(PLACEHOLDER_TOKENS))!r}, regexp_extract_all({f}, '[a-z]+')))))")


def sql_name_macros(con) -> None:
    """DuckDB macros: nsur(x) surname, nfirst(x) first given token, ncompat(a, b) = same_person_name (approximately:
    surname + first given name or initial, or the two-token East-Asian swap)."""
    f = sql_fold("x")
    # names: titles dropped, Cyrillic to Latin one letter each (name_fold), then folded
    con.execute(f"""CREATE OR REPLACE MACRO nnorm(x) AS replace(translate(regexp_replace(coalesce(x, ''), '^\\s*(prof|dr|mr|mrs|ms|sir)\\.?\\s+', '', 'i'),
                    '{CYR + CYR.upper()}', '{LAT + LAT.upper()}'), '_', '')""")
    f = lambda e: sql_fold(e)  # noqa: E731
    con.execute(f"""CREATE OR REPLACE MACRO nsur(x) AS regexp_replace(CASE WHEN contains(nnorm(x), ',') THEN {f("split_part(nnorm(x), ',', 1)")}
                    ELSE {f("split_part(trim(nnorm(x)), ' ', -1)")} END, '[^a-z]', '', 'g')""")
    con.execute(f"""CREATE OR REPLACE MACRO nfirst(x) AS split_part(trim(regexp_replace(CASE WHEN contains(nnorm(x), ',')
                    THEN {f("split_part(nnorm(x), ',', 2)")} ELSE CASE WHEN len(string_split(trim(nnorm(x)), ' ')) > 1
                    THEN {f("nnorm(x)")}[1:length({f("nnorm(x)")}) - length(split_part(trim({f("nnorm(x)")}), ' ', -1))] ELSE '' END END,
                    '[^a-z]+', ' ', 'g')), ' ', 1)""")
    con.execute("""CREATE OR REPLACE MACRO ncompat(a, b) AS
        (nsur(a) <> '' AND nsur(a) = nsur(b) AND (nfirst(a) = '' OR nfirst(b) = '' OR
            (left(nfirst(a), 1) = left(nfirst(b), 1) AND (length(nfirst(a)) = 1 OR length(nfirst(b)) = 1 OR nfirst(a) = nfirst(b)))))
        OR (nfirst(a) <> '' AND nfirst(a) = nsur(b) AND nfirst(b) = nsur(a))""")
    # Two names under ONE ORCID (the ORCID already says one person): only an ORCID on the wrong person must be caught.
    # Compatible: ncompat; or two shared name words of 3+ letters ('Harel Noam Y.' / 'Noam Y. Harel'); or the same first
    # initial and either a shared name word ('María Morales Suárez-Varela' / 'María Morales-Suárez-Varela') or similar
    # surnames (Jaro-Winkler >= 0.85: 'Yarmolovich' / 'Yarmolovych'), with given names that agree ('Laurie L.' /
    # 'Leah Ruby Williams' fails). Cyrillic is compared in Latin letters; another
    # script cannot vouch for anything ('沈志强' / 'Paul Tiede' fails), nor can a shared surname alone ('J. W. Chen' /
    # 'Yi-Chun Chen' fails).
    con.execute(f"""CREATE OR REPLACE MACRO nwords(x) AS list_filter(regexp_extract_all({sql_fold("nnorm(x)")}, '[a-z]+'), t -> length(t) >= 3)""")
    con.execute("""CREATE OR REPLACE MACRO ncompat_orcid(a, b) AS
        ncompat(a, b) OR len(list_intersect(nwords(a), nwords(b))) >= 2
        OR (nfirst(a) <> '' AND nfirst(b) <> '' AND left(nfirst(a), 1) = left(nfirst(b), 1)
            -- given names agree: an initial, a prefix ('chris' / 'christopher') or a near-spelling ('yuliya' / 'yulia')
            AND (length(nfirst(a)) = 1 OR length(nfirst(b)) = 1 OR starts_with(nfirst(a), nfirst(b)) OR starts_with(nfirst(b), nfirst(a))
                 OR jaro_winkler_similarity(nfirst(a), nfirst(b)) >= 0.85)
            AND (len(list_intersect(nwords(a), nwords(b))) >= 1
                 OR (nsur(a) <> '' AND nsur(b) <> '' AND jaro_winkler_similarity(nsur(a), nsur(b)) >= 0.85)))""")


def sql_surname_tokens(names_list: str, n: int = 10) -> str:
    """List of possible surname tokens over the first n names of a list of raw names (rule of surname_tokens,
    without the particle handling: particles only add tokens, never remove the true surname)."""
    one = (f"list_filter([regexp_replace({sql_fold('CASE WHEN contains(x, \',\') THEN split_part(x, \',\', 1) ELSE split_part(trim(x), \' \', -1) END')}, '[^a-z]', '', 'g'),"
           f" regexp_replace({sql_fold('CASE WHEN contains(x, \',\') THEN trim(split_part(split_part(x, \',\', 2), \' \', 1)) ELSE split_part(trim(x), \' \', 1) END')}, '[^a-z]', '', 'g')],"
           f" t -> length(t) >= 2 AND t NOT IN {tuple(sorted(PLACEHOLDER_TOKENS))!r})")
    person = sql_is_person("x")
    return f"list_distinct(flatten(list_transform(list_filter({names_list}[1:{n}], x -> {person}), x -> {one})))"
