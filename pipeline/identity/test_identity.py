import sys
from pathlib import Path
import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import names as N


def test_split_and_keys():
    assert N.split_name("Yanagi, Hidekatsu") == ("hidekatsu", "yanagi")
    assert N.split_name("Hidekatsu Yanagi") == ("hidekatsu", "yanagi")
    assert N.split_name("Jolanta Gołka") == ("jolanta", "golka")
    assert N.split_name("Anna van der Berg") == ("anna", "vanderberg")
    assert N.name_key("Ersanlı, C.C.") == "ersanli c"
    assert N.same_person_name("J. P. Smith", "John Paul Smith")
    assert not N.same_person_name("John Smith", "James Smith")
    assert N.same_person_name("Wei Lei", "Lei Wei")
    assert N.title_key("Short title") is None
    assert N.title_key("The Interplay between WASH practices, and vaccination") == N.title_key("the interplay between wash practices and vaccination")


def test_sql_matches_python():
    con = duckdb.connect()
    titles = ["The Interplay between WASH practices, and vaccination", "Too short", "Ünïcödé façade of Łódź streets in winter"]
    for t in titles:
        assert con.execute(f"SELECT {N.sql_title_key('t')} FROM (SELECT ?::VARCHAR AS t)", [t]).fetchone()[0] == N.title_key(t), t
    assert not N.is_person_name("South Carolina Office of Resilience") and not N.is_person_name("None Available")
    assert not N.is_person_name("Dr Asim Khan, Dr Syeda Mahrukh Shakir, Dr Mubashir Bhutta") and N.is_person_name("Yanagi, Hidekatsu")
    orgs = ["South Carolina Office of Resilience", "Parole | Pardon Services", "University of South Carolina Salkehatchie"]
    assert con.execute(f"SELECT {N.sql_surname_tokens('l')} FROM (SELECT ?::VARCHAR[] AS l)", [orgs]).fetchone()[0] == []
    names = ["Yanagi, Hidekatsu", "Hidekatsu Yanagi", "Wei Lei", "Lei Wei", "J. Smith", "Smaı̈li Kamel"]
    got = con.execute(f"SELECT {N.sql_surname_tokens('l')} FROM (SELECT ?::VARCHAR[] AS l)", [names]).fetchone()[0]
    want = set().union(*(N.surname_tokens(x) for x in names))
    assert set(got) == want, (got, want)


def test_sql_name_macros():
    con = duckdb.connect()
    N.sql_name_macros(con)
    cases = [("M.K. Patil", "Mihir K. Patil", True), ("J. P. Smith", "John Paul Smith", True), ("John Smith", "James Smith", False), ("Wei Lei", "Lei Wei", True),
             ("Yanagi, Hidekatsu", "Hidekatsu Yanagi", True), ("Jolanta Gołka", "Golka, J.", True),
             ("Karel Pacák", "Abhishek Jha", False), ("Ruiqing Liu", "Xiujing Lin", False)]
    for a, b, want in cases:
        assert con.execute("SELECT ncompat(?, ?)", [a, b]).fetchone()[0] == want, (a, b)
        assert N.same_person_name(a, b) == want, (a, b)
    assert N.is_person_name("王伟") and N.is_person_name("Иван Петров")
    for x in ["South Carolina Office of Resilience", "None Available", "Yanagi, Hidekatsu", "Dr A, Dr B, Dr C", "王伟", "Иван Петров"]:
        assert con.execute(f"SELECT {N.sql_is_person('x')} FROM (SELECT ?::VARCHAR AS x)", [x]).fetchone()[0] == N.is_person_name(x), x


def test_same_orcid_names():
    con = duckdb.connect()
    N.sql_name_macros(con)
    yes = [("Oksana Yarmolovich", "Oksana Yarmolovych"), ("Marta Baselga Bacardit", "M. Baselga"), ("Josh Earl", "Joshua P. Earl"),
           ("Harel Noam Y.", "Noam Y. Harel"), ("Evgenii I. Zakurdaev", "Е. И. Закурдаев"), ("Johan Fernando Devia-Alvira", "Johan Devia"),
           ("S.V. Linnikov", "С. В. Линников"), ("T. Kulinich", "Тетяна Кулініч"), ("Prof. M.  L. Lekala", "M. L. Lekala"),
           ("María Manuela Morales Suárez-Varela", "María Morales‐Suárez‐Varela"), ("Christopher Seal", "Chris Seal"),
           ("Yuliya S. Lukina", "Yulia S. Lukina"), ("Natalia Burillo-Navarrete", "Natalia Burillo")]
    no = [("Karel Pacák", "Abhishek Jha"), ("Ruiqing Liu", "Xiujing Lin"), ("J. W. Chen", "Yi‐Chun Chen"), ("沈志强", "Paul Tiede"),
          ("Laurie L. Williams", "Leah Ruby Williams"), ("Issiaka Traoré", "Ibrahim Alain Traoré"), ("Yu‐Fang Lin", "Yen‐You Lin")]
    for a, b in yes + no:
        want = (a, b) in yes
        assert con.execute("SELECT ncompat_orcid(?, ?)", [a, b]).fetchone()[0] == want, (a, b)
        assert N.same_orcid_compatible(a, b) == want, (a, b)


def test_person_names():
    people = ["Russell, Charles Edward,\n    1860-1941\n    http://id.loc.gov/authorities/names/n50022289\n    http://viaf.org/viaf/30419881",
              "Melvin S. Center", "Jun Chen (4238)", "A Adiguzel (17270359)", "NIDN. 0021047107 Nurul Hartini",
              "Maria Teresa Pitta de Meirelles Sollari Allegro Alves de Sousa", "Pınar Inci", "Stafford Hutchins", "Jörg Staff", "KAN Li-li1", "Martina ; https://orcid.org/0000-0003-4518-8601 Hielscher-Fastabend", "Henri de 1896-1991 Lubac", "Gerrard, Martha, editor.", "Matthew J. Press",
              "Lolkje Theodora Wilhelmina de Jong-van den Berg", "Scott, David, 1964 Jan. 11-", "王伟"]
    orgs = ["CMS Collaboration", "the JT- Team", "Client Advocacy Office", "Hung HF; Kao PF; Lin YS; Chen FC; Liu JC; Chen FC;  Tsai* JC;  Chan P",
            "Missouri State University", "Clemson University", "International Genomics of Alzheimer's Project", "Anonymous",
            ":unav", "Dr Asim Khan, Dr Syeda Mahrukh Shakir, Dr Mubashir Bhutta", "Office of Sustainable Transportation"]
    con = duckdb.connect()
    for x in people + orgs:
        want = x in people
        assert N.is_person_name(x) == want, x
        assert con.execute(f"SELECT {N.sql_is_person('x')} FROM (SELECT ?::VARCHAR AS x)", [x]).fetchone()[0] == want, x
