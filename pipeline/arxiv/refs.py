"""arXiv reference lists from the PDFs: PDF (Kaggle's free copy of arXiv, gs://arxiv-dataset) -> GROBID
processReferences -> TEI (kept, gzip) -> one JSON line per reference. Standard library only.

    python refs.py run MONTHS... [--workers 24]   e.g. 2501 2502 ... : every new-style id of those months in arXiv's
                                                metadata snapshot; resumes (a paper with its TEI is skipped)
    python refs.py parse MONTH                  TEI of a month -> /mnt/bulk/arxiv/grobid/refs/MONTH.jsonl
    python refs.py sample SAMPLE.json OUT.jsonl the pilot (research/arxiv/pilot)

Each reference: {aid, n, title, year, surnames, journal, volume, page, doi, arxiv, raw}. GROBID: 127.0.0.1:8070
(pipeline/arxiv/grobid.sh). PDFs are deleted once read; arXiv's licence lets us index and extract, not redistribute."""
import argparse, glob, gzip, json, os, re, subprocess, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

TEI = {"t": "http://www.tei-c.org/ns/1.0"}
ROOT = Path(os.environ.get("ARXIV_REFS", "/mnt/bulk/arxiv/grobid"))
PDF = ROOT / "pdf"; TEIDIR = ROOT / "tei"
GCS = "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/{yymm}/{aid}{v}.pdf"
ARXIV = re.compile(r"(?:arxiv[:\s/]*|abs/)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def fetch(r):
    """The paper's latest version in the bucket (newer versions may not be copied yet: then older ones)."""
    f = PDF / f"{r['aid']}.pdf"
    if f.exists() and f.stat().st_size >= 1000:
        return f
    n = int(r["version"].lstrip("v"))
    for v in [n, *range(n - 1, 0, -1)][:4]:
        url = GCS.format(yymm=r["aid"][:4], aid=r["aid"], v=f"v{v}")
        for attempt in range(3):
            try:
                f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
                return f
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                time.sleep(5)
            except Exception:
                time.sleep(5)
    return f


def tei_path(aid):
    return TEIDIR / aid[:4] / f"{aid.replace('/', '_')}.tei.xml.gz"


def grobid(f, aid):
    t = tei_path(aid)
    if t.exists():
        return gzip.decompress(t.read_bytes()).decode()
    boundary = "----srl" + str(time.time_ns())
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"input\"; filename=\"{f.name}\"\r\n"
            "Content-Type: application/pdf\r\n\r\n").encode() + f.read_bytes() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"includeRawCitations\"\r\n\r\n1"
            f"\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request("http://127.0.0.1:8070/api/processReferences", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    for attempt in range(4):
        try:
            x = urllib.request.urlopen(req, timeout=600).read().decode()
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_suffix(".tmp"); tmp.write_bytes(gzip.compress(x.encode())); tmp.rename(t)
            return x
        except urllib.error.HTTPError as e:
            if e.code != 503:
                return ""
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return ""


def text(e):
    return " ".join("".join(e.itertext()).split()) if e is not None else ""


def refs(aid, xml):
    if not xml:
        return []
    root = ET.fromstring(xml)
    out = []
    for n, b in enumerate(root.iter("{http://www.tei-c.org/ns/1.0}biblStruct")):
        if b.get("{http://www.w3.org/XML/1998/namespace}id") is None:
            continue                                     # the paper's own header, not a reference
        title = text(b.find("t:analytic/t:title", TEI)) or text(b.find("t:monogr/t:title[@level='m']", TEI))
        if not title and b.find("t:analytic", TEI) is None:
            title = text(b.find("t:monogr/t:title", TEI)) if b.find("t:monogr/t:title[@level='j']", TEI) is None else ""
        sur = [text(s) for s in b.findall(".//t:author/t:persName/t:surname", TEI)]
        date = b.find(".//t:imprint/t:date", TEI)
        year = (date.get("when") or "")[:4] if date is not None else ""
        raw = text(b.find("t:note[@type='raw_reference']", TEI))
        idno = {i.get("type"): text(i) for i in b.findall(".//t:idno", TEI)}
        doi = idno.get("DOI") or (DOI.search(raw).group(1).rstrip(".,;)") if DOI.search(raw) else "")
        ax = idno.get("arXiv") or (ARXIV.search(raw).group(1) if ARXIV.search(raw) else "")
        ax = re.sub(r"^arxiv:", "", ax, flags=re.I).split("v")[0] if re.match(r"(arxiv:)?\d{4}\.\d", ax, re.I) else ax
        page = b.find(".//t:biblScope[@unit='page']", TEI)
        out.append({"aid": aid, "n": n, "title": title, "year": int(year) if year.isdigit() else None, "surnames": sur[:3],
                    "journal": text(b.find("t:monogr/t:title[@level='j']", TEI)),
                    "volume": text(b.find(".//t:biblScope[@unit='volume']", TEI)),
                    "page": (page.get("from") or text(page)) if page is not None else "",
                    "doi": doi.lower(), "arxiv": ax.lower(), "raw": raw})
    return out


def one(r, keep_pdf=False):
    if tei_path(r["aid"]).exists():
        return r["aid"], "had"
    f = fetch(r)
    if not f.exists() or f.stat().st_size < 1000:
        f.unlink(missing_ok=True)
        return r["aid"], "no pdf"
    x = grobid(f, r["aid"])
    if not keep_pdf:
        f.unlink(missing_ok=True)
    return r["aid"], "done" if x else "grobid failed"


def month_papers(months):
    """New-style ids of these months with their latest version, from arXiv's metadata snapshot (duckdb CLI)."""
    files = sorted(glob.glob("/mnt/bulk/arxiv/metadata-snapshot/*.parquet"))
    q = (f"COPY (SELECT id, versions[-1].version v FROM read_parquet({files!r}) WHERE substr(id, 1, 4) IN "
         f"({','.join(repr(m) for m in months)}) ORDER BY id) TO '/dev/stdout' (FORMAT csv, HEADER false)")
    out = subprocess.run(["nix", "run", "nixpkgs#duckdb", "--", "-c", q], capture_output=True, text=True, check=True).stdout
    return [{"aid": a, "version": v} for a, v in (line.split(",") for line in out.split())]


def run(papers, workers, log):
    t0 = time.time(); counts = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (aid, how) in enumerate(ex.map(one, papers), 1):
            counts[how] = counts.get(how, 0) + 1
            if how in ("no pdf", "grobid failed"):
                log.write(f"{aid}\t{how}\n"); log.flush()
            if i % 1000 == 0 or i == len(papers):
                rate = (i - counts.get("had", 0)) / max(1, time.time() - t0)
                print(f"{i:,}/{len(papers):,} {counts} {rate:.1f} papers/s", flush=True)


def parse_month(m):
    out = ROOT / "refs" / f"{m}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out.with_suffix(".tmp"), "w") as fo:
        for t in sorted((TEIDIR / m).glob("*.tei.xml.gz")):
            aid = t.name[:-len(".tei.xml.gz")].replace("_", "/")
            try:
                rs = refs(aid, gzip.decompress(t.read_bytes()).decode())
            except ET.ParseError:
                continue
            for x in rs:
                fo.write(json.dumps(x) + "\n"); n += 1
    out.with_suffix(".tmp").rename(out)
    print(m, n, "references", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "parse", "sample")); ap.add_argument("args", nargs="+")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    PDF.mkdir(parents=True, exist_ok=True); TEIDIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        papers = month_papers(a.args)
        print(f"{len(papers):,} papers in {len(a.args)} months", flush=True)
        with open(ROOT / "failed.tsv", "a") as log:
            run(papers, a.workers, log)
        print("DONE", flush=True)
    elif a.cmd == "parse":
        for m in a.args:
            parse_month(m)
    else:                                                  # the pilot: sample.json -> refs jsonl
        sample = json.load(open(a.args[0]))
        with open(a.args[1], "w") as fo:
            for r in sample:
                one(r, keep_pdf=True)
                x = gzip.decompress(tei_path(r["aid"]).read_bytes()).decode() if tei_path(r["aid"]).exists() else ""
                for ref in refs(r["aid"], x):
                    fo.write(json.dumps(ref) + "\n")
