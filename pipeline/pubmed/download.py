"""Download PubMed's annual baseline and its daily update files (free, NLM FTP over HTTPS), verified by their .md5.

    python3 pipeline/pubmed/download.py [--root /mnt/bulk/pubmed] [--workers 3]

Resumable: a file already present with a matching md5 is skipped; downloads go to *.part and are renamed.
"""
import argparse, hashlib, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/"


def listing(sub):
    html = urllib.request.urlopen(BASE + sub, timeout=60).read().decode()
    return sorted(set(re.findall(r'(pubmed\w+\.xml\.gz)"', html)))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(sub, name, root):
    dst = root / sub / name
    want = urllib.request.urlopen(BASE + sub + name + ".md5", timeout=60).read().decode().split("=")[-1].strip()
    if dst.exists() and md5(dst) == want:
        return name, "had"
    for attempt in range(5):
        try:
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(BASE + sub + name, timeout=120) as r, open(tmp, "wb") as f:
                while b := r.read(1 << 20):
                    f.write(b)
            if md5(tmp) != want:
                raise IOError("md5 mismatch")
            os.replace(tmp, dst)
            return name, "got"
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return name, "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/bulk/pubmed"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = []
    for sub in ("baseline/", "updatefiles/"):
        (a.root / sub).mkdir(parents=True, exist_ok=True)
        jobs += [(sub, n) for n in listing(sub)]
    print(f"{len(jobs)} files", flush=True)
    t0, bad = time.time(), []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (name, st) in enumerate(ex.map(lambda j: fetch(*j, a.root), jobs), 1):
            if st == "FAILED":
                bad.append(name)
            if i % 50 == 0:
                print(f"{time.time() - t0:6.0f}s {i}/{len(jobs)}", flush=True)
    print(f"done; failed: {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
