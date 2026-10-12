# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==26.0.0"]
# ///
"""Make the export folder a complete Hugging Face dataset repository: card, scripts, search index.

    uv run pipeline/hf/card.py --out /mnt/bulk/hf-export/openalex-denseon --repo maximerivest/openalex-denseon-embeddings

Reads <out>/export.json (written by export.py), copies the release's search index into <out>/index/
(a copy, so the nightly clean-up cannot delete what is being uploaded), and writes README.md from
repo/README.template.md with this release's numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RELEASES = Path("/mnt/fast/scholarsreadinglist/release")
LICENSES = {
    "cc0-1.0": "CC0 1.0: no rights reserved. Use, share and build on it for anything, commercial or not, "
               "without asking. Credit is appreciated, not required (see Cite).",
    "cc-by-4.0": "CC BY 4.0: use, share and build on it for anything, commercial or not, as long as you "
                 "credit this dataset (see Cite).",
}
# Measured 2026-10-12 on release 20261012T073823Z (200 paper-vector questions, nprobe 128,
# reference = depth 1000 at nprobe 256, exactly re-scored). Re-measure when the index changes.
RECALL_INDEX_ONLY = 0.75
RECALL_RERANKED = 0.99


def rows_meta(rel: dict) -> dict:
    return json.loads((Path(rel["paths"]["rows"]) / "meta.json").read_text())


def top_non_publication_types(files: list[Path], meta: dict, n: int = 4) -> str:
    """The most common types that are not publications, with their counts, e.g. 'paratext 21.0M'."""
    counts: dict[str, int] = {}
    for f in files:
        vc = pq.read_table(f, columns=["type"])["type"].value_counts()
        for item in vc.to_pylist():
            counts[item["values"]] = counts.get(item["values"], 0) + item["counts"]
    other = sorted(((c, t) for t, c in counts.items() if t not in meta["scholarly_types"]), reverse=True)
    return ", ".join(f"`{t}` {c / 1e6:.1f}M" for c, t in other[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--license", default="cc0-1.0", choices=sorted(LICENSES))
    ap.add_argument("--no-index", action="store_true", help="do not copy the search index")
    args = ap.parse_args()

    ex = json.loads((args.out / "export.json").read_text())
    rel = json.loads((RELEASES / "releases" / ex["release"] / "manifest.json").read_text())
    files = sorted((args.out / "data").rglob("part-*.parquet"))
    n_updates = sum(pq.ParquetFile(f).metadata.num_rows for f in files if "/updates/" in str(f))
    updates_until = max(f.parent.name for f in files if "/updates/" in str(f))

    index_dir = Path(rel["paths"]["index"])
    index_meta = json.loads((index_dir / "manifest.json").read_text())
    dst = args.out / "index"
    if not args.no_index:
        dst.mkdir(exist_ok=True)
        for name in ["ivf.faiss", "work_ids.npy"]:
            src = index_dir / name
            if not (dst / name).exists() or (dst / name).stat().st_size != src.stat().st_size:
                print(f"copying {src} -> {dst / name}", flush=True)
                shutil.copyfile(src, dst / name)
        (dst / "index.json").write_text(json.dumps({
            "release": ex["release"], "factory": index_meta["index"]["factory"],
            "metric": index_meta["index"]["metric"], "vectors": index_meta["papers"],
            "ids": "FAISS ids = the dataset's `row` column; work_ids.npy[row] = OpenAlex W-number",
            "covers": "rows with in_search_index = true", "faiss": "faiss-cpu==1.15.1",
            "suggested": {"nprobe": 128, "candidates_to_rescore": 300},
            "recall_at_10": {"index_only": RECALL_INDEX_ONLY, "with_exact_rescoring": RECALL_RERANKED},
        }, indent=1))

    for name in ["search.py", "make_vectors.py"]:
        shutil.copyfile(HERE / "repo" / name, args.out / name)

    emb = ex["embedding"]
    gb = ex["bytes"] / 1e9
    values = {
        "license": args.license, "license_text": LICENSES[args.license], "repo": args.repo,
        "rows": ex["rows"], "rows_m": round(ex["rows"] / 1e6, 1), "rows_minus_1": ex["rows"] - 1,
        "searchable": ex["searchable"], "searchable_m": round(ex["searchable"] / 1e6),
        "non_pub": rel["embeddings"]["rows"] - rel["counts"]["publications"],
        "non_pub_share": (rel["embeddings"]["rows"] - rel["counts"]["publications"]) / rel["embeddings"]["rows"],
        "dups": rel["counts"]["publications"] - rel["counts"]["searchable"],
        "top_non_pub": top_non_publication_types(files, rows_meta(rel)),
        "gb": gb, "files": ex["files"], "file_gb": max(f.stat().st_size for f in files) / 1e9,
        "snapshot": Path(rel["paths"]["updates"]).parents[1].name.removeprefix("openalex-"), "updates_until": f"{updates_until[:4]}-{updates_until[4:6]}-{updates_until[6:8]}",
        "release": ex["release"], "revision": emb["revision"], "st_version": "5.3.0",
        "base": ex["rows"] - n_updates, "updates": n_updates, "atlas": ex["atlas"],
        "index_factory": index_meta["index"]["factory"],
        "index_gb": (index_dir / "ivf.faiss").stat().st_size / 1e9,
        "ids_gb": (index_dir / "work_ids.npy").stat().st_size / 1e9,
        "vectors_gb": ex["rows"] * 768 * 2 / 1e9,
        "recall_index": RECALL_INDEX_ONLY, "recall_rerank": RECALL_RERANKED, "year": ex["release"][:4],
    }
    card = (HERE / "repo" / "README.template.md").read_text().format(**values)
    (args.out / "README.md").write_text(card)
    print(f"wrote {args.out / 'README.md'} ({len(card):,} characters)")


if __name__ == "__main__":
    main()
