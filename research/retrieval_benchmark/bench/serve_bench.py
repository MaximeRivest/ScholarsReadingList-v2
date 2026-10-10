"""Serving speed of compressed indexes over the 1.2M-paper collection, 4 CPU cores.

Two memory regimes per configuration:
  ram   index files pre-read into the page cache; no memory limit.
  disk  index files evicted from the page cache (posix_fadvise DONTNEED), then the
        worker runs in a systemd scope with a fixed 4 GiB total memory limit and
        no swap. This includes program memory AND cached files, not a guaranteed
        cache allowance. Smaller indexes may fit; measured disk reads reveal this.
        Files live on NVMe (/mnt/fast, WD SN850X), NOT spinning HDD.
Quality is computed from the ram run (all 1,673 queries). The disk run times a subset.
Query encoding is not included (see results/query_encoding.json).

  uv run python -m bench.serve_bench run [--only NAME ...] [--regimes ram disk]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import DATA, DATASETS, write_json

ROOT = Path(os.environ.get("BENCH_INDEX_ROOT", str(DATA / "index2")))
NAME = "beir4+openalex1m"
OUT = Path(os.environ.get("BENCH_RESULTS", str(DATA / "results" / "compress-v2")))
CORES = os.environ.get("BENCH_CORES", "8-11")
THREADS = 4
MEMORY_MAX = int(os.environ.get("BENCH_MEMORY_MIB", "4096")) * 2**20
DISK_QUERIES_PER_DATASET = int(os.environ.get("BENCH_QUERIES_PER_DATASET", "100"))
TOP_K = 100


def configs() -> dict[str, dict]:
    out = {}
    for model, pool in (("lateon", 1), ("lateon_pool2", 2), ("lateon_pool4", 4)):
        for nb in (4, 2, 1):
            for nprobe in (None, 32):
                name = f"lateon-pool{pool}-{nb}bit-np{nprobe or 'auto'}"
                out[name] = {"engine": "warp", "model": model, "nbits": nb, "nprobe": nprobe,
                             "dir": ROOT / f"warp-nb{nb}" / model / NAME}
    fdir = ROOT / "faiss" / "denseon"
    out["denseon-hnsw32-ef128"] = {"engine": "faiss", "index": "hnsw32", "efSearch": 128,
                                   "mmap": False, "dir": fdir, "files": [f"{NAME}.hnsw32.index"]}
    for nprobe in (16, 64):
        out[f"denseon-ivfsq8-np{nprobe}"] = {"engine": "faiss", "index": "ivf4096_sq8",
                                            "nprobe": nprobe, "mmap": True, "dir": fdir,
                                            "files": [f"{NAME}.ivf4096_sq8.index"]}
    if os.environ.get("BENCH_TEST_WARP_DIR"):          # harness self-test only
        out["test-warp"] = {"engine": "warp", "model": "lateon", "nbits": 4, "nprobe": None,
                            "dir": Path(os.environ["BENCH_TEST_WARP_DIR"])}
    return out


# ---------------------------------------------------------------- worker side

def warp_params(meta: dict, top_k: int, n_tokens: int, nprobe: int | None):
    """xtr_warp.hyperparams.optimize, with nprobe optionally forced; the parameters that
    depend on nprobe (bound, max_candidates, t_prime) are derived from the forced value."""
    import math
    from xtr_warp.hyperparams import optimize
    auto = optimize(meta, top_k, n_tokens)
    if nprobe is None:
        return auto
    num_emb = meta["num_embeddings"]
    parts = meta.get("num_centroids", meta["num_partitions"])
    density = num_emb / max(1, parts)
    nprobe = max(2, min(nprobe, parts))
    bound = max(nprobe * 8, int(0.05 * parts))
    max_cand = min(max(int(density * nprobe * max(1, n_tokens)), top_k * 50), num_emb) // 2
    t_prime = int(density * nprobe * max(1, n_tokens // 2))
    if meta["avg_doclen"] > 0 and density < 256:
        t_prime = int(t_prime * max(0.35, min(120.0 / meta["avg_doclen"], 1.0)))
    t_prime = min(max(5_000, min(t_prime, 200_000)), num_emb)
    return bound, nprobe, auto[2], max_cand, t_prime


def _status(key: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0


def _cgroup_read_bytes() -> int | None:
    rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    stat = Path("/sys/fs/cgroup") / rel.lstrip("/") / "io.stat"
    if not stat.exists():
        return None
    return sum(int(f.split("=")[1]) for line in stat.read_text().splitlines()
               for f in line.split()[1:] if f.startswith("rbytes="))


def _pressure() -> float:
    """Share of the last 10 s in which some task waited for memory (/proc/pressure)."""
    line = Path("/proc/pressure/memory").read_text().splitlines()[0]
    return float(line.split()[1].split("=")[1])


def _cgroup_memory_peak() -> int | None:
    rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    f = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak"
    return int(f.read_text()) if f.exists() else None


def _mapped_index_bytes(index_dir: Path) -> int:
    files = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith(str(index_dir)):
            files.add(parts[5])
    return sum(os.path.getsize(f) for f in files if os.path.exists(f))


def worker(name: str, regime: str) -> dict:
    print(f"STAGE imports {name} {regime}", file=sys.stderr, flush=True)
    import numpy as np
    import torch
    torch.set_num_threads(THREADS)
    from . import beir_data
    from .metrics import evaluate
    cfg = configs()[name]
    print("STAGE loading index", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    if cfg["engine"] == "warp":
        from xtr_warp import XTRWarp
        index = XTRWarp(index=str(cfg["dir"])).load(device="cpu", mmap=True)
        meta = json.loads((cfg["dir"] / "metadata.json").read_text())
        ids = json.loads((cfg["dir"].parent / f"{NAME}.ids.json").read_text())

        def search(q):
            b, npb, thr, mc, tp = warp_params(meta, TOP_K, len(q), cfg["nprobe"])
            x = torch.from_numpy(np.ascontiguousarray(q)).float().unsqueeze(0)
            return index.search(queries_embeddings=x, top_k=TOP_K, num_threads=THREADS,
                                bound=b, nprobe=npb, centroid_score_threshold=thr,
                                max_candidates=mc, t_prime=tp)[0]

        def queries(ds):
            folder = DATA / "emb" / cfg["model"] / ds
            Q = np.load(folder / "queries.npy")
            ql = np.load(folder / "qlens.npy")
            qids = json.loads((folder / "query_ids.json").read_text())
            return qids, [Q[i, : ql[i]] for i in range(len(qids))]
    else:
        import faiss
        faiss.omp_set_num_threads(THREADS)
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if cfg["mmap"] else 0
        index = faiss.read_index(str(cfg["dir"] / cfg["files"][0]), flags)
        if "efSearch" in cfg:
            index.hnsw.efSearch = cfg["efSearch"]
        else:
            index.nprobe = cfg["nprobe"]
        ids = json.loads((cfg["dir"] / f"{NAME}.ids.json").read_text())

        def search(q):
            s, ix = index.search(q[None, :], TOP_K)
            return [(int(i), float(v)) for i, v in zip(ix[0], s[0]) if i >= 0]

        def queries(ds):
            folder = DATA / "emb" / "denseon" / ds
            Q = np.load(folder / "queries.npy").astype(np.float32)
            qids = json.loads((folder / "query_ids.json").read_text())
            return qids, list(Q)
    load_s = time.perf_counter() - t0
    files = search_files(cfg)
    if regime == "ram":
        warm(files)
    cached_at_start = cached_fraction(files)
    if regime == "ram" and cached_at_start < 0.99:
        raise RuntimeError(f"RAM test invalid: only {cached_at_start:.1%} cached")
    print("STAGE loading queries and warming kernels", file=sys.stderr, flush=True)
    anon_after_load = _status("RssAnon")
    per_ds = {ds: queries(ds) for ds in DATASETS}
    qrels_all = {ds: beir_data.load_qrels(ds) for ds in DATASETS} if regime == "ram" else {}
    for q in per_ds["scifact"][1][:10]:          # warm-up
        search(q)
    cap = DISK_QUERIES_PER_DATASET if regime == "disk" else None
    rb0, lat, quality, anon_peak = _cgroup_read_bytes(), [], {}, anon_after_load
    matched_lat, per_dataset, query_timings = [], {}, []
    pressure0 = _pressure()
    print("STAGE timed queries", file=sys.stderr, flush=True)
    for ds, (qids, qs) in per_ds.items():
        n = len(qids) if cap is None else min(cap, len(qids))
        run, dataset_lat = {}, []
        for i in range(n):
            t = time.perf_counter()
            hits = search(qs[i])
            elapsed = time.perf_counter() - t
            lat.append(elapsed)
            dataset_lat.append(elapsed)
            if i < DISK_QUERIES_PER_DATASET:
                matched_lat.append(elapsed)
                query_timings.append({"dataset": ds, "qid": qids[i], "milliseconds": elapsed * 1000})
            if i % 25 == 0:
                anon_peak = max(anon_peak, _status("RssAnon"))
            run[qids[i]] = {f"{ids[d].split(':', 1)[1]}" if ids[d].startswith(ds + ":") else ids[d]: sc
                            for d, sc in hits}
        per_dataset[ds] = latency_stats(dataset_lat[:DISK_QUERIES_PER_DATASET])
        if cap is None:
            quality[ds] = evaluate(run, {q: qrels_all[ds][q] for q in qids})["ndcg@10"]
    rb1 = _cgroup_read_bytes()
    anon_peak = max(anon_peak, _status("RssAnon"))
    return {"name": name, "regime": regime, "config": {k: str(v) for k, v in cfg.items()},
            "load_seconds": load_s, "search_latency": latency_stats(matched_lat),
            "all_query_latency": latency_stats(lat), "per_dataset_latency": per_dataset,
            "query_timings": query_timings,
            "latency_protocol": f"same first {DISK_QUERIES_PER_DATASET} queries/dataset (or all if fewer); top 100; sequential; encoding excluded",
            "engine_variant": os.environ.get("BENCH_ENGINE_VARIANT", "released wheel xtr-warp-rs 2.1.0.2110"),
            "storage_device": os.environ.get("BENCH_STORAGE_LABEL", "WD SN850X NVMe /mnt/fast"),
            "memory_limit_bytes": MEMORY_MAX if regime == "disk" else None,
            "ndcg@10": quality, "ndcg@10_mean": (sum(quality.values()) / len(quality)) if quality else None,
            "rss_anon_after_load": anon_after_load, "rss_anon_end": _status("RssAnon"),
            "rss_anon_peak": anon_peak, "cached_fraction_at_start": cached_at_start,
            "cached_fraction_at_end": cached_fraction(files),
            "memory_pressure_avg10": [pressure0, _pressure()],
            "rss_peak": _status("VmHWM"), "cgroup_memory_peak": _cgroup_memory_peak(),
            "mapped_index_bytes": _mapped_index_bytes(cfg["dir"]),
            "disk_read_bytes_per_query": None if rb0 is None else (rb1 - rb0) / len(lat),
            "threads": THREADS}


# ------------------------------------------------------------ orchestrator side

def latency_stats(seconds):
    import statistics
    ms = sorted(s * 1000 for s in seconds)
    return {"n": len(ms), "p50_ms": statistics.median(ms),
            "p95_ms": ms[min(len(ms) - 1, round(0.95 * (len(ms) - 1)))],
            "mean_ms": statistics.fmean(ms)}


def index_files(cfg) -> list[str]:
    if "files" in cfg:
        return [str(cfg["dir"] / f) for f in cfg["files"]]
    return [f for f in glob.glob(str(cfg["dir"] / "*")) if os.path.isfile(f)]


def search_files(cfg) -> list[str]:
    """Files a read-only search server needs: WARP's packed copy + small files
    (not the per-chunk build copies, which only serve later updates)."""
    files = index_files(cfg)
    if cfg["engine"] == "warp":
        files = [f for f in files if not os.path.basename(f)[0].isdigit()
                 and not os.path.basename(f).startswith("doclens.")]
    return files


def search_bytes(cfg) -> int:
    return sum(os.path.getsize(f) for f in search_files(cfg))


def warm(files):
    for f in files:
        with open(f, "rb") as fh:
            while fh.read(1 << 24):
                pass


def evict(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def cached_fraction(files) -> float:
    res = size = 0
    for i in range(0, len(files), 500):
        out = subprocess.run(["fincore", "-b", "-n", "-o", "RES,SIZE", *files[i:i + 500]],
                             capture_output=True, text=True, check=True).stdout.split()
        res += sum(map(int, out[0::2]))
        size += sum(map(int, out[1::2]))
    return min(1.0, res / max(1, size))


def run_worker(name, regime, mem_max=None) -> dict:
    env = os.environ | {"OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS),
                        "RAYON_NUM_THREADS": str(THREADS), "CUDA_VISIBLE_DEVICES": ""}
    cmd = ["taskset", "-c", CORES, sys.executable, "-m", "bench.serve_bench", "worker",
           name, regime]
    import uuid
    unit = f"srl-bench-query-{uuid.uuid4().hex[:12]}.scope"
    # Isolated scope makes memory and I/O accounting specific to this worker.
    limits = ["-p", f"MemoryMax={mem_max}", "-p", "MemorySwapMax=0"] if mem_max else []
    cmd = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", *limits, "--", *cmd]
    print(f"START {name} {regime} memory_limit={mem_max}", flush=True)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800,
                           cwd=Path(__file__).resolve().parent.parent)
    except subprocess.TimeoutExpired:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=30)
        return {"name": name, "regime": regime, "error": "timeout after 30 minutes",
                "stderr_tail": "worker exceeded configured run budget"}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"name": name, "regime": regime, "error": f"exit {p.returncode}",
                "stderr_tail": p.stderr[-2000:]}
    return json.loads(lines[-1])


def orchestrate(names, regimes):
    cfgs = configs()
    failures = []
    for name in names:
        cfg = cfgs[name]
        if cfg.get("nbits") == 1:
            for regime in regimes:
                write_json(OUT / regime / f"{name}.json", {
                    "name": name, "regime": regime,
                    "unsupported": "xtr-warp-rs 2.1.0.2110 CPU scorer supports only 2 or 4 bits; this is residual quantization, not plain sign-bit scoring"})
            print(f"UNSUPPORTED {name}: WARP CPU does not support 1-bit scoring", flush=True)
            continue
        files = index_files(cfg)
        if not files:
            print(f"skip {name}: index missing", flush=True)
            failures.append(name + ": index missing")
            continue
        ram_path = OUT / "ram" / f"{name}.json"
        if "ram" in regimes:
            r = run_worker(name, "ram") | {"search_bytes": search_bytes(cfg)}
            write_json(ram_path, r)
            _print(r)
            if "error" in r:
                failures.append(name + ": ram " + r["error"])
        if "disk" in regimes:
            if not cfg.get("mmap", True):
                write_json(OUT / "disk" / f"{name}.json", {"name": name, "regime": "disk",
                           "skipped": "index must be fully loaded in RAM (no mmap support)"})
                continue
            # Same total memory budget for every engine; record startup failures rather
            # than inflating the limit until the whole index fits in memory.
            evict(files)
            r = run_worker(name, "disk", MEMORY_MAX) | {
                "search_bytes": search_bytes(cfg), "memory_max_bytes": MEMORY_MAX}
            write_json(OUT / "disk" / f"{name}.json", r)
            _print(r)
            if "error" in r:
                failures.append(name + ": disk " + r["error"])
        write_json(OUT / "status.json", {"state": "running", "last_finished": name,
                                        "failures": failures})
    write_json(OUT / "status.json", {"state": "partial" if failures else "complete",
                                    "failures": failures})
    if failures:
        raise SystemExit(1)


def _print(r):
    if "error" in r:
        print(f"{r['name']:32s} {r['regime']:4s} ERROR {r['error']} {r['stderr_tail'][-300:]}", flush=True)
        return
    lat = r["search_latency"]
    q = f"ndcg={r['ndcg@10_mean']:.4f}" if r.get("ndcg@10_mean") is not None else "ndcg=   -  "
    rb = r.get("disk_read_bytes_per_query")
    print(f"{r['name']:32s} {r['regime']:4s} {q} p50={lat['p50_ms']:7.1f}ms "
          f"p95={lat['p95_ms']:7.1f}ms index={r['search_bytes'] / 1e9:5.2f}GB "
          f"anon={r['rss_anon_peak'] / 1e9:5.2f}GB "
          f"cached={r['cached_fraction_at_start']:.2f}->{r['cached_fraction_at_end']:.2f} "
          f"psi={max(r['memory_pressure_avg10']):.1f} "
          f"disk/q={(rb or 0) / 1e6:6.2f}MB", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("name")
    w.add_argument("regime", choices=["ram", "disk"])
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="+")
    r.add_argument("--regimes", nargs="+", default=["ram", "disk"])
    a = p.parse_args()
    if a.cmd == "worker":
        print(json.dumps(worker(a.name, a.regime)), flush=True)
    else:
        orchestrate(a.only or list(configs()), a.regimes)


if __name__ == "__main__":
    main()
