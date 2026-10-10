"""Atlas position of every embedded paper, computed once per embedding shard (shards never change).

    cd pipeline/embeddings && uv run python ../release/positions.py [--device cuda] [--limit N]

Same frozen map function as citewalk/science/placement.py (the atlas build's rule: "mean" for the
first build, "mode" from 2026-10-09 on; see placement.Rule), computed exactly from the k most
similar of the atlas's 291,556 reference papers. On the GPU: fp16 scores against all references pick 64
candidates, which are re-scored in fp32 (the reference vectors are stored in fp16, so that is
exact); the top k of those give the position. tests: same result as faiss exact search on CPU.
The CPU path is ~600 papers/s (0.27 TFLOPS measured on lambda), so the full corpus needs a GPU
(~40 min on an RTX 3090 for 203M; unmeasured at that scale).

GPU policy, as the embedding backfill: physical GPU chosen by index (ATLAS_GPU, default 1),
waits while it has less than 2 GiB free, never stops other services, caps its own allocation.

Writes /mnt/fast/scholarsreadinglist/release/positions/<atlas id>/<shard dir>/xy.npy (float32
[rows, 2], unit square, row order = the shard's) plus done.json, atomically per shard.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "citewalk" / "science"))
import placement  # noqa: E402

SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")
OUT = Path(os.environ.get("POSITIONS_OUT", "/mnt/fast/scholarsreadinglist/release/positions"))
CAND = 64


def gpu_uuid(index: str, min_free_mib: int, wait: bool = True) -> str:
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        gpus = {i: (u, int(f)) for i, u, f in (l.split(", ") for l in out.strip().splitlines())}
        uuid, free = gpus[index]
        if free >= min_free_mib:
            return uuid
        if not wait:
            raise SystemExit(f"GPU {index}: {free} MiB free (< {min_free_mib}); not waiting")
        print(f"GPU {index}: {free} MiB free (< {min_free_mib}); waiting", flush=True)
        time.sleep(120)


class Placer:
    def __init__(self, build: Path, device: str, batch: int):
        import torch
        self.torch, self.device, self.batch = torch, device, batch
        ref = np.load(build / "reference.npz")
        self.rule = placement.Rule(ref)
        self.k, self.tau = self.rule.k, self.rule.tau
        self.R = torch.from_numpy(np.ascontiguousarray(ref["vectors"], dtype=np.float16)).to(device)
        self.Y = torch.from_numpy(ref["coords"].astype(np.float32)).to(device)
        if device == "cpu":
            self.R32 = self.R.float()

    def place(self, X16: np.ndarray, works: np.ndarray | None = None) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(X16), 2), np.float32)
        with torch.inference_mode():
            for a in range(0, len(X16), self.batch):
                x = torch.from_numpy(np.ascontiguousarray(X16[a:a + self.batch])).to(self.device)
                if self.device == "cpu":
                    s = x.float() @ self.R32.T
                    sims, nn = torch.topk(s, self.k, dim=1)
                else:
                    cand = torch.topk(x @ self.R.T, CAND, dim=1).indices          # fp16 scores
                    exact = torch.einsum("bd,bkd->bk", x.float(), self.R[cand].float())   # fp32 re-score
                    sims, j = torch.topk(exact, self.k, dim=1)
                    nn = torch.gather(cand, 1, j)
                if self.rule.kind == "mean":
                    w = torch.softmax((sims - sims[:, :1]) / self.tau, dim=1)
                    out[a:a + len(x)] = (w[:, :, None] * self.Y[nn]).sum(1).cpu().numpy()
                else:
                    out[a:a + len(x)] = self.rule(sims.cpu().numpy(), nn.cpu().numpy(),
                                                  None if works is None else works[a:a + len(x)])
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=Path, default=Path("/mnt/fast/science-atlas/build/20261008T131030-tsne"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, help="at most this many shards (tests)")
    ap.add_argument("--min-free-mib", type=int, default=2048)
    ap.add_argument("--shards", nargs="*", help="only these shard dirs")
    ap.add_argument("--no-wait", action="store_true", help="exit instead of waiting for GPU memory")
    a = ap.parse_args()
    atlas_id = json.loads((a.atlas / "meta.json").read_text())["id"]
    if a.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid(os.environ.get("ATLAS_GPU", "1"), a.min_free_mib, wait=not a.no_wait)
    import torch
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(.9, 2.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    else:
        torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    placer = Placer(a.atlas, a.device, a.batch)
    root = OUT / atlas_id
    dirs = a.shards or sorted(str(c.parent.relative_to(SHARDS)) for c in SHARDS.glob("year=*/rg-*/part-*/commit.json"))
    todo = [d for d in dirs if not (root / d / "done.json").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(dirs):,} shards, {len(todo):,} to place (atlas {atlas_id}, {a.device})", flush=True)
    t0, rows = time.time(), 0
    for n, d in enumerate(todo):
        src = SHARDS / d
        X = np.load(src / "vectors.npy")
        xy = placer.place(X, np.load(src / "work_ids.npy") if placer.rule.kind == "mode" else None)
        dst = root / d
        dst.mkdir(parents=True, exist_ok=True)
        # written, flushed to disk, then renamed: a reboot right after must not leave an empty xy.npy behind a
        # done.json (2026-10-09: one shard, written at the minute of a reboot, was 0 bytes and stopped a release)
        with open(dst / "xy.tmp.npy", "wb") as f:
            np.save(f, xy); f.flush(); os.fsync(f.fileno())
        os.replace(dst / "xy.tmp.npy", dst / "xy.npy")
        with open(dst / "done.tmp.json", "w") as f:
            f.write(json.dumps({"rows": len(xy), "atlas": atlas_id, "device": a.device,
                                                   "k": placer.k, "tau": placer.tau, "rule": placer.rule.kind,
                                                   "candidates": CAND}))
            f.flush(); os.fsync(f.fileno())
        os.replace(dst / "done.tmp.json", dst / "done.json")
        fd = os.open(dst, os.O_RDONLY); os.fsync(fd); os.close(fd)
        rows += len(xy)
        if n % 200 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            print(f"{n + 1:,}/{len(todo):,} shards, {rows:,} papers, {rows / max(el, 1e-9):,.0f}/s", flush=True)


if __name__ == "__main__":
    main()
