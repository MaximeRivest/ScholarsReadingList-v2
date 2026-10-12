"""Territories, growth and the bar race from exact counts of every publication (full_counts.py), for the viewer.

Same pictures as static/layers.js computes from the 1 % sample dots, from the full counts instead. What changes
with 100x more papers, and why:

- Statistical rules (a territory is firm, a growth colour is shown) use the real counts, so far fewer places are
  "too uncertain": the old rules were there to hide 1 % sampling noise.
- Visual weight (how strongly a sparse place is tinted) is measured in "dot equivalents" (count x the sample
  rate), so a place looks as strong as before; only its certainty changed.
- Growth: with exact counts nearly every place's share differs "significantly" from ten years earlier, often by
  a few percent. A place is coloured only if its share changed by more than 15 % (|log2| > 0.2, the bar race's
  "=" band), reaching full colour by ~40 %.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

TG, T_MIN = 256, 3                          # layers.js
HG, HEAT_TARGET, HEAT_PRIOR = 160, 40, 10   # layers.js
SAME = 0.2                                  # |log2 ratio| below this is "no change" (bar race "=")


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """layers.js gaussianBlur: zero outside the grid, kernel radius ceil(3 sigma), normalised over the kernel."""
    r = math.ceil(3 * sigma)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma))
    k /= k.sum()
    a = ndimage.correlate1d(a, k, axis=-1, mode="constant")
    return ndimage.correlate1d(a, k, axis=-2, mode="constant")


class FullCounts:
    def __init__(self, root: Path, field_rgb: list[tuple[int, int, int]], uncl: int, display_rate: float):
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text())
        self.y0, self.y1 = self.meta["years"]
        ny = self.y1 - self.y0 + 1
        self.terr = np.load(root / "terr.npy", mmap_mode="r")                      # [ny, nf, TG*TG]
        self.heat = np.fromfile(root / "heat.u32", np.uint32).reshape(ny, HG * HG)
        self.sub = np.fromfile(root / "sub.u32", np.uint32).reshape(ny, -1)
        self.rgb, self.uncl, self.rate = field_rgb, uncl, display_rate
        self.cache: dict[tuple, dict] = {}

    def _years(self, a: int, b: int) -> slice:
        """Index slice of years a..b inclusive, clipped (empty when outside)."""
        return slice(max(0, a - self.y0), max(0, min(b, self.y1) - self.y0 + 1))

    def summary(self) -> dict:
        """Per year: papers counted, and papers per subfield (bar race, timeline)."""
        return {"atlas": self.meta["atlas"], "release": self.meta["release"], "years": self.meta["years"],
                "papers_per_year": self.meta["papers_per_year"], "papers": self.meta["papers"],
                "sub_per_year": self.sub.T.tolist(), "left_out": self.meta["left_out"], "what": self.meta["what"]}

    def _remember(self, key, value):
        if len(self.cache) > 400:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
        return value

    # ------------------------------------------------------------ territories
    def territory(self, a: int, b: int) -> dict:
        key = ("t", a, b)
        if key in self.cache:
            return self.cache[key]
        ys = self._years(a, b)
        raw = np.asarray(self.terr[ys], np.float64).sum(0) if ys.stop > ys.start else np.zeros(self.terr.shape[1:])
        nf = raw.shape[0]
        n_all = raw.sum()
        # layers.js widens the blur below 250,000 dots; here below 250,000 papers (less smoothing: less noise)
        sigma = max(1.6, min(4.0, 1.6 * (250000 / max(1.0, n_all)) ** 0.25))
        area = 2 * math.pi * sigma * sigma
        n = raw.sum(1)
        top = np.zeros(TG * TG); second = np.zeros(TG * TG); total = np.zeros(TG * TG)
        best = np.full(TG * TG, -1, np.int16)
        for f in range(nf):
            if f == self.uncl or n[f] < 20:
                continue
            v = blur(raw[f].reshape(TG, TG), sigma).ravel() * area
            total += v
            up = v > top
            second = np.where(up, top, np.maximum(second, v))
            best = np.where(up, f, best).astype(np.int16)
            top = np.where(up, v, top)
        empty = (best < 0) | (total < T_MIN)
        best[empty] = -1
        with np.errstate(invalid="ignore", divide="ignore"):
            sure = (~empty) & ((top - second) / np.sqrt(top + second) >= 2.5)
        # label anchors: densest cell of each field's largest connected firm territory (>= 6 cells)
        anchor = [None] * nf
        B, Sg, T = best.reshape(TG, TG), sure.reshape(TG, TG), top.reshape(TG, TG)
        for f in np.unique(best[sure]):
            lab, k = ndimage.label(Sg & (B == f))
            if not k:
                continue
            cells = np.bincount(lab.ravel(), minlength=k + 1)
            mass = np.bincount(lab.ravel(), weights=T.ravel(), minlength=k + 1)
            mass[0] = 0
            mass[cells < 6] = 0
            c = int(mass.argmax())
            if mass[c] <= 0:
                continue
            m = np.where(lab == c, T, -1)
            peak = int(m.argmax())
            anchor[int(f)] = {"x": (peak % TG + .5) / TG, "y": (peak // TG + .5) / TG}
        # colours, as layers.js; strength from dot equivalents (count x sample rate) so sparse places look the same
        rgba = np.zeros((TG * TG, 4), np.float64)
        rgb = np.array(self.rgb + [(0, 0, 0)], np.float64)
        col = rgb[np.where(best >= 0, best, len(self.rgb))]
        rgba[:, :3] = col
        rgba[~empty & ~sure, 3] = 22
        bg = B
        other_r = np.zeros((TG, TG), bool); other_d = np.zeros((TG, TG), bool)
        other_r[:, :-1] = Sg[:, 1:] & (bg[:, 1:] != bg[:, :-1])
        other_d[:-1, :] = Sg[1:, :] & (bg[1:, :] != bg[:-1, :])
        border = (other_r | other_d).ravel() & sure
        tdots = total * self.rate
        strength = np.minimum(1, np.maximum(0, (tdots - T_MIN) / 12) + .35)
        with np.errstate(invalid="ignore", divide="ignore"):
            margin = np.where(total > 0, (top - second) / total, 0)
        firm = sure & ~border
        rgba[firm, 3] = 255 * strength[firm] * (.22 + .5 * margin[firm])
        rgba[border, :3] = 8
        rgba[border, 3] = 200 * strength[border]
        rgba[empty] = 0
        return self._remember(key, {"rgba": np.clip(np.round(rgba), 0, 255).astype(np.uint8).tobytes(),
                                    "size": TG, "anchor": anchor, "papers": int(n_all),
                                    "firm_share": float(sure.sum() / max(1, (~empty).sum()))})

    # ------------------------------------------------------------ growth
    def growth(self, a0: int, a1: int, b0: int, b1: int) -> dict | None:
        key = ("g", a0, a1, b0, b1)
        if key in self.cache:
            return self.cache[key]
        now = self.heat[self._years(a0, a1)].sum(0).astype(np.float64)
        before = self.heat[self._years(b0, b1)].sum(0).astype(np.float64)
        A, Bn = now.sum(), before.sum()
        if A < 200 or Bn < 200:
            return self._remember(key, None)
        m = now + before * A / Bn
        occupied = m > 0
        density = m[occupied].sum() / max(1, occupied.sum()) / 2
        sigma = max(2.5, min(12.0, math.sqrt(HEAT_TARGET / (2 * math.pi * density))))
        area = 2 * math.pi * sigma * sigma
        an = blur(now.reshape(HG, HG), sigma).ravel() * area
        e = blur(before.reshape(HG, HG), sigma).ravel() * area * A / Bn     # expected now if the share had not changed
        scale = max(1.0, A / Bn)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (an - e) / np.sqrt(an + e * scale)
        sure = np.clip((np.abs(z) - 2) / 2, 0, 1)
        r = np.log2((an + HEAT_PRIOR) / (e + HEAT_PRIOR))
        big = np.clip((np.abs(r) - SAME) / 0.3, 0, 1)                      # > 15 % change, full by ~40 %
        t = np.clip(r / 2, -1, 1)
        alpha = sure * big * (.3 + .7 * np.abs(t)) * .8
        alpha[(an + e) < 4] = 0
        rgba = np.zeros((HG * HG, 4), np.float64)
        rgba[t > 0, :3] = (255, 92, 56)
        rgba[t <= 0, :3] = (56, 150, 255)
        rgba[:, 3] = alpha * 255
        rgba[alpha <= 0] = 0
        return self._remember(key, {"rgba": np.clip(np.round(rgba), 0, 255).astype(np.uint8).tobytes(),
                                    "size": HG, "anchor": None, "papers": [int(A), int(Bn)]})


def as_json(layer: dict | None) -> dict:
    if layer is None:
        return {"rgba": None}
    return {**{k: v for k, v in layer.items() if k != "rgba"}, "rgba": base64.b64encode(layer["rgba"]).decode()}
