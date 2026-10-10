'use strict';
// Map layers computed for one time window. Pure functions shared by the page and its Web Worker
// (layers-worker.js), so animation never waits on them. Both work on fixed world grids, so a
// layer does not depend on zoom or pan.

const TG = 256, T_MIN = 3;                          // territories: 256×256 cells over the unit square
const HG = 160, HEAT_TARGET = 40, HEAT_PRIOR = 10;  // growth: 160×160 cells (≈ 6 px at zoom 1)

function gaussianBlur(src, w, h, sigma) {
  const r = Math.ceil(3 * sigma), k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-i * i / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < w) v += src[y * w + xx] * k[i + r]; }
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 0; for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < h) v += tmp[yy * w + x] * k[i + r]; }
    out[y * w + x] = v;
  }
  return out;
}

// Which field dominates each place among dots [s, e). Counts smoothed per field, winner per cell.
// A winner is "sure" when its lead over the runner-up is beyond sampling luck (z ≥ 2.5); contested
// cells are barely tinted. Label anchor = densest cell of each field's largest connected territory.
function computeTerritory(XY, FIELD, s, e, NF, uncl, rgb) {
  const sigma = Math.max(1.6, Math.min(4, 1.6 * Math.pow(250000 / Math.max(1, e - s), .25)));
  const C = TG * TG, area = 2 * Math.PI * sigma * sigma;
  const raw = Array.from({ length: NF }, () => null), n = new Int32Array(NF);
  for (let i = s; i < e; i++) {
    const f = FIELD[i];
    if (f === uncl) continue;
    const gx = (XY[2 * i] * TG) | 0, gy = (XY[2 * i + 1] * TG) | 0;
    if (gx < 0 || gy < 0 || gx >= TG || gy >= TG) continue;
    (raw[f] ||= new Float32Array(C))[gy * TG + gx]++; n[f]++;
  }
  const best = new Int16Array(C).fill(-1), top = new Float32Array(C), second = new Float32Array(C), total = new Float32Array(C);
  for (let f = 0; f < NF; f++) {
    if (!raw[f] || n[f] < 20) continue;               // too few papers for a territory
    const b = gaussianBlur(raw[f], TG, TG, sigma);
    for (let k = 0; k < C; k++) {
      const v = b[k] * area; total[k] += v;
      if (v > top[k]) { second[k] = top[k]; top[k] = v; best[k] = f; } else if (v > second[k]) second[k] = v;
    }
  }
  const sure = new Uint8Array(C);
  for (let k = 0; k < C; k++) {
    if (best[k] < 0 || total[k] < T_MIN) { best[k] = -1; continue; }
    sure[k] = (top[k] - second[k]) / Math.sqrt(top[k] + second[k]) >= 2.5 ? 1 : 0;
  }
  const anchor = new Array(NF).fill(null), comp = new Int32Array(C).fill(-1), stack = new Int32Array(C);
  const bestMass = new Float32Array(NF);
  for (let k0 = 0; k0 < C; k0++) {
    if (!sure[k0] || comp[k0] >= 0) continue;
    const f = best[k0];
    let sp = 0, mass = 0, cells = 0, peak = -1, peakVal = -1;
    stack[sp++] = k0; comp[k0] = k0;
    while (sp) {
      const k = stack[--sp], x = k % TG;
      mass += top[k]; cells++;
      if (top[k] > peakVal) { peakVal = top[k]; peak = k; }
      for (const nb of [x > 0 ? k - 1 : -1, x < TG - 1 ? k + 1 : -1, k - TG, k + TG]) {
        if (nb >= 0 && nb < C && comp[nb] < 0 && sure[nb] && best[nb] === f) { comp[nb] = k0; stack[sp++] = nb; }
      }
    }
    if (cells >= 6 && mass > bestMass[f]) {
      bestMass[f] = mass; anchor[f] = { x: (peak % TG + .5) / TG, y: ((peak / TG | 0) + .5) / TG };
    }
  }
  const d = new Uint8ClampedArray(C * 4);
  for (let k = 0; k < C; k++) {
    const f = best[k]; if (f < 0) continue;
    const x = k % TG, y = k / TG | 0, c = rgb[f];
    d[4 * k] = c[0]; d[4 * k + 1] = c[1]; d[4 * k + 2] = c[2];
    if (!sure[k]) { d[4 * k + 3] = 22; continue; }
    const other = (j) => sure[j] && best[j] !== f;
    const border = (x + 1 < TG && other(k + 1)) || (y + 1 < TG && other(k + TG));
    const margin = (top[k] - second[k]) / total[k];
    const strength = Math.min(1, (total[k] - T_MIN) / 12 + .35);
    if (border) { d[4 * k] = d[4 * k + 1] = d[4 * k + 2] = 8; d[4 * k + 3] = 200 * strength; continue; }
    d[4 * k + 3] = 255 * strength * (.22 + .5 * margin);
  }
  return { best, sure, anchor, rgba: d, size: TG };
}

// Share of research in dots [a0, a1) vs [b0, b1), per place. Shares, not counts (science grows
// overall). Smoothing widens where dots are sparse (~40 per neighbourhood); a place is coloured
// only where the difference is statistically clear (Poisson |z| > 2, full colour at 4).
function computeGrowth(XY, a0, a1, b0, b1) {
  const A = a1 - a0, B = b1 - b0;
  if (A < 200 || B < 200) return null;
  const C = HG * HG;
  const bin = (from, to) => {
    const g = new Float32Array(C);
    for (let i = from; i < to; i++) {
      const gx = (XY[2 * i] * HG) | 0, gy = (XY[2 * i + 1] * HG) | 0;
      if (gx < 0 || gy < 0 || gx >= HG || gy >= HG) continue;
      g[gy * HG + gx]++;
    }
    return g;
  };
  const rawNow = bin(a0, a1), rawBefore = bin(b0, b1);
  let dots = 0, occupied = 0;
  for (let k = 0; k < C; k++) { const m = rawNow[k] + rawBefore[k] * A / B; if (m > 0) { dots += m; occupied++; } }
  const density = dots / Math.max(1, occupied) / 2;
  const sigma = Math.max(2.5, Math.min(12, Math.sqrt(HEAT_TARGET / (2 * Math.PI * density))));
  const area = 2 * Math.PI * sigma * sigma;
  const now = gaussianBlur(rawNow, HG, HG, sigma), before = gaussianBlur(rawBefore, HG, HG, sigma);
  const scale = Math.max(1, A / B), d = new Uint8ClampedArray(C * 4);
  for (let k = 0; k < C; k++) {
    const a = now[k] * area, e = before[k] * area * A / B;    // e: expected now if the share had not changed
    if (a + e < 4) continue;
    const z = (a - e) / Math.sqrt(a + e * scale);
    const sure = Math.max(0, Math.min(1, (Math.abs(z) - 2) / 2));
    if (!sure) continue;
    const r = Math.log2((a + HEAT_PRIOR) / (e + HEAT_PRIOR));
    const t = Math.max(-1, Math.min(1, r / 2));
    const alpha = sure * (.3 + .7 * Math.abs(t)) * .8;
    const c = t > 0 ? [255, 92, 56] : [56, 150, 255];
    d[4 * k] = c[0]; d[4 * k + 1] = c[1]; d[4 * k + 2] = c[2]; d[4 * k + 3] = alpha * 255;
  }
  return { rgba: d, size: HG, anchor: null };
}
