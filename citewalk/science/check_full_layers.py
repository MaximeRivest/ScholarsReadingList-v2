"""Check: full_layers.py (numpy, server) draws the same territories and growth as static/layers.js (browser), on
the sample dots. Run: cd pipeline/embeddings && .venv/bin/python ../../citewalk/science/check_full_layers.py
2026-10-12: 99.98-100 % of coloured cells identical (within 2/255), every label anchor identical."""
# the numpy port (full_layers) on the sample's dots, against layers.js on the same dots (node)
import json, subprocess, sys, numpy as np, tempfile
from pathlib import Path
sys.path.insert(0, "/home/maxime/Projects/scholarsreadinglist/citewalk/science")
import full_layers as FL
B = Path("/mnt/fast/science-atlas/build/20261008T131030-tsne")
m = json.loads((B / "meta.json").read_text())
xy = np.fromfile(B / "points.f32", np.float32).reshape(-1, 2); yr = np.fromfile(B / "year.u16", np.uint16).astype(int)
fd = np.fromfile(B / "field.u8", np.uint8).astype(int)
NY, NF = 227, len(m["fields"])
d = Path(tempfile.mkdtemp())
ok = (xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0) & (xy[:, 1] < 1)
cT = (xy[:, 1] * 256).astype(int) * 256 + (xy[:, 0] * 256).astype(int)
cH = (xy[:, 1] * 160).astype(int) * 160 + (xy[:, 0] * 160).astype(int)
t = np.zeros((NY, NF, 65536), np.uint32); np.add.at(t, (yr[ok] - 1800, fd[ok], cT[ok]), 1); np.save(d / "terr.npy", t)
h = np.zeros((NY, 25600), np.uint32); np.add.at(h, (yr[ok] - 1800, cH[ok]), 1); h.tofile(d / "heat.u32")
np.zeros((NY, 246), np.uint32).tofile(d / "sub.u32")
(d / "meta.json").write_text(json.dumps({"years": [1800, 2026], "papers_per_year": [0]*NY, "papers": 0, "atlas": "", "release": "", "left_out": {}, "what": ""}))
rgb = [tuple(int(f["colour"][o:o+2], 16) for o in (1, 3, 5)) for f in m["fields"]]
uncl = [f["name"] for f in m["fields"]].index("Unclassified")
fc = FL.FullCounts(d, rgb, uncl, 1.0)
off = m["offsets"]; yi = lambda y: y - 1800
cases = [(2017, 2021), (1990, 1994), (1950, 1954)]
js = f"""
const fs=require('fs'); require('vm').runInThisContext(fs.readFileSync('/home/maxime/Projects/scholarsreadinglist/citewalk/science/static/layers.js','utf8').replace("'use strict';",''));
const B='{B}/'; const XY=new Float32Array(fs.readFileSync(B+'points.f32').buffer.slice(0)); const FIELD=new Uint8Array(fs.readFileSync(B+'field.u8'));
const off={json.dumps(off)}, rgb={json.dumps(rgb)}, out={{}};
for (const [a,b] of {json.dumps(cases)}) {{ const r=computeTerritory(XY,FIELD,off[a-1800],off[b-1800+1],{NF},{uncl},rgb); out[a]=Buffer.from(r.rgba.buffer).toString('base64'); out[a+'s']=Array.from(r.sure).reduce((x,y)=>x+y,0); out[a+'anc']=r.anchor; }}
const g=computeGrowth(XY,off[2017-1800],off[2021-1800+1],off[2007-1800],off[2011-1800+1]); out.g=Buffer.from(g.rgba.buffer).toString('base64');
process.stdout.write(JSON.stringify(out));"""
_r = subprocess.run(["node", "-e", js], capture_output=True, text=True); print(_r.stderr[-800:]); o = json.loads(_r.stdout)
import base64
for a, b in cases:
    py = np.frombuffer(fc.territory(a, b)["rgba"], np.uint8).reshape(-1, 4).astype(int)
    j = np.frombuffer(base64.b64decode(o[str(a)]), np.uint8).reshape(-1, 4).astype(int)
    owner = (py[:, 3] > 0) | (j[:, 3] > 0)
    print(f"territory {a}-{b}: cells same colour+alpha(±2) {np.mean(np.all(np.abs(py - j) <= 2, 1)[owner]):.4f} of {owner.sum()} coloured cells; "
          f"anchors equal: {sum(1 for x, y in zip(fc.territory(a,b)['anchor'], o[str(a)+'anc']) if (x is None) == (y is None) and (x is None or (abs(x['x']-y['x'])<1e-6 and abs(x['y']-y['y'])<1e-6)))}/{NF}")
# growth parity (without the new >15 % rule)
FL.SAME = -1e9
fc.cache.clear()
pg = np.frombuffer(fc.growth(2017, 2021, 2007, 2011)["rgba"], np.uint8).reshape(-1, 4).astype(int)
jg = np.frombuffer(base64.b64decode(o["g"]), np.uint8).reshape(-1, 4).astype(int)
col = (pg[:, 3] > 0) | (jg[:, 3] > 0)
print(f"growth (old rule): cells equal(±2) {np.mean(np.all(np.abs(pg - jg) <= 2, 1)[col]):.4f} of {col.sum()}")
