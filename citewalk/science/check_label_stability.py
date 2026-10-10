"""Measure whether map labels switch on/off or jump as the year changes (isolated browser).

CHROMIUM_BIN=/path/to/chromium uv run --with playwright==1.55.0 \
  python citewalk/science/check_label_stability.py [--measure-only]

Records every label drawn by drawLabels() (text, position, opacity) for each year 1900-2026,
in territory and dots mode, at overview zoom (field names) and zoom 2.5 (subfield names).
"""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

RECORD = """async ([show, zoom]) => {
  S.show = show; S.view = {cx: .5, cy: .5, zoom}; S.focus = -1; S.selected = -1; S.search = null;
  const rec = []; let on = false;
  const orig = octx.fillText.bind(octx);
  octx.fillText = (t, x, y) => { if (on) rec.push({t, x: Math.round(x), y: Math.round(y), a: +octx.globalAlpha.toFixed(3)}); return orig(t, x, y); };
  const inner = drawLabels;
  drawLabels = () => { on = true; try { inner(); } finally { on = false; } };
  const out = {};
  for (let y = 1900; y <= M.years[M.years.length - 1]; y++) {
    S.t = S.target = S.year = y; dirty = true;
    // let the territory worker finish this year's keyframes, then draw exactly once
    for (let k = 0; k < 40 && show === 'terr' && !(keyframes.has(terrJob(y).key) && keyframes.has(terrJob(y - 1).key)); k++)
      { drawOverlay(); await new Promise(r => setTimeout(r, 25)); }
    rec.length = 0; drawOverlay();
    out[y] = rec.filter(r => r.a > 0.02).map(r => r);
  }
  drawLabels = inner; octx.fillText = orig;
  return out;
}"""


def summarize(per_year):
    years = sorted(per_year, key=int)
    flips = moves = 0
    names = set()
    for a, b in zip(years, years[1:]):
        A = {r['t']: r for r in per_year[a]}; B = {r['t']: r for r in per_year[b]}
        flips += len(A.keys() ^ B.keys())
        moves += sum(1 for t in A.keys() & B.keys() if abs(A[t]['x'] - B[t]['x']) + abs(A[t]['y'] - B[t]['y']) > 2)
        names |= A.keys() | B.keys()
    counts = [len(per_year[y]) for y in years]
    return {'label_appear_or_disappear_events': flips, 'label_position_jumps': moves,
            'distinct_labels': len(names), 'labels_per_year_min': min(counts), 'labels_per_year_max': max(counts),
            'min_alpha': min((r['a'] for y in years for r in per_year[y]), default=None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8806/')
    p.add_argument('--chromium', default=os.environ.get('CHROMIUM_BIN'))
    p.add_argument('--measure-only', action='store_true')
    args = p.parse_args()
    out = Path('/tmp/science-label-stability'); out.mkdir(exist_ok=True)
    report, errors = {}, []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=args.chromium, headless=True,
                               args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        page = b.new_page(viewport={'width': 1440, 'height': 900})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.base + '#y=2026&w=5', wait_until='networkidle')
        page.wait_for_selector('#loading', state='detached', timeout=120000)
        for show in ('terr', 'dots'):
            for zoom in (1, 2.5):
                per_year = page.evaluate(RECORD, [show, zoom])
                report[f'{show}@zoom{zoom}'] = summarize(per_year)
        page.evaluate("() => { S.show='terr'; S.view={cx:.5,cy:.5,zoom:1}; S.t=S.target=S.year=1960; dirty=true; }")
        page.wait_for_timeout(1500); page.screenshot(path=str(out / 'terr-1960.png'))
        page.evaluate("() => { S.t=S.target=S.year=2024; dirty=true; }")
        page.wait_for_timeout(1500); page.screenshot(path=str(out / 'terr-2024.png'))
        b.close()
    report['errors'] = errors
    (out / ('before.json' if args.measure_only else 'after.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not args.measure_only:
        assert not errors, errors
        for k, v in report.items():
            if k == 'errors':
                continue
            assert v['label_appear_or_disappear_events'] == 0, (k, v)
            assert v['label_position_jumps'] == 0, (k, v)


if __name__ == '__main__':
    main()
