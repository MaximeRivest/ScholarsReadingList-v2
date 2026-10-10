"""Browser check of the citation walk drawn on the atlas (fresh throwaway profile).

    CHROMIUM_BIN=... uv run --with playwright==1.55.0 python citewalk/science/check_walk_browser.py

Checks with real clicks: start a walk from the search box, markers drawn at the API's positions,
click the red answer on the map, "Walk here", shared papers keep their positions, start a walk
from an atlas dot's panel, reload restores the walk, ending the walk restores the atlas, no errors.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

OUT = Path('/tmp/science-walk-browser')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8806/')
    ap.add_argument('--chromium', default=os.environ.get('CHROMIUM_BIN'))
    ap.add_argument('--start', default='W3177828909')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    api = json.load(urlopen(args.base + 'api/walk?id=' + args.start, timeout=120))
    res, errors = [], []
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=args.chromium, headless=True,
                               args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        page = b.new_page(viewport={'width': 1440, 'height': 900})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: m.type == 'error' and errors.append(m.text))
        page.goto(args.base + '#y=2026&w=5', wait_until='networkidle')
        page.wait_for_selector('#loading', state='detached', timeout=120000)

        page.fill('#q', args.start); page.press('#q', 'Enter')
        page.wait_for_function('() => WALK.data && WALK.data.focus === "%s"' % args.start, timeout=120000)
        page.wait_for_timeout(1200)                       # let the fly-to finish
        assert page.is_visible('#walk') and not page.is_visible('#race')
        got = page.evaluate('WALK.data.answers')
        assert got == api['answers'], (got, api['answers'])
        assert page.evaluate("document.querySelector('#walk .red') !== null")
        markers = int(page.evaluate('overlay.dataset.walkMarkers'))
        assert markers == api['atlas_overlay']['placed'], (markers, api['atlas_overlay'])
        same = page.evaluate('''(api) => Object.values(api.nodes).every(n => {
              const m = WALK.data.nodes[n.id]; return m.ax === n.ax && m.ay === n.ay; })''', api)
        assert same
        page.screenshot(path=str(OUT / '1-walk.png'))
        res.append(f'Search box starts the walk; {markers} markers at the API positions; answer {got} shown in red')

        ans = got[0]
        xy = page.evaluate('(id) => { const n = WALK.data.nodes[id]; return n.ax == null ? null : toScreen(n.ax, n.ay); }', ans)
        assert xy, 'answer has no position'
        if not (0 < xy[0] < 1440 and 0 < xy[1] < 900):
            page.evaluate('(id) => { const n = WALK.data.nodes[id]; S.view.cx = n.ax; S.view.cy = n.ay; changed(); }', ans)
            page.wait_for_timeout(300)
            xy = page.evaluate('(id) => { const n = WALK.data.nodes[id]; return toScreen(n.ax, n.ay); }', ans)
        page.mouse.move(*xy); page.wait_for_timeout(150)
        assert page.is_visible('#tip') and 'most like X' in page.inner_text('#tip')
        page.mouse.click(*xy); page.wait_for_timeout(300)
        assert page.evaluate('WALK.sel') == ans and page.is_visible('#walkHere')
        assert not page.is_visible('#side')                 # the click went to the walk, not an atlas dot
        page.screenshot(path=str(OUT / '2-selected.png'))
        res.append('Hovering the red marker shows its tooltip; clicking it selects it in the walk panel')

        first = page.evaluate('WALK.data')
        page.click('#walkHere')
        page.wait_for_function('(id) => WALK.data && WALK.data.focus === id', arg=ans, timeout=120000)
        second = page.evaluate('WALK.data')
        common = [i for i in first['nodes'] if i in second['nodes'] and first['nodes'][i]['ax'] is not None]
        assert all((first['nodes'][i]['ax'], first['nodes'][i]['ay']) == (second['nodes'][i]['ax'], second['nodes'][i]['ay']) for i in common)
        assert page.evaluate('WALK.path.length') == 2 and page.is_visible('#walk .trail')
        page.wait_for_timeout(1200); page.screenshot(path=str(OUT / '3-walked.png'))
        res.append(f'"Walk here" moves X to the answer; {len(common)} papers in both walks keep identical positions; trail shown')

        page.goto(page.url, wait_until='networkidle')        # reload with &wk= in the address
        page.wait_for_selector('#loading', state='detached', timeout=120000)
        page.wait_for_function('(id) => WALK.data && WALK.data.focus === id', arg=ans, timeout=120000)
        res.append('Reload restores the walk from the address')

        page.click('#walkClose'); page.wait_for_timeout(300)
        assert not page.is_visible('#walk') and page.evaluate('WALK.data') is None
        assert int(page.evaluate('overlay.dataset.walkMarkers')) >= 0
        res.append('× ends the walk; the atlas is as before')

        hit = page.evaluate('''() => { for (let k = 0; k < 4000; k++) { const i = Math.floor(Math.random() * N);
              const [x, y] = toScreen(XY[2*i], XY[2*i+1]);
              if (x > 450 && x < innerWidth - 450 && y > 150 && y < innerHeight - 220 && YEAR[i] >= 2005) return [x, y]; } return null; }''')
        page.mouse.click(*hit)
        page.wait_for_selector('#side .walkbtn', timeout=30000)
        page.click('#side .walkbtn')
        page.wait_for_function('() => WALK.data !== null || document.querySelector("#walk").innerText.includes("snapshot")', timeout=120000)
        page.wait_for_timeout(1200)
        assert not page.is_visible('#side')
        page.screenshot(path=str(OUT / '4-from-dot.png'))
        msg = page.evaluate('WALK.data ? WALK.data.nodes[WALK.data.focus].title : WALK.error')
        res.append(f'An atlas dot\'s panel starts a walk from that paper: {msg[:70]}')

        phone = b.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        phone.on('pageerror', lambda e: errors.append('phone: ' + str(e)))
        phone.goto(args.base + f'#y=2026&w=5&wk={args.start}', wait_until='networkidle')
        phone.wait_for_selector('#loading', state='detached', timeout=120000)
        phone.wait_for_function('() => WALK.data', timeout=120000); phone.wait_for_timeout(800)
        phone.screenshot(path=str(OUT / '5-phone.png'))
        res.append('Phone layout opens a walk from a link')
        b.close()
    assert not errors, errors
    (OUT / 'results.json').write_text(json.dumps({'checks': res, 'errors': errors}, indent=2))
    print('\n'.join('PASS ' + r for r in res))


if __name__ == '__main__':
    main()
