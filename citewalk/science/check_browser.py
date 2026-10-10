"""Browser check of the science-atlas viewer (fresh throwaway profile, never the user's).

    CHROMIUM_BIN=$(which chromium) uv run --with playwright==1.55.0 --with pillow python check_browser.py --base http://127.0.0.1:8806/

Checks: page loads with no console errors, WebGL draws non-black pixels, the year slider changes
the dot count, the map does not move when the year changes, a click selects a paper and shows
its neighbour score, search places a marker, reload restores the state from the URL, phone layout.
Screenshots go to /tmp/science-atlas-browser/.
"""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path('/tmp/science-atlas-browser')


def lit_fraction(path):
    from PIL import Image
    im = Image.open(path).convert('L')
    w, h = im.size
    px = im.crop((int(w * .3), int(h * .15), int(w * .7), int(h * .7))).getdata()   # away from panels
    return round(sum(v > 12 for v in px) / len(px), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8806/')
    ap.add_argument('--chromium', default=os.environ.get('CHROMIUM_BIN'))
    ap.add_argument('--query', default='CRISPR gene editing')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    results, errors = {}, []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium, headless=True,
                                     args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                           '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.on('console', lambda m: m.type == 'error' and errors.append(m.text))
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.base + '#y=2026&w=0', wait_until='networkidle')
        page.wait_for_selector('#loading', state='detached', timeout=120_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / 'all-2026.png'))
        results['lit_all'] = lit_fraction(OUT / 'all-2026.png')
        count_all = page.inner_text('#count')
        view_before = page.evaluate('() => location.hash.match(/v=([^&]+)/)[1]')
        page.select_option('#mode', '5')
        page.keyboard.press('ArrowLeft')
        for _ in range(60):
            page.keyboard.press('ArrowLeft')
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / 'window-1965.png'))
        results['lit_1965'] = lit_fraction(OUT / 'window-1965.png')
        results['count_all'] = count_all
        results['count_1965'] = page.inner_text('#count')
        results['view_unchanged_by_year'] = view_before == page.evaluate('() => location.hash.match(/v=([^&]+)/)[1]')
        # click the brightest-ish area: try a few spots until a paper is selected
        page.select_option('#mode', '0')
        for _ in range(70):
            page.keyboard.press('ArrowRight')
        page.wait_for_timeout(500)
        hit = page.evaluate("""() => { for (let k = 0; k < 4000; k++) {
              const i = Math.floor(Math.random() * N);
              const x = (XY[2*i] - S.view.cx) * scalePx() + innerWidth / 2, y = (XY[2*i+1] - S.view.cy) * scalePx() + innerHeight / 2;
              if (x > 450 && x < innerWidth - 420 && y > 150 && y < innerHeight - 200) return [x, y]; } return null; }""")
        page.mouse.click(*hit)
        page.wait_for_selector('#side .score', timeout=20_000)
        results['selected_score'] = page.inner_text('#side .score').split('\n')[0]
        page.screenshot(path=str(OUT / 'selected.png'))
        page.mouse.move(700, 450); page.mouse.wheel(0, -800); page.wait_for_timeout(400)
        page.mouse.move(700, 450); page.mouse.down(); page.mouse.move(760, 480, steps=5); page.mouse.up()
        page.wait_for_timeout(500)
        zoomed_hash = page.evaluate('() => location.hash')
        page.fill('#q', args.query); page.press('#q', 'Enter')
        page.wait_for_selector('#side ol li', timeout=60_000)
        page.wait_for_timeout(1200)
        results['search_first'] = page.inner_text('#side ol li >> nth=0')
        page.screenshot(path=str(OUT / 'search.png'))
        page.goto(args.base + zoomed_hash, wait_until='networkidle')
        page.wait_for_selector('#loading', state='detached', timeout=120_000)
        page.wait_for_timeout(1000)
        results['reload_keeps_view'] = page.evaluate('() => location.hash').split('&p=')[0].split('&v=')[1] == \
            zoomed_hash.split('&p=')[0].split('&v=')[1]
        phone = browser.new_page(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        phone.on('pageerror', lambda e: errors.append('phone: ' + str(e)))
        phone.goto(args.base, wait_until='networkidle')
        phone.wait_for_selector('#loading', state='detached', timeout=120_000)
        phone.wait_for_timeout(1000)
        phone.screenshot(path=str(OUT / 'phone.png'))
        browser.close()
    results['errors'] = errors
    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
