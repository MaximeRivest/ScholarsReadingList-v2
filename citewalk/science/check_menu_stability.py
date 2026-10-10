"""Check changing atlas counters/play glyphs cannot move menu controls.

Uses an isolated browser. It exercises real updateText/updateRace functions, plus
actual playback, without changing map data or the user's browser profile.

CHROMIUM_BIN=/path/to/standalone/chromium uv run --with playwright==1.55.0 \
  python citewalk/science/check_menu_stability.py [--measure-only]
"""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

GEOMETRY = """() => Object.fromEntries(
 ['bottom','controls','play','year','mode','show','lag','colour','eq','legendBtn','aboutBtn','count','timeline','race','raceSub'].map(id=>{
   const e=document.getElementById(id),r=e.getBoundingClientRect();
   return [id,{x:r.x,y:r.y,w:r.width,h:r.height,scrollHeight:e.scrollHeight,clientHeight:e.clientHeight}];
 }))"""


def variations(frames):
    out={}
    for name in frames[0]:
        delta={k:round(max(f[name][k] for f in frames)-min(f[name][k] for f in frames),3) for k in ('x','y','w','h')}
        if any(v>0.1 for v in delta.values()):out[name]=delta
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:8806/')
    p.add_argument('--chromium',default=os.environ.get('CHROMIUM_BIN'))
    p.add_argument('--measure-only',action='store_true')
    args=p.parse_args()
    if not args.chromium:p.error('Set CHROMIUM_BIN to an isolated Chromium binary')
    out=Path('/tmp/science-menu-stability');out.mkdir(exist_ok=True)
    report={};errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=args.chromium,headless=True,args=[
            '--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.base+'#y=2020&w=5',wait_until='networkidle')
        page.wait_for_selector('#loading',state='detached',timeout=120000)
        page.evaluate('document.fonts.ready')
        # Measure text/layout updates synchronously, so rendering millions of dots
        # does not dominate a menu regression check. State/canvas projection is unchanged.
        for width in (1600,1440,1280,1024,768,760,390,320):
            page.set_viewport_size({'width':width,'height':900});page.wait_for_timeout(100)
            frames=page.evaluate('''() => {
              const originalYear=S.year, originalWindow=S.window, originalText=$('play').textContent;
              const capture='''+GEOMETRY+'''; const frames=[];
              for(const window of [0,1,5,10]) {
                S.window=window;$('mode').value=String(window);
                for(const year of [1900,1910,1950,1980,1999,2000,2010,2025,2026]) {
                  S.year=year;updateText();
                  for(const text of ['▶','❚❚']) { $('play').textContent=text;frames.push(capture()); }
                }
              }
              S.year=originalYear;S.window=originalWindow;$('mode').value=String(originalWindow);
              $('play').textContent=originalText;updateText();return frames;
            }''')
            delta=variations(frames)
            counter_overflow=max(f['count']['scrollHeight']-f['count']['clientHeight'] for f in frames)
            report[str(width)]={'changes':delta,'counter_vertical_overflow':counter_overflow}
            if not args.measure_only:
                stable=('bottom','controls','play','year','mode','show','lag','colour','eq','legendBtn','aboutBtn','count','timeline')
                assert not any(k in delta for k in stable),(width,delta)
                assert counter_overflow<=1,(width,counter_overflow)
        # Real play/pause interaction, starting before the incomplete-year suffix appears.
        page.set_viewport_size({'width':1440,'height':900})
        page.evaluate('setYear(2022)');page.wait_for_timeout(100)
        baseline=page.evaluate(GEOMETRY)
        page.click('#play');frames=[baseline]
        for _ in range(14):
            page.wait_for_timeout(100);frames.append(page.evaluate(GEOMETRY))
        page.click('#play');frames.append(page.evaluate(GEOMETRY))
        report['playback']={'changes':variations(frames)}
        if not args.measure_only:
            assert not any(k in report['playback']['changes'] for k in ('play','year','mode','controls','bottom','timeline')),report['playback']
        page.screenshot(path=str(out/'desktop.png'))
        page.set_viewport_size({'width':390,'height':844});page.evaluate('setYear(2026)');page.wait_for_timeout(200)
        page.screenshot(path=str(out/'phone.png'))
        browser.close()
    report['errors']=errors
    assert not errors,errors
    (out/('before.json' if args.measure_only else 'after.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
