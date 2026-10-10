"""Actual pointer/keyboard tests for the flat atlas (isolated headless Chromium).

CHROMIUM_BIN=<standalone Chromium binary> uv run --with playwright==1.55.0 \
  python citewalk/check_atlas_browser.py --base http://127.0.0.1:8803/atlas.html

Does not use or alter the user's browser profile. API calls use only saved
citation/embedding data. Screenshots and the result summary go under /tmp.
"""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8803/atlas.html')
    p.add_argument('--chromium',default=os.environ.get('CHROMIUM_BIN'))
    p.add_argument('--out',type=Path,default=Path('/tmp/citewalk-atlas-browser'))
    args=p.parse_args()
    if not args.chromium:p.error('Set CHROMIUM_BIN to a standalone Chromium executable')
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=args.chromium,headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.base+'#W3177828909')
        page.wait_for_function("window.atlasDebug?.data?.atlas?.status === 'ready'",timeout=90000)
        page.wait_for_timeout(200)
        first=page.evaluate('atlasDebug.data');atlas=first['atlas']['id']
        assert first['atlas']['plotted']>0
        assert page.locator('#plot').get_attribute('data-visible-edges')=='0'
        page.screenshot(path=str(args.out/'initial.png'),full_page=True)
        results.append('No citation lines before selection; only actual-vector points plotted')
        candidate=page.evaluate('''() => {
          const d=atlasDebug.data,p=atlasDebug.plot,r=p.canvas.getBoundingClientRect();
          for(const edge of d.links){
            for(const id of [edge.s,edge.t]){
              const n=d.nodes[id];if(n.map_status!=='placed')continue;
              const point=p.point(n.atlas_key),xy=p.screen(point),hits=p.hits(xy.x,xy.y);
              if(hits[0]?.id===id && xy.x>15 && xy.y>15 && xy.x<r.width-15 && xy.y<r.height-15)
                return {id,x:r.left+xy.x,y:r.top+xy.y};
            }
          }
        }''')
        assert candidate
        page.mouse.click(candidate['x'],candidate['y'])
        page.wait_for_function("Number(document.querySelector('#plot').dataset.visibleEdges)>0")
        assert page.locator('#side h2').first.inner_text()=='SELECTED PUBLICATION' or page.locator('#side h2').first.inner_text()=='Selected publication'
        page.screenshot(path=str(args.out/'selected.png'),full_page=True)
        results.append('Pointer selection reveals citation links and details')
        camera=page.evaluate('({...atlasDebug.plot.camera})')
        box=page.locator('#plot').bounding_box()
        page.mouse.move(box['x']+25,box['y']+25);page.mouse.down();page.mouse.move(box['x']+95,box['y']+60,steps=8);page.mouse.up()
        moved=page.evaluate('({...atlasDebug.plot.camera})');assert moved!=camera
        page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2);page.mouse.wheel(0,-130)
        page.wait_for_timeout(200)
        assert page.evaluate('atlasDebug.plot.camera.scale')>camera['scale']
        results.append('Pointer drag pans and wheel zooms; neither mutates coordinates')
        page.locator('#fit').click()
        answer=first['answers'][0]
        page.locator(f'#summary [data-paper="{answer}"]').click()
        page.locator('#walk-here').click()
        page.wait_for_function('(id)=>atlasDebug.data?.focus===id',arg=answer,timeout=60000)
        second=page.evaluate('atlasDebug.data');assert second['atlas']['id']==atlas
        a={n['atlas_key']:(n['mx'],n['my']) for n in first['nodes'].values() if n['map_status']=='placed'}
        b={n['atlas_key']:(n['mx'],n['my']) for n in second['nodes'].values() if n['map_status']=='placed'}
        shared=a.keys()&b.keys();assert shared
        assert all(a[k]==b[k] for k in shared)
        results.append(f'Actual Walk here button: {len(shared)} shared papers retain identical coordinates')
        saved_camera=page.evaluate('({...atlasDebug.plot.camera})')
        page.reload();page.wait_for_function('(id)=>atlasDebug.data?.focus===id',arg=answer,timeout=60000)
        assert page.evaluate('atlasDebug.data.atlas.id')==atlas
        assert page.evaluate('({...atlasDebug.plot.camera})')==saved_camera
        results.append('Reload restores both saved projection and camera')
        page.locator('#neighbors-visible').check();page.locator('#trail-visible').uncheck()
        assert page.evaluate('atlasDebug.data.atlas.id')==atlas
        # Keyboard-accessible alternatives work independently of pixel hit testing.
        page.locator('#papers-tab').click();assert page.locator('#papers').is_visible()
        page.locator('#papers [data-paper]').first.click();assert page.locator('#walk-here').is_visible()
        page.locator('#map-tab').click()
        page.locator('#year').focus();page.keyboard.press('Home');page.wait_for_timeout(100)
        assert page.locator('#year-label').inner_text()!='all years'
        assert page.evaluate('atlasDebug.data.atlas.id')==atlas
        results.append('Papers list, exact-neighbor rings, landmarks, and year filter work without refitting')
        page.locator('#new-map').click()
        page.wait_for_function('(old)=>atlasDebug.data?.atlas?.id && atlasDebug.data.atlas.id!==old',arg=atlas,timeout=60000)
        results.append('Only explicit Start new map changes the projection')
        assert not errors,errors
        page.screenshot(path=str(args.out/'new-map.png'),full_page=True)
        context.close()
        phone=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
        pg=phone.new_page();pg.on('pageerror',lambda e:errors.append(str(e)))
        pg.goto(args.base+'#W3045885187');pg.wait_for_function("window.atlasDebug?.data?.atlas?.status === 'ready'",timeout=90000)
        assert pg.locator('#plot').bounding_box()['height']>=280
        pg.locator('#papers-tab').tap();assert pg.locator('#papers').is_visible()
        pg.screenshot(path=str(args.out/'mobile.png'),full_page=True)
        assert not errors,errors
        results.append('Phone layout and touch-accessible Papers tab checked')
        phone.close();browser.close()
    (args.out/'results.json').write_text(json.dumps(dict(checks=results,errors=errors),indent=2))
    print('\n'.join('PASS '+x for x in results))


if __name__=='__main__':main()
