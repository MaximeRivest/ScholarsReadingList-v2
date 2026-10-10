// Citation walk drawn on the atlas. The walk is computed by citewalk/server.py over the full citation
// snapshot (all years) and returned with each paper's atlas position (exact for atlas dots, the
// atlas's frozen map function otherwise; papers with no vector are listed, never placed).
// Nothing here moves the map: markers are drawn at fixed world positions on the overlay canvas.
'use strict';
const WALK = { data: null, path: [], sel: null, hover: null, busy: false, all: false, error: null };
const WALK_COL = { x: '#ffffff', ref: '#b8c2d1', similar: '#47c6ff', candidate: '#ffb347', answer: '#ff3b3b' };

const walkRole = (n) => n.column === 'x' ? 'x' : n.column.startsWith('refs') ? 'ref'
  : n.column.startsWith('similar') ? 'similar' : n.answer ? 'answer' : 'candidate';
const walkPlaced = (n) => n && n.ax != null && n.ay != null;

function isWalkQuery(t) {
  t = String(t || '').trim();
  return /^(https?:\/\/(api\.)?openalex\.org\/(works\/)?)?W\d+$/i.test(t)
      || /^(doi:|https?:\/\/(dx\.)?doi\.org\/)?10\.\d{4,9}\/\S+$/i.test(t);
}
function walkHashPart() { return WALK.data ? '&wk=' + WALK.data.focus : ''; }

async function startWalk(id, opts = {}) {
  id = String(id || '').trim(); if (!id) return;
  if (id.startsWith('doi:')) id = id.slice(4);
  WALK.busy = true; WALK.error = null; renderWalk();
  let d;
  try {
    const r = await fetch('/api/walk?id=' + encodeURIComponent(id) + '&k=30');
    d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || r.statusText);
  } catch (e) { WALK.busy = false; WALK.error = e.message; renderWalk(); return; }
  WALK.busy = false; WALK.data = d; WALK.sel = null; WALK.hover = null;
  if (opts.reset) WALK.path = [];
  const f = d.nodes[d.focus];
  if (!WALK.path.length || WALK.path[WALK.path.length - 1].id !== f.id) WALK.path.push({ id: f.id, title: f.title });
  document.body.classList.add('walking');
  if (S.selected >= 0 || S.search) closeSide();
  renderWalk(); changed();
  if (opts.fit !== false) fitWalk();
}
function endWalk() {
  WALK.data = null; WALK.path = []; WALK.sel = WALK.hover = null; WALK.error = null;
  document.body.classList.remove('walking'); renderWalk(); changed();
}
function fitWalk(all = false) {
  // Default: the walk's core (most of its papers sit close together), always keeping X and the answers;
  // a few far-away references would otherwise force a view of the whole atlas. all = every placed paper.
  const d = WALK.data; if (!d) return;
  const ps = Object.values(d.nodes).filter(walkPlaced);
  if (!ps.length) return;
  let keep = ps;
  if (!all && ps.length > 4) {
    const med = (a) => { const t = [...a].sort((p, q) => p - q); return t[t.length >> 1]; };
    const cx = med(ps.map((n) => n.ax)), cy = med(ps.map((n) => n.ay));
    const dist = (n) => Math.hypot(n.ax - cx, n.ay - cy);
    const r = [...ps.map(dist)].sort((p, q) => p - q)[Math.floor(.8 * (ps.length - 1))];
    keep = ps.filter((n) => dist(n) <= r || n.id === d.focus || d.answers.includes(n.id));
  }
  const xs = keep.map((n) => n.ax), ys = keep.map((n) => n.ay);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const m = Math.min(innerWidth, innerHeight) * .92, wide = innerWidth > 900;
  const zoom = Math.max(.6, Math.min(80, Math.min((wide ? .5 : .85) * innerWidth / (Math.max(x1 - x0, .03) * m),
                                                  .55 * innerHeight / (Math.max(y1 - y0, .03) * m))));
  const shift = wide ? 210 / (zoom * m) : 0;           // keep clear of the walk panel on the right
  flyTo((x0 + x1) / 2 + shift, (y0 + y1) / 2 + 40 / (zoom * m), zoom);
}
function flyToWalkNode(id) {
  const n = WALK.data && WALK.data.nodes[id];
  if (walkPlaced(n)) flyTo(n.ax, n.ay, Math.max(S.view.zoom, 8));
}

// ---------- drawing (called at the end of drawOverlay) ----------
function walkScreen() {
  const P = {};
  for (const [id, n] of Object.entries(WALK.data.nodes)) if (walkPlaced(n)) P[id] = toScreen(n.ax, n.ay);
  return P;
}
function drawWalk() {
  const d = WALK.data; if (!d) return;
  const P = walkScreen(), active = WALK.hover || WALK.sel, near = new Set(active ? [active] : []);
  const answers = new Set(d.answers);
  if (active) for (const l of d.links) { if (l.s === active) near.add(l.t); if (l.t === active) near.add(l.s); }
  octx.save(); octx.lineCap = 'round';
  let edges = 0;
  for (const l of d.links) {
    const a = P[l.s], b = P[l.t]; if (!a || !b) continue;
    const touches = active && (l.s === active || l.t === active);
    const toAnswer = l.kind === 'y-cand' && answers.has(l.t);
    let colour, alpha, width;
    if (touches) { colour = l.s === active ? '#ffd166' : '#8cc8ff'; alpha = .95; width = 1.8; }
    else if (active) { if (!WALK.all) continue; colour = '#ffffff'; alpha = .08; width = 1; }
    else if (toAnswer) { colour = WALK_COL.answer; alpha = .8; width = 1.7; }
    else if (l.kind === 'x-ref') { colour = '#ffffff'; alpha = .3; width = 1; }
    else if (WALK.all) { colour = '#ffffff'; alpha = .16; width = 1; }
    else continue;
    walkArrow(a, b, colour, alpha, width); edges++;
  }
  // markers: references, candidates, similar, answers, then X on top
  const order = { ref: 0, candidate: 1, similar: 2, answer: 3, x: 4 };
  const ids = Object.keys(P).sort((p, q) => order[walkRole(d.nodes[p])] - order[walkRole(d.nodes[q])]);
  for (const id of ids) {
    const n = d.nodes[id], role = walkRole(n), [x, y] = P[id];
    if (x < -20 || y < -20 || x > innerWidth + 20 || y > innerHeight + 20) continue;
    let alpha = active && !near.has(id) ? .3 : 1;
    if (n.year != null && n.year > S.year) alpha *= .4;          // published after the slider year
    const out = n.column.endsWith('_out'), r = role === 'x' ? 7 : role === 'answer' ? 8 : role === 'ref' ? 3.5 : 4.5;
    octx.globalAlpha = alpha;
    octx.beginPath();
    if (role === 'answer') { octx.moveTo(x, y - r); octx.lineTo(x + r, y); octx.lineTo(x, y + r); octx.lineTo(x - r, y); octx.closePath(); }
    else octx.arc(x, y, r, 0, 7);
    octx.lineWidth = 3; octx.strokeStyle = 'rgba(0,0,0,.85)'; octx.stroke();
    if (out) { octx.lineWidth = 1.6; octx.strokeStyle = WALK_COL[role]; octx.stroke(); }
    else { octx.fillStyle = WALK_COL[role]; octx.fill(); }
    if (role === 'x') { octx.beginPath(); octx.arc(x, y, 12, 0, 7); octx.lineWidth = 2; octx.strokeStyle = '#fff'; octx.stroke(); }
    if (id === WALK.sel) { octx.beginPath(); octx.arc(x, y, 14, 0, 7); octx.lineWidth = 2.5; octx.strokeStyle = '#ffd166'; octx.stroke(); }
  }
  octx.globalAlpha = 1;
  // labels: X, the answers, and the paper under the pointer / selected
  const boxes = [];
  octx.font = '650 13px system-ui'; octx.textBaseline = 'middle'; octx.textAlign = 'left'; octx.lineJoin = 'round';
  for (const id of [d.focus, ...d.answers, ...(active && active !== d.focus && !answers.has(active) ? [active] : [])]) {
    const p = P[id]; if (!p) continue;
    const n = d.nodes[id], t = (id === d.focus ? 'X · ' : '') + (n.title || id);
    const text = t.length > 60 ? t.slice(0, 59) + '…' : t;
    const w = octx.measureText(text).width;
    let lx = Math.min(innerWidth - w - 8, p[0] + 14), ly = p[1] - 14;
    while (boxes.some((b) => lx < b[2] && lx + w > b[0] && ly - 9 < b[3] && ly + 9 > b[1])) ly -= 18;
    boxes.push([lx, ly - 9, lx + w, ly + 9]);
    octx.lineWidth = 4; octx.strokeStyle = 'rgba(0,0,0,.9)'; octx.strokeText(text, lx, ly);
    octx.fillStyle = answers.has(id) ? '#ff7a7a' : '#fff'; octx.fillText(text, lx, ly);
  }
  octx.restore();
  overlay.dataset.walkMarkers = String(ids.length); overlay.dataset.walkEdges = String(edges);
}
function walkArrow(a, b, colour, alpha, width) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy); if (len < 14) return;
  const ux = dx / len, uy = dy / len, ex = b[0] - 7 * ux, ey = b[1] - 7 * uy;
  octx.globalAlpha = alpha; octx.strokeStyle = colour; octx.fillStyle = colour; octx.lineWidth = width;
  octx.beginPath(); octx.moveTo(a[0] + 6 * ux, a[1] + 6 * uy); octx.lineTo(ex, ey); octx.stroke();
  octx.beginPath(); octx.moveTo(ex, ey); octx.lineTo(ex - 6 * ux + 3 * uy, ey - 6 * uy - 3 * ux); octx.lineTo(ex - 6 * ux - 3 * uy, ey - 6 * uy + 3 * ux);
  octx.closePath(); octx.fill(); octx.globalAlpha = 1;
}

// ---------- pointer (called from the atlas's own handlers, before its dot picking) ----------
function walkHit(sx, sy, radius) {
  if (!WALK.data) return null;
  let best = null, bd = radius * radius;
  for (const [id, n] of Object.entries(WALK.data.nodes)) {
    if (!walkPlaced(n)) continue;
    const [x, y] = toScreen(n.ax, n.ay), d2 = (x - sx) ** 2 + (y - sy) ** 2;
    if (d2 <= bd) { bd = d2; best = id; }
  }
  return best;
}
function walkHover(e) {
  const id = walkHit(e.clientX, e.clientY, 10);
  if (id !== WALK.hover) { WALK.hover = id; changed(); }
  if (!id) return false;
  const n = WALK.data.nodes[id], tip = $('tip');
  tip.innerHTML = `<b>${esc(n.title || id)}</b><br><span class="small">${n.year ?? '?'} · ${esc(walkSentence(n))}<br>click for details</span>`;
  tip.style.display = 'block'; canvas.style.cursor = 'pointer';
  tip.style.left = Math.min(innerWidth - 350, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  return true;
}
function walkClick(e) {
  const id = walkHit(e.clientX, e.clientY, e.pointerType === 'touch' ? 16 : 10);
  if (!id) return false;
  WALK.sel = id; renderWalk(); changed();
  return true;
}

// ---------- panel ----------
function walkSentence(n) {
  const k = WALK.data.stats.k;
  if (n.column === 'x') return 'Paper X, where the walk stands.';
  if (n.column.startsWith('refs')) return `X cites it; ${n.ref_support ?? 0} of the ${k} papers most like X cite it too.`;
  if (n.column.startsWith('similar')) return `One of the ${k} papers most like X: it shares ${n.shared ?? 0} of X's references.`
    + (n.column === 'similar_out' ? ' Left out as an outlier.' : '');
  return `${n.support ?? 0} of the ${k} papers most like X cite it; X does not.` + (n.answer ? ' No paper is cited by more of them.' : '')
    + (n.column === 'candidates_out' ? ' Left out as an outlier.' : '');
}
function walkItem(n, extra) {
  return `<li data-walk="${n.id}" class="${walkPlaced(n) ? '' : 'off'}">${esc(n.title || n.id)} <span class="y">${n.year ?? '?'}${extra ? ' · ' + extra : ''}${walkPlaced(n) ? '' : ' · not on the map'}</span></li>`;
}
function renderWalk() {
  const el = $('walk');
  if (!WALK.data && !WALK.busy && !WALK.error) { el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.style.bottom = innerWidth <= 760 ? ($('bottom').offsetHeight + 22) + 'px' : '';   // phones: above the controls
  let h = '<button class="close" id="walkClose" title="End the walk">×</button><h2>Citation walk</h2>';
  if (WALK.busy) h += '<div class="small">Walking…</div>';
  if (WALK.error) h += `<div class="small" style="color:#ff8080">${esc(WALK.error)}</div>`;
  const d = WALK.data;
  if (d) {
    const N = d.nodes, x = N[d.focus], s = d.stats, o = d.atlas_overlay;
    if (WALK.path.length > 1) h += '<div class="trail">' + WALK.path.map((p, i) =>
      `<a data-step="${i}" title="${esc(p.title)}">${esc((p.title || p.id).slice(0, 28))}${(p.title || '').length > 28 ? '…' : ''}</a>`).join(' → ') + '</div>';
    h += `<div class="xrow"><span class="dot" style="background:#fff"></span><b data-fly="${x.id}">${esc(x.title)}</b> <span class="y">${x.year ?? ''}</span></div>`;
    h += `<div class="ans"><div class="small">The ${s.k} papers that share the most references with X most often also cite:</div>`
      + d.answers.map((id) => `<div class="red" data-walk="${id}">◆ ${esc(N[id].title)} <span class="y">${N[id].year ?? ''} · ${N[id].support} of ${s.k}${walkPlaced(N[id]) ? '' : ' · not on the map'}</span></div>`).join('')
      + '<div class="small">X does not cite it.</div></div>';
    if (WALK.sel && N[WALK.sel]) {
      const n = N[WALK.sel];
      h += `<div class="selbox"><b>${esc(n.title)}</b><div class="small">${n.year ?? '?'}${n.topic ? ' · ' + esc(n.topic) : ''} · cited by ${n.cited_by >= 0 ? n.cited_by.toLocaleString() : '?'}</div>
        <div>${esc(walkSentence(n))}</div>
        <div class="small">${walkPlaced(n) ? ({ dot: 'Shown at its atlas dot.', reference: 'Shown at its atlas reference position.', placed: 'Placed by the atlas’s own rule from its text.' })[n.place] : 'Not on the map: no meaning vector for it.'}</div>
        <div class="btns">${n.id !== d.focus ? `<button id="walkHere">Walk here →</button>` : ''}${walkPlaced(n) ? `<button data-fly="${n.id}">Fly there</button>` : ''}
        <a href="https://openalex.org/${n.id}" target="_blank" rel="noopener">OpenAlex ↗</a></div></div>`;
    }
    const cands = d.columns.candidates.filter((id) => !d.answers.includes(id)).map((id) => N[id])
      .sort((p, q) => q.support - p.support || q.cited_by - p.cited_by).slice(0, 6);
    if (cands.length) h += `<details open><summary>Runners-up</summary><ol>${cands.map((n) => walkItem(n, `${n.support} of ${s.k}`)).join('')}</ol></details>`;
    h += `<details><summary>The ${s.k} papers most like X</summary><ol>${d.similar.map((id) => walkItem(N[id], `shares ${N[id].shared}`)).join('')}</ol></details>`;
    h += `<details><summary>Papers X cites (${d.columns.refs.length + d.columns.refs_out.length} shown of ${s.ref_publications})</summary><ol>${d.columns.refs.concat(d.columns.refs_out).map((id) => walkItem(N[id], `${N[id].ref_support ?? 0} of ${s.k}`)).join('')}</ol></details>`;
    h += `<div class="legend"><span><i style="background:#fff"></i>X</span><span><i style="background:${WALK_COL.ref}"></i>X cites</span>
      <span><i style="background:${WALK_COL.similar}"></i>${s.k} most like X</span><span><i style="background:${WALK_COL.candidate}"></i>they cite, X doesn't</span>
      <span><i class="dia"></i>most cited of those</span></div>`;
    h += `<label class="small"><input type="checkbox" id="walkAll" ${WALK.all ? 'checked' : ''}> show every citation line</label> <button id="walkFit">Zoom to the walk</button> <button id="walkFitAll">Show all of it</button>`;
    const un = o.unplaced.map((id) => N[id]);
    h += `<div class="small note">On the map: ${o.placed} of ${o.total} papers (${o.by.dot} are atlas dots, ${o.by.placed + o.by.reference} placed by the atlas’s rule from their text).
      ${un.length ? `<details><summary>${un.length} without a position (no meaning vector)</summary><ol>${un.map((n) => walkItem(n, '')).join('')}</ol></details>` : ''}
      Computed on the full citation snapshot (all years); the year slider only dims walk papers published after it.
      Outliers (over ${s.max_refs} references or ${s.max_cites} citations) are hollow and left out of the calculation.</div>`;
  }
  el.innerHTML = h;
  $('walkClose').onclick = endWalk;
  if (!d) return;
  el.querySelectorAll('[data-walk]').forEach((e2) => e2.onclick = () => { WALK.sel = e2.dataset.walk; renderWalk(); flyToWalkNode(WALK.sel); changed(); });
  el.querySelectorAll('[data-fly]').forEach((e2) => e2.onclick = () => flyToWalkNode(e2.dataset.fly));
  el.querySelectorAll('[data-step]').forEach((a) => a.onclick = () => { WALK.path = WALK.path.slice(0, +a.dataset.step + 1); startWalk(WALK.path[WALK.path.length - 1].id); });
  if ($('walkHere')) $('walkHere').onclick = () => startWalk(WALK.sel);
  $('walkAll').onchange = () => { WALK.all = $('walkAll').checked; changed(); };
  $('walkFit').onclick = () => fitWalk(false);
  $('walkFitAll').onclick = () => fitWalk(true);
}

(function initWalk() {
  const css = document.createElement('style');
  css.textContent = `
  #walk { top:12px; right:12px; width:min(400px, calc(100vw - 24px)); max-height:calc(100vh - 190px); overflow:auto;
    padding:12px 14px; display:none; font-size:13px; }
  #walk h2 { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--dim); margin:0 0 6px; }
  #walk .close { float:right; margin:-4px -6px 0 8px; padding:2px 8px; }
  #walk .trail { font-size:12px; color:var(--dim); margin-bottom:6px; }
  #walk .trail a { color:#8cc8ff; cursor:pointer; }
  #walk .xrow b { cursor:pointer; } #walk .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
  #walk .y { color:var(--dim); font-size:12px; font-weight:400; }
  #walk .ans { margin:8px 0; padding:8px 10px; border:1px solid #ff3b3b66; border-radius:8px; background:#ff3b3b12; }
  #walk .red { color:#ff8a8a; font-weight:650; cursor:pointer; margin:4px 0; }
  #walk .selbox { margin:8px 0; padding:8px 10px; border:1px solid #ffd16666; border-radius:8px; }
  #walk .btns { display:flex; gap:8px; align-items:center; margin-top:6px; }
  #walk ol { padding-left:20px; margin:4px 0; } #walk li { margin:3px 0; cursor:pointer; } #walk li:hover { color:var(--accent); }
  #walk li.off { opacity:.6; } #walk summary { cursor:pointer; margin-top:6px; color:var(--dim); }
  #walk .legend { display:flex; flex-wrap:wrap; gap:4px 12px; margin:8px 0 6px; font-size:12px; color:var(--dim); }
  #walk .legend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; border:1px solid #000; }
  #walk .legend i.dia { border-radius:0; transform:rotate(45deg); background:#ff3b3b; width:8px; height:8px; }
  #walk .note { margin-top:8px; line-height:1.4; }
  body.walking #race { display:none !important; }
  .walkbtn { margin-top:8px; }
  @media (max-width: 760px) { #walk { top:auto; max-height:38vh; } }`;
  document.head.appendChild(css);
  const el = document.createElement('div'); el.id = 'walk'; el.className = 'panel'; document.body.appendChild(el);
  // the atlas's paper panel takes the same corner: hide the walk panel while it is open
  const sideEl = document.getElementById('side');
  new MutationObserver(() => { el.style.visibility = sideEl.style.display === 'block' ? 'hidden' : 'visible'; })
    .observe(sideEl, { attributes: true, attributeFilter: ['style'] });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && WALK.sel) { WALK.sel = null; renderWalk(); changed(); } });
  const wk = new URLSearchParams(location.hash.slice(1)).get('wk');
  if (wk) {                                   // restore a walk from the address once the atlas has loaded
    const wait = () => document.getElementById('loading') ? setTimeout(wait, 200) : startWalk(wk, { reset: true, fit: false });
    wait();
  }
})();
