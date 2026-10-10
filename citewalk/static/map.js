// Map view: the land is meaning (title + abstract vectors), research areas are districts,
// papers are buildings (height ~ citations), citations are roads and rivers.
//   river  = the citations that flow into the answer (they converge like tributaries)
//   roads  = X's own references, from X's tower
//   streets = every other citation; shown when you hover a building or tick "all links"
// Papers without a meaning vector are placed beside the papers they are linked to, marked as such.
const MAP = (() => {
  const C30 = Math.cos(Math.PI / 6), S30 = Math.sin(Math.PI / 6);
    const ROLE = {                                   // [roof, left wall, right wall]
    x: ['#3a424c', '#1c2025', '#2a3037'],
    ref: ['#d3d8de', '#9aa3ad', '#b4bcc4'],
    similar: ['#7aa9e0', '#2f63a5', '#4c82c6'],
    candidate: ['#f6c4c4', '#d78c8c', '#e7a7a7'],
    answer: ['#f2554f', '#a51c1c', '#cc302b'],
  };
  const LAND = ['#dfeccf', '#f1e4c4', '#dfe0f2', '#f5d9cf', '#d3ebe6', '#ece0ef', '#e8ecc8'];
  const LAND_INK = ['#4f6b33', '#7a5d22', '#4b4f8a', '#8a4b36', '#2f6b60', '#6d4a78', '#626b23'];
  const f1 = v => v.toFixed(1);
  const P = (u, v, z = 0) => [(u - v) * C30, (u + v) * S30 - z];
  let st = null;

  const roleOf = n => n.column === 'x' ? 'x' : n.column.startsWith('refs') ? 'ref'
    : n.column.startsWith('similar') ? 'similar' : n.answer ? 'answer' : 'candidate';

  function hull(pts) {                              // monotone chain convex hull
    const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], hi = [];
    for (const q of p) { while (lo.length >= 2 && cross(lo.at(-2), lo.at(-1), q) <= 0) lo.pop(); lo.push(q); }
    for (const q of [...p].reverse()) { while (hi.length >= 2 && cross(hi.at(-2), hi.at(-1), q) <= 0) hi.pop(); hi.push(q); }
    return lo.slice(0, -1).concat(hi.slice(0, -1));
  }

  function layout(data) {
    const N = data.nodes, ids = Object.keys(N);
    const pos = new Map();
    // 1. meaning coordinates → world
    const sem = ids.filter(id => N[id].mx != null);
    if (sem.length) {
      const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
      const cu = med(sem.map(id => N[id].mx)), cv = med(sem.map(id => N[id].my));
      const rr = sem.map(id => Math.hypot(N[id].mx - cu, N[id].my - cv)).sort((x, y) => x - y);
      const q = Math.max(rr[Math.floor(rr.length * 0.75)] || 0, 1e-6), lim = 1.4 * q;
      // far-away papers are pulled in (same direction, shorter distance) so one outlier cannot squash the map
      const pts = new Map(sem.map(id => {
        let du = N[id].mx - cu, dv = N[id].my - cv; const r = Math.hypot(du, dv);
        if (r > lim) { const r2 = lim + (r - lim) * 0.2; du *= r2 / r; dv *= r2 / r; }
        return [id, [du, dv]];
      }));
      // a gentle pull towards each research area's centre, so areas form neighbourhoods
      const cen = new Map();
      sem.forEach(id => { const r = N[id].region; if (r == null) return; const c = cen.get(r) || [0, 0, 0];
                          const q2 = pts.get(id); cen.set(r, [c[0] + q2[0], c[1] + q2[1], c[2] + 1]); });
      sem.forEach(id => { const c = cen.get(N[id].region); if (!c) return; const q2 = pts.get(id);
                          q2[0] += (c[0] / c[2] - q2[0]) * 0.6; q2[1] += (c[1] / c[2] - q2[1]) * 0.6; });
      // turn the layout so its long direction runs across the screen (u = -v is screen-horizontal here),
      // and stretch the short direction to at least 55% of the long one, so the map is not a thin streak
      {
        const arr = [...pts.values()], n2 = arr.length;
        const mu = arr.reduce((s2, q2) => s2 + q2[0], 0) / n2, mv = arr.reduce((s2, q2) => s2 + q2[1], 0) / n2;
        let a2 = 0, b2 = 0, c2 = 0;
        arr.forEach(q2 => { const x = q2[0] - mu, y = q2[1] - mv; a2 += x * x; b2 += x * y; c2 += y * y; });
        const th = 0.5 * Math.atan2(2 * b2, a2 - c2);                  // principal axis angle
        const e1 = [Math.cos(th), Math.sin(th)], e2 = [-Math.sin(th), Math.cos(th)];
        const s1 = Math.sqrt(arr.reduce((s2, q2) => s2 + ((q2[0] - mu) * e1[0] + (q2[1] - mv) * e1[1]) ** 2, 0) / n2) || 1e-6;
        const s2 = Math.sqrt(arr.reduce((s3, q2) => s3 + ((q2[0] - mu) * e2[0] + (q2[1] - mv) * e2[1]) ** 2, 0) / n2) || 1e-6;
        const stretch = Math.max(1, 0.55 * s1 / s2);
        const H = [Math.SQRT1_2, -Math.SQRT1_2], V = [Math.SQRT1_2, Math.SQRT1_2];   // screen axes in (u, v)
        pts.forEach(q2 => {
          const a3 = (q2[0] - mu) * e1[0] + (q2[1] - mv) * e1[1], b3 = ((q2[0] - mu) * e2[0] + (q2[1] - mv) * e2[1]) * stretch;
          q2[0] = a3 * H[0] + b3 * V[0]; q2[1] = a3 * H[1] + b3 * V[1];
        });
      }
      // size the map to the screen: spread papers until the projected map is ~1000 px wide, ~560 px tall;
      // buildings and text keep a fixed size, so there is ground (and room for roads) between them
      const proj = [...pts.values()].map(([u, v]) => P(u, v));
      const wx = Math.max(...proj.map(q2 => q2[0])) - Math.min(...proj.map(q2 => q2[0])) || 1e-6;
      const wy = Math.max(...proj.map(q2 => q2[1])) - Math.min(...proj.map(q2 => q2[1])) || 1e-6;
      const k = Math.min(Math.min(1000, 120 * Math.sqrt(ids.length)) / wx, 560 / wy);
      pts.forEach((q2, id) => pos.set(id, {u: q2[0] * k, v: q2[1] * k, drift: false}));
    }
    // 2. papers without a vector: beside the papers they are linked to (repeat so chains settle)
    const adj = new Map();
    data.links.forEach(l => { for (const [a, b] of [[l.s, l.t], [l.t, l.s]]) { if (!adj.has(a)) adj.set(a, []); adj.get(a).push(b); } });
    const hash = s => { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) | 0; return h; };
    for (let pass = 0; pass < 6; pass++) {
      ids.forEach(id => {
        if (N[id].mx != null) return;
        const ns = (adj.get(id) || []).filter(m => pos.has(m) && m !== id);
        if (!ns.length) return;
        const a = (hash(id) % 360) * Math.PI / 180;
        const u = ns.reduce((s, m) => s + pos.get(m).u, 0) / ns.length + 24 * Math.cos(a);
        const v = ns.reduce((s, m) => s + pos.get(m).v, 0) / ns.length + 24 * Math.sin(a);
        pos.set(id, {u, v, drift: true});
      });
    }
    // 3. anything still unplaced (no vector, no placed neighbour): a separate "uncharted" plot
    const loose = ids.filter(id => !pos.has(id));
    // under the map, in a row (screen coordinates converted back to the ground plane)
    const scr = [...pos.values()].map(p => P(p.u, p.v));
    const sx0 = scr.length ? Math.min(...scr.map(q => q[0])) : 0, sy1 = scr.length ? Math.max(...scr.map(q => q[1])) + 110 : 0;
    const ground = (x, y) => ({u: (x / C30 + y / S30) / 2, v: (y / S30 - x / C30) / 2});
    loose.forEach((id, k) => pos.set(id, {...ground(sx0 + 40 + (k % 12) * 34, sy1 + Math.floor(k / 12) * 30), drift: true, loose: true}));
    // 4. no two buildings on the same plot: push apart, keeping the meaning layout as far as possible
    const size = id => id === data.focus ? 10 : 4 + 2 * Math.log10(1 + Math.max(0, N[id].cited_by));
    const list = ids.map(id => ({id, p: pos.get(id), r: size(id)}));
    for (let it = 0; it < 80; it++) {
      let moved = false;
      for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
        const a = list[i], b = list[j], dx = b.p.u - a.p.u, dy = b.p.v - a.p.v;
        const d = Math.hypot(dx, dy) || 0.01, need = (a.r + b.r) * 1.2 + 6;
        if (d < need) {
          const push = (need - d) / 2, ux = dx / d || 1, uy = dy / d;
          a.p.u -= ux * push; a.p.v -= uy * push; b.p.u += ux * push; b.p.v += uy * push; moved = true;
        }
      }
      if (!moved) break;
    }
    list.forEach(e => { e.p.r = e.r; e.p.h = e.id === data.focus ? 64 : Math.min(54, 6 + 12 * Math.log10(1 + Math.max(0, N[e.id].cited_by))); });
    return {pos, loose};
  }

  function building(id, p, n, roles) {
    const r = p.r, c = roles, h = p.h;
    const A = P(p.u - r, p.v - r), B = P(p.u + r, p.v - r), Cc = P(p.u + r, p.v + r), D = P(p.u - r, p.v + r);
    const up = q => [q[0], q[1] - h];
    const pts = arr => arr.map(q => f1(q[0]) + ',' + f1(q[1])).join(' ');
    // visible walls in this projection: the two faces towards the viewer (+u side and +v side)
    return `<polygon class="wall" points="${pts([D, Cc, up(Cc), up(D)])}" fill="${c[1]}"/>
      <polygon class="wall" points="${pts([B, Cc, up(Cc), up(B)])}" fill="${c[2]}"/>
      <polygon class="roof" points="${pts([up(A), up(B), up(Cc), up(D)])}" fill="${c[0]}"/>`;
  }

  function render(svg, data, hooks) {
    const N = data.nodes, L = layout(data), pos = L.pos;
    const bb = {x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9};
    const grow = (x, y) => { bb.x0 = Math.min(bb.x0, x); bb.x1 = Math.max(bb.x1, x); bb.y0 = Math.min(bb.y0, y); bb.y1 = Math.max(bb.y1, y); };
    const pt = (u, v, z = 0) => { const q = P(u, v, z); return f1(q[0]) + ',' + f1(q[1]); };
    // ground: land around every building; districts (research areas) = land around their own buildings.
    // On this ground plane a circle of radius R projects to an ellipse (rx = R·√2·cos30, ry = R·√2·sin30).
    const disc = (p, R, extra = '') => { const [x, y] = P(p.u, p.v); grow(x - R * 1.25, y - R * 0.75); grow(x + R * 1.25, y + R * 0.75);
      return `<ellipse cx="${f1(x)}" cy="${f1(y)}" rx="${f1(R * Math.SQRT2 * C30)}" ry="${f1(R * Math.SQRT2 * S30)}"${extra}/>`; };
    let ground = '<g class="land">' + [...pos.values()].filter(p => !p.loose).map(p => disc(p, 46)).join('') + '</g>';
    if (L.loose.length) {
      ground += '<g class="land uncharted">' + L.loose.map(id => disc(pos.get(id), 26)).join('') + '</g>';
      const lp = L.loose.map(id => P(pos.get(id).u, pos.get(id).v));
      const c = [lp.reduce((a, q) => a + q[0], 0) / lp.length, Math.max(...lp.map(q => q[1]))];
      ground += `<text class="district-name uncharted" x="${f1(c[0])}" y="${f1(c[1] + 34)}">UNCHARTED · no meaning vector, not linked to a placed paper</text>`;
    }
    let districts = '', names = '';
    const nameBoxes = [];
    (data.regions || []).forEach(reg => {
      const members = Object.keys(N).filter(id => N[id].region === reg.id && pos.has(id));
      if (!members.length) return;
      const pts = members.map(id => P(pos.get(id).u, pos.get(id).v));
      const col = LAND[reg.id % LAND.length], ink = LAND_INK[reg.id % LAND_INK.length];
      districts += `<g class="district" data-region="${reg.id}" fill="${col}">${members.map(id => disc(pos.get(id), 30)).join('')}</g>`;
      const c = pts.reduce((a, q) => [a[0] + q[0] / pts.length, a[1] + q[1] / pts.length], [0, 0]);
      const words = reg.name.toUpperCase().split(' '), lines = [''];
      words.forEach(w => { if ((lines.at(-1) + ' ' + w).trim().length > 24) lines.push(w); else lines[lines.length - 1] = (lines.at(-1) + ' ' + w).trim(); });
      let y = c[1] + 4;
      const w = Math.max(...lines.map(l => l.length)) * 7.4, hgt = lines.length * 14;
      while (nameBoxes.some(b => c[0] - w / 2 < b.x1 && c[0] + w / 2 > b.x0 && y - 12 < b.y1 && y - 12 + hgt > b.y0)) y += 15;
      nameBoxes.push({x0: c[0] - w / 2, x1: c[0] + w / 2, y0: y - 12, y1: y - 12 + hgt});
      grow(c[0] - w / 2, y); grow(c[0] + w / 2, y + hgt);
      const share = reg.share ? ` (${Math.round(reg.share * 100)}% of its papers)` : '';
      names += `<text class="district-name" fill="${ink}" x="${f1(c[0])}" y="${f1(y)}"><title>${esc(reg.name)}: ${reg.size} papers; named after the most common OpenAlex ${reg.name_kind}${share}</title>${lines.map((l, i) => `<tspan x="${f1(c[0])}" dy="${i ? 14 : 0}">${esc(l)}</tspan>`).join('')}</text>`;
    });
    // citations: rivers into the answer (converging), roads from X, streets for the rest
    const answers = new Set(data.answers);
    let rivers = '', roads = '', streets = '';
    const conf = new Map();                            // confluence point per answer
    data.answers.forEach(a => {
      const src = data.links.filter(l => l.t === a && pos.has(l.s)).map(l => pos.get(l.s)), t = pos.get(a);
      if (!src.length || !t) return;
      const mu = src.reduce((s, p) => s + p.u, 0) / src.length, mv = src.reduce((s, p) => s + p.v, 0) / src.length;
      conf.set(a, {u: t.u + (mu - t.u) * 0.35, v: t.v + (mv - t.v) * 0.35, n: src.length});
    });
    data.answers.forEach(a => {
      const c = conf.get(a), t = pos.get(a); if (!c) return;
      rivers += `<path class="river mouth" style="stroke-width:${f1(3 + c.n * 0.9)}" d="M${pt(c.u, c.v)} L${pt(t.u, t.v)}"/>`;
    });
    data.links.forEach((l, i) => {
      const a = pos.get(l.s), b = pos.get(l.t); if (!a || !b) return;
      if (l.kind === 'y-cand' && answers.has(l.t) && conf.get(l.t)) {
        const c = conf.get(l.t);
        const mid = {u: (a.u + c.u) / 2, v: (a.v + c.v) / 2};
        rivers += `<path class="river" data-i="${i}" d="M${pt(a.u, a.v)} Q${pt(mid.u + (c.v - a.v) * 0.12, mid.v - (c.u - a.u) * 0.12)} ${pt(c.u, c.v)}"/>`;
        return;
      }
      const bend = 0.15 * ((i % 2) ? 1 : -1);
      const m = {u: (a.u + b.u) / 2 + (b.v - a.v) * bend, v: (a.v + b.v) / 2 - (b.u - a.u) * bend};
      const d = `M${pt(a.u, a.v)} Q${pt(m.u, m.v)} ${pt(b.u, b.v)}`;
      if (l.kind === 'x-ref') roads += `<path class="road-case" data-i="${i}" d="${d}"/><path class="road" data-i="${i}" d="${d}"/>`;
      else streets += `<path class="street ${l.kind}" data-i="${i}" d="${d}"/>`;
    });
    // buildings, back to front
    const order = [...pos.keys()].sort((a, b) => (pos.get(a).u + pos.get(a).v) - (pos.get(b).u + pos.get(b).v));
    let blocks = '';
    order.forEach(id => {
      const p = pos.get(id), n = N[id];
      const [x, y] = P(p.u, p.v); grow(x - 30, y - p.h - 30); grow(x + 30, y + 20);
      blocks += `<g class="blk${n.column.endsWith('_out') ? ' out' : ''}${p.drift ? ' drift' : ''}" data-id="${id}" data-year="${n.year ?? ''}">${building(id, p, n, ROLE[roleOf(n)])}</g>`;
    });
    // labels: X and the answer(s)
    let labels = '';
    const boxes = [];
    [data.focus, ...data.answers].forEach(id => {
      const p = pos.get(id); if (!p) return;
      const [x, y0] = P(p.u, p.v), y = y0 - p.h, txt = (id === data.focus ? 'X · ' : '') + short(N[id].title, 48);
      const w = txt.length * 6.7 + 8;
      let ly = y - 16;
      while (boxes.some(b => x - 4 < b.x1 && x - 4 + w > b.x0 && ly - 13 < b.y1 && ly + 4 > b.y0)) ly -= 17;
      boxes.push({x0: x - 4, x1: x - 4 + w, y0: ly - 13, y1: ly + 4});
      grow(x + w, ly - 16);
      labels += `<line class="lead" x1="${f1(x)}" y1="${f1(y - 2)}" x2="${f1(x)}" y2="${f1(ly + 4)}"/>
        <text class="lbl ${id === data.focus ? 'lx' : 'la'}" x="${f1(x - 4)}" y="${f1(ly)}">${esc(txt)}</text>`;
    });
    svg.innerHTML = `<g class="ground">${ground}${districts}</g>
      <g class="streets">${streets}</g><g class="roads">${roads}</g><g class="rivers">${rivers}</g>
      <g class="blocks">${blocks}</g><g class="names">${names}</g><g class="labels">${labels}</g><text id="map-hover" class="lbl lh" x="0" y="0"></text>`;
    st = {svg, data, hooks, pos, idx: new Map(), bb};
    data.links.forEach((l, i) => { for (const id of [l.s, l.t]) { if (!st.idx.has(id)) st.idx.set(id, []); st.idx.get(id).push(i); } });
    fit();
    svg.querySelectorAll('.blk').forEach(g => {
      g.addEventListener('mouseenter', ev => hooks.hover(g.dataset.id, ev));
      g.addEventListener('mousemove', ev => hooks.move(ev));
      g.addEventListener('mouseleave', () => hooks.leave());
      g.addEventListener('click', () => hooks.select(g.dataset.id));
      g.addEventListener('dblclick', () => hooks.walk(g.dataset.id));
    });
    const years = Object.values(N).map(n => n.year).filter(y => y != null);
    return {withVector: Object.values(N).filter(n => n.mx != null).length, nodes: Object.keys(N).length,
            drift: [...pos.values()].filter(p => p.drift).length, minYear: Math.min(...years), maxYear: Math.max(...years)};
  }

  function highlight(id, showAll) {
    if (!st) return;
    const svg = st.svg;
    svg.classList.toggle('focusing', !!id);
    svg.classList.toggle('all', !!showAll);
    svg.querySelectorAll('.on').forEach(e => e.classList.remove('on'));
    svg.querySelectorAll('.lit').forEach(e => e.classList.remove('lit'));
    const hov = svg.querySelector('#map-hover');
    hov.textContent = '';
    if (!id) return;
    const lit = new Set([id]);
    (st.idx.get(id) || []).forEach(i => {
      const l = st.data.links[i]; lit.add(l.s); lit.add(l.t);
      svg.querySelectorAll(`[data-i="${i}"]`).forEach(e => e.classList.add('on'));
    });
    lit.forEach(n => { const g = svg.querySelector(`.blk[data-id="${n}"]`); if (g) g.classList.add('lit'); });
    const p = st.pos.get(id);
    if (p && id !== st.data.focus && !st.data.answers.includes(id)) {
      const [x, y] = P(p.u, p.v);
      hov.setAttribute('x', f1(x + 8)); hov.setAttribute('y', f1(y - p.h - 8));
      hov.textContent = short(st.data.nodes[id].title, 60);
    }
  }

  function untilYear(year) {                       // fade buildings (and their citations) newer than year
    if (!st) return;
    st.svg.querySelectorAll('.blk').forEach(g => g.classList.toggle('future', g.dataset.year !== '' && +g.dataset.year > year));
    const future = new Set([...st.svg.querySelectorAll('.blk.future')].map(g => g.dataset.id));
    st.svg.querySelectorAll('[data-i]').forEach(e => { const l = st.data.links[+e.dataset.i]; e.classList.toggle('future', future.has(l.s) || future.has(l.t)); });
  }

  let vb = null;
  function setVB() { st.svg.setAttribute('viewBox', vb.map(f1).join(' ')); }
  function fit() {
    const pad = 20, b = st.bb;
    if (![b.x0, b.y0, b.x1, b.y1].every(Number.isFinite)) throw new Error('map: invalid scene bounds');
    vb = [b.x0 - pad, b.y0 - pad, b.x1 - b.x0 + 2 * pad, b.y1 - b.y0 + 2 * pad]; setVB();
  }
  function attachPanZoom(svg) {
    let drag = null;
    svg.addEventListener('wheel', ev => {
      if (!vb) return; ev.preventDefault();
      const r = svg.getBoundingClientRect(), s = Math.exp(ev.deltaY * 0.0015);
      const scale = Math.max(vb[2] / r.width, vb[3] / r.height);
      const cx = vb[0] + (ev.clientX - r.left - (r.width - vb[2] / scale) / 2) * scale;
      const cy = vb[1] + (ev.clientY - r.top - (r.height - vb[3] / scale) / 2) * scale;
      vb = [cx - (cx - vb[0]) * s, cy - (cy - vb[1]) * s, vb[2] * s, vb[3] * s]; setVB();
    }, {passive: false});
    svg.addEventListener('mousedown', ev => { if (ev.target.closest('.blk')) return; drag = {x: ev.clientX, y: ev.clientY, vb: [...vb]}; svg.style.cursor = 'grabbing'; });
    window.addEventListener('mousemove', ev => {
      if (!drag) return;
      const r = svg.getBoundingClientRect(), scale = Math.max(vb[2] / r.width, vb[3] / r.height);
      vb = [drag.vb[0] - (ev.clientX - drag.x) * scale, drag.vb[1] - (ev.clientY - drag.y) * scale, vb[2], vb[3]]; setVB();
    });
    window.addEventListener('mouseup', () => { drag = null; svg.style.cursor = ''; });
  }
  return {render, highlight, fit, attachPanZoom, untilYear};
})();
