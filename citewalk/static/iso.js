// Timeline view, 3-D: vertical = publication year (newest at the top); horizontal = position found
// only by citation forces; depth = semantic distance to X (DenseOn vectors). Drag rotates around time.
// Every publication in the analysis is one small 3-D cube; size ~ log(citations), colour = role.
const ISO = (() => {
  const YU = 36;                                  // px per year step
  const STEPS = 600, K = 0.02, R = 90, G = 0.004, MAXSTEP = 25;
  const COLORS = {                                 // [front, top, side]
    x: ['#2a3037', '#4a535e', '#1b1f24'],
    ref: ['#9aa3ad', '#c3c9d0', '#7f8892'],
    similar: ['#2f6fb3', '#6d9edb', '#24578f'],
    candidate: ['#e09a9a', '#f3c1c1', '#c98282'],
    answer: ['#c62828', '#ef5350', '#9b1c1c'],
  };
  const f1 = v => v.toFixed(1);
  let st = null;

  const roleOf = n => n.column === 'x' ? 'x' : n.column.startsWith('refs') ? 'ref'
    : n.column.startsWith('similar') ? 'similar' : n.answer ? 'answer' : 'candidate';

  function rng(seed) {                             // deterministic layout for a given paper
    let a = 0; for (const ch of seed) a = (a * 31 + ch.charCodeAt(0)) | 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
                   t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  function crossings(E, X, Y) {
    const segs = E.map(([a, b]) => [[X[a], Y[a]], [X[b], Y[b]], a, b]);
    const ccw = (A, B, C) => (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0]);
    let n = 0;
    for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
      const [p, q, a, b] = segs[i], [r, s, c, d] = segs[j];
      if (a === c || a === d || b === c || b === d) continue;
      if (ccw(p, r, s) !== ccw(q, r, s) && ccw(p, q, r) !== ccw(p, q, s)) n++;
    }
    return n;
  }

  function layout(data) {
    const N = data.nodes, ids = Object.keys(N), n = ids.length;
    const index = new Map(ids.map((id, i) => [id, i]));
    const yearOf = id => N[id].year ?? null;
    const years = [...new Set(ids.map(yearOf).filter(y => y != null))].sort((a, b) => a - b);
    const S = ids.map(id => id === data.focus ? 22 : Math.min(22, 8 + 4 * Math.log10(1 + Math.max(0, N[id].cited_by))));
    const E = data.links.map(l => [index.get(l.s), index.get(l.t)]).filter(([a, b]) => a != null && b != null);
    const fixed = index.get(data.focus);
    // provisional vertical position for the forces: one step per year (gaps shortened)
    const step = new Map(); let z = 0;
    years.forEach((y, i) => { if (i) z += y - years[i - 1] > 1 ? 1.7 : 1; step.set(y, z); });
    const Y0 = ids.map(id => -((yearOf(id) != null ? step.get(yearOf(id)) : -1.6) * YU));
    // 1. horizontal positions from citation forces only
    const rand = rng(data.focus);
    const X = new Float64Array(n).map(() => (rand() - 0.5) * 60 * Math.sqrt(n));
    X[fixed] = 0;
    for (let t = 0; t < STEPS; t++) {
      const alpha = 1 - t / STEPS, F = new Float64Array(n);
      for (const [a, b] of E) { const d = X[b] - X[a]; F[a] += K * d; F[b] -= K * d; }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const dy = Y0[i] - Y0[j]; if (Math.abs(dy) > 4 * YU) continue;           // repulsion acts between nearby years
        const dx = X[i] - X[j] || (i < j ? -0.01 : 0.01);
        const f = R * 40 * dx / (dx * dx + dy * dy + 100);
        F[i] += f; F[j] -= f;
      }
      for (let i = 0; i < n; i++) {
        if (i === fixed) continue;
        F[i] -= G * X[i];
        X[i] += Math.max(-MAXSTEP, Math.min(MAXSTEP, F[i] * alpha));
      }
    }
    // 2. one uniform horizontal scale so the picture is not too wide (keeps order and proportions)
    const W = Math.max(500, Math.min(1100, 34 * Math.sqrt(n) * 4));
    const sorted = [...X].sort((a, b) => a - b);
    const q = f => sorted[Math.min(n - 1, Math.floor(f * (n - 1)))];
    const half = Math.max(Math.abs(q(0.03)), Math.abs(q(0.97)), 1);
    const scale = Math.min(1, (W / 2) / half);
    for (let i = 0; i < n; i++) X[i] = Math.max(-W / 2 - 60, Math.min(W / 2 + 60, X[i] * scale));
    // 3. vertical: one band per year, newest on top; crowded years get sub-rows (still within their year)
    const SUB = 26, members = new Map();
    ids.forEach((id, i) => { const y = yearOf(id); if (!members.has(y)) members.set(y, []); members.get(y).push(i); });
    const capacity = Math.max(4, Math.floor((W + 120) / 34));
    const bandTop = new Map(), subrows = new Map(), breaks = [];
    let yCursor = 0;                      // grows upward (negative screen y); oldest first
    const ordered = [...(members.has(null) ? [null] : []), ...years];
    ordered.forEach((y, k) => {
      if (k && y != null && ordered[k - 1] != null && y - ordered[k - 1] > 1) {
        yCursor -= 0.7 * YU; breaks.push({y: yCursor + 0.35 * YU + 5, from: ordered[k - 1], to: y});
      }
      const m = members.get(y), rows = Math.ceil(m.length / capacity);
      subrows.set(y, rows);
      bandTop.set(y, yCursor);
      yCursor -= YU + (rows - 1) * SUB;
    });
    const Y = new Float64Array(n);
    members.forEach((m, y) => {
      const rows = subrows.get(y), base = bandTop.get(y);
      m.sort((a, b) => X[a] - X[b]);
      const lines = Array.from({length: rows}, () => []);
      m.forEach((i, k) => lines[k % rows].push(i));
      lines.forEach((line, r) => {
        line.forEach(i => { Y[i] = base - r * SUB; });
        for (let k = 1; k < line.length; k++) {               // no overlaps on a line
          const a = line[k - 1], b = line[k], gap = (S[a] + S[b]) / 2 + 8;
          if (X[b] - X[a] < gap) X[b] = X[a] + gap;
        }
      });
    });
    if (members.get(yearOf(data.focus))) {                     // keep X itself at the centre of its line
      const line = members.get(yearOf(data.focus)).filter(i => Y[i] === Y[fixed]), d = -X[fixed];
      line.forEach(i => X[i] += d);
    }
    const after = crossings(E, X, Y);
    const R2 = rng(data.focus + '#baseline');
    const before = crossings(E, X.map(() => (R2() - 0.5) * W), Y);
    const pos = new Map(ids.map((id, i) => [id, {x: X[i], y: Y[i], s: S[i]}]));
    return {pos, years, bandTop, subrows, breaks, unknown: members.has(null), before, after};
  }

  function cube(x, y, s, c) {     // cube glyph: front square centred on (x, y), depth up and to the right
    const h = s / 2, dx = s * 0.45, dy = -s * 0.35;
    const P = pts => pts.map(([a, b]) => f1(a) + ',' + f1(b)).join(' ');
    return `<polygon points="${P([[x - h, y - h], [x + h, y - h], [x + h + dx, y - h + dy], [x - h + dx, y - h + dy]])}" fill="${c[1]}"/>
      <polygon points="${P([[x + h, y - h], [x + h, y + h], [x + h + dx, y + h + dy], [x + h + dx, y - h + dy]])}" fill="${c[2]}"/>
      <polygon class="front" points="${P([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])}" fill="${c[0]}"/>`;
  }

  // ── 3-D: x = citation forces, y = time (vertical), z = semantic distance to X ──
  const cam = {yaw: -0.6, pitch: 0.16};
  function project(x, y, z) {
    const c = Math.cos(cam.yaw), s = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const xr = x * c - z * s, zr = x * s + z * c;          // rotate around the vertical (time) axis
    return {x: xr, y: y * cp - zr * sp, d: zr * cp + y * sp};
  }

  function render(svg, data, hooks) {
    const N = data.nodes, L = layout(data);
    const xs = [...L.pos.values()].map(p => p.x);
    const x0 = Math.min(...xs) - 40, x1 = Math.max(...xs) + 40;
    const sems = Object.values(N).map(n => n.sem).filter(v => v != null);
    const semMax = Math.max(0.05, ...sems);
    const DEPTH = Math.max(300, (x1 - x0) * 0.55);
    const NOVEC = DEPTH * 1.18;
    L.pos.forEach((p, id) => { p.z = N[id].sem != null ? N[id].sem / semMax * DEPTH : NOVEC; p.novec = N[id].sem == null; });
    st = {svg, data, hooks, L, links: data.links, idx: new Map(), x0, x1, DEPTH, NOVEC, semMax, hasSem: sems.length > 0};
    data.links.forEach((l, i) => { for (const id of [l.s, l.t]) { if (!st.idx.has(id)) st.idx.set(id, []); st.idx.get(id).push(i); } });
    draw();
    fit();
    return {crossings: L.after, before: L.before, withVector: sems.length, nodes: Object.keys(N).length};
  }

  function draw() {
    const {svg, data, hooks, L, x0, x1, DEPTH, NOVEC, hasSem} = st, N = data.nodes;
    const zFar = hasSem ? DEPTH : 0;
    const bb = {x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9};
    const grow = (x, y) => { bb.x0 = Math.min(bb.x0, x); bb.x1 = Math.max(bb.x1, x); bb.y0 = Math.min(bb.y0, y); bb.y1 = Math.max(bb.y1, y); };
    const pr = new Map();
    L.pos.forEach((p, id) => { const q = project(p.x, p.y, hasSem ? p.z : 0); pr.set(id, {...q, s: p.s}); grow(q.x - 20, q.y - 20); grow(q.x + 20, q.y + 20); });
    // year planes: front edge with the year, plus a faint outline of the plane
    let grid = '';
    const line = (a, b, cls) => `<line class="${cls}" x1="${f1(a.x)}" y1="${f1(a.y)}" x2="${f1(b.x)}" y2="${f1(b.y)}"/>`;
    L.bandTop.forEach((y, yr) => {
      const A = project(x0, y, 0), B = project(x1, y, 0), C = project(x1, y, zFar), D = project(x0, y, zFar);
      grid += `<polygon class="plane${yr == null ? ' unknown' : ''}" points="${[A, B, C, D].map(q => f1(q.x) + ',' + f1(q.y)).join(' ')}"/>`;
      grid += line(A, B, 'yline' + (yr == null ? ' unknown' : ''));
      const T = A.x < D.x ? A : D;
      grid += `<text class="year" x="${f1(Math.min(A.x, D.x) - 8)}" y="${f1((A.x < D.x ? A.y : D.y) + 4)}" text-anchor="end">${yr == null ? 'year ?' : yr}</text>`;
      grow(Math.min(A.x, D.x) - 60, A.y); grow(Math.max(B.x, C.x), C.y);
    });
    L.breaks.forEach(b => { const A = project(x0, b.y, 0); grid += `<text class="gap" x="${f1(A.x - 8)}" y="${f1(A.y)}" text-anchor="end">⋯</text>`; });
    if (hasSem) {                                     // depth axis on the oldest plane, and the "no vector" strip
      const yb = Math.max(...L.bandTop.values()) + 18;
      const A = project(x0, yb, 0), B = project(x0, yb, DEPTH), C = project(x0, yb, NOVEC);
      grid += `<line class="axis" x1="${f1(A.x)}" y1="${f1(A.y)}" x2="${f1(B.x)}" y2="${f1(B.y)}" marker-end="url(#axis)"/>
        <text class="axis-name" x="${f1((A.x + B.x) / 2)}" y="${f1((A.y + B.y) / 2 + 16)}" text-anchor="middle">further in meaning from X →</text>
        <text class="axis-name novec" x="${f1(C.x)}" y="${f1(C.y + 16)}" text-anchor="middle">no meaning vector</text>`;
      grow(A.x, A.y + 24); grow(B.x, B.y + 24); grow(C.x + 60, C.y + 24);
    }
    // links (screen-space S curves from the citing paper to the cited one)
    const answers = new Set(data.answers);
    let halos = '', links = '';
    data.links.forEach((l, i) => {
      const a = pr.get(l.s), b = pr.get(l.t); if (!a || !b) return;
      let d;
      if (Math.abs(a.y - b.y) < 2) {
        const lift = 18 + Math.abs(b.x - a.x) * 0.25;
        d = `M${f1(a.x)},${f1(a.y - a.s / 2)} Q${f1((a.x + b.x) / 2)},${f1(Math.min(a.y, b.y) - lift)} ${f1(b.x)},${f1(b.y - b.s / 2)}`;
      } else {
        const sg = Math.sign(b.y - a.y), ya = a.y + sg * a.s / 2, yb = b.y - sg * b.s / 2, m = (ya + yb) / 2;
        d = `M${f1(a.x)},${f1(ya)} C${f1(a.x)},${f1(m)} ${f1(b.x)},${f1(m)} ${f1(b.x)},${f1(yb)}`;
      }
      const cls = `${l.kind}${l.kind === 'y-cand' && answers.has(l.t) ? ' to-answer' : ''}`;
      const mk = l.kind === 'x-ref' ? 'g' : l.kind.startsWith('y-cand') ? (answers.has(l.t) ? 'r' : 'p') : 'b';
      halos += `<path class="halo ${cls}" data-i="${i}" d="${d}"/>`;
      links += `<path class="lk ${cls}" data-i="${i}" d="${d}" marker-end="url(#ar-${mk})"/>`;
    });
    // cubes, far to near
    const order = [...pr.keys()].sort((a, b) => pr.get(b).d - pr.get(a).d);
    let blocks = '';
    order.forEach(id => {
      const p = pr.get(id), n = N[id];
      blocks += `<g class="blk${n.column.endsWith('_out') ? ' out' : ''}${L.pos.get(id).novec && hasSem ? ' novec' : ''}" data-id="${id}">${cube(p.x, p.y, p.s, COLORS[roleOf(n)])}</g>`;
    });
    let labels = '';
    const boxes = [];
    [data.focus, ...data.answers].forEach(id => {
      const p = pr.get(id), txt = (id === data.focus ? 'X · ' : '') + short(N[id].title, 50);
      const w = txt.length * 6.7 + 8, lx = p.x + p.s / 2 + 12;
      let ly = p.y + 4;
      while (boxes.some(b => lx < b.x1 && lx + w > b.x0 && ly - 13 < b.y1 && ly + 4 > b.y0)) ly += 17;
      boxes.push({x0: lx, x1: lx + w, y0: ly - 13, y1: ly + 4});
      grow(lx + w, ly + 6);
      labels += `<line class="lead" x1="${f1(p.x + p.s / 2 + 2)}" y1="${f1(p.y)}" x2="${f1(lx - 2)}" y2="${f1(ly - 4)}"/>
        <text class="lbl ${id === data.focus ? 'lx' : 'la'}" x="${f1(lx)}" y="${f1(ly)}">${esc(txt)}</text>`;
    });
    const mk = (id, color) => `<marker id="ar-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${color}"/></marker>`;
    svg.innerHTML = `<defs>${mk('g', '#7d8792')}${mk('b', '#2f6fb3')}${mk('p', '#d47c7c')}${mk('r', '#c62828')}
      <marker id="axis" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8b949e"/></marker></defs>
      <g class="floor">${grid}</g><g class="halos">${halos}</g><g class="links">${links}</g><g class="blocks">${blocks}</g>
      <g class="labels">${labels}</g><text id="iso-hover" class="lbl lh" x="0" y="0"></text>`;
    st.pr = pr; st.bb = bb;
    svg.querySelectorAll('.blk').forEach(g => {
      g.addEventListener('mouseenter', ev => hooks.hover(g.dataset.id, ev));
      g.addEventListener('mousemove', ev => hooks.move(ev));
      g.addEventListener('mouseleave', () => hooks.leave());
      g.addEventListener('click', () => hooks.select(g.dataset.id));
      g.addEventListener('dblclick', () => hooks.walk(g.dataset.id));
    });
    if (st.lastHighlight !== undefined) highlight(st.lastHighlight, st.lastAll);
  }

  function highlight(id, showAll) {
    if (!st) return;
    st.lastHighlight = id; st.lastAll = showAll;
    const svg = st.svg;
    svg.classList.toggle('focusing', !!id);
    svg.classList.toggle('all', !!showAll);
    svg.querySelectorAll('.on').forEach(e => e.classList.remove('on'));
    svg.querySelectorAll('.lit').forEach(e => e.classList.remove('lit'));
    const hov = svg.querySelector('#iso-hover');
    hov.textContent = '';
    if (!id) return;
    const lit = new Set([id]);
    (st.idx.get(id) || []).forEach(i => {
      const l = st.links[i]; lit.add(l.s); lit.add(l.t);
      svg.querySelectorAll(`[data-i="${i}"]`).forEach(e => e.classList.add('on'));
    });
    lit.forEach(n => { const g = svg.querySelector(`.blk[data-id="${n}"]`); if (g) g.classList.add('lit'); });
    const p = st.pr.get(id);
    if (p && id !== st.data.focus && !st.data.answers.includes(id)) {
      hov.setAttribute('x', f1(p.x + p.s / 2 + 8)); hov.setAttribute('y', f1(p.y - p.s / 2 - 6));
      hov.textContent = short(st.data.nodes[id].title, 60);
    }
  }

  let vb = null;
  function setVB() { st.svg.setAttribute('viewBox', vb.map(f1).join(' ')); }
  function fit() {
    const pad = 20, b = st.bb;
    if (![b.x0, b.y0, b.x1, b.y1].every(Number.isFinite)) throw new Error('timeline: invalid scene bounds');
    vb = [b.x0 - pad, b.y0 - pad, b.x1 - b.x0 + 2 * pad, b.y1 - b.y0 + 2 * pad];
    setVB();
  }
  let raf = 0;
  function redraw() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
  function setView(yaw, pitch) { cam.yaw = yaw; cam.pitch = pitch; if (st) { draw(); fit(); } }
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
    svg.addEventListener('mousedown', ev => {
      if (ev.target.closest('.blk')) return;
      drag = {x: ev.clientX, y: ev.clientY, vb: [...vb], yaw: cam.yaw, pitch: cam.pitch, pan: ev.shiftKey || ev.button === 1};
      svg.style.cursor = 'grabbing'; ev.preventDefault();
    });
    window.addEventListener('mousemove', ev => {
      if (!drag) return;
      if (drag.pan) {
        const r = svg.getBoundingClientRect(), scale = Math.max(vb[2] / r.width, vb[3] / r.height);
        vb = [drag.vb[0] - (ev.clientX - drag.x) * scale, drag.vb[1] - (ev.clientY - drag.y) * scale, vb[2], vb[3]]; setVB();
      } else {                       // rotate around the time axis; tilt to look from above
        cam.yaw = drag.yaw + (ev.clientX - drag.x) * 0.008;
        cam.pitch = Math.max(0, Math.min(1.1, drag.pitch + (ev.clientY - drag.y) * 0.006));
        redraw();
      }
    });
    window.addEventListener('mouseup', () => { drag = null; svg.style.cursor = ''; });
  }
  return {render, highlight, fit, attachPanZoom, setView};
})();
