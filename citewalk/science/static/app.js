'use strict';
// Atlas of science — pilot viewer. Positions are fixed by the build; the year controls only
// choose which (year-sorted) slice of the point array is drawn.

const $ = (id) => document.getElementById(id);
const S = { year: 2026, window: 5, colour: 'field', eq: true, focus: -1, selected: -1, similar: null,
            show: 'terr', lag: 10, raceHover: -1,
            search: null, playing: false, view: { cx: .5, cy: .5, zoom: 1 },
            t: 2026, target: 2026 };   // continuous time: dots of year Y appear while t goes from Y-1 to Y
let M, N, XY, YEAR, TIME, FIELD, SUB, gl, prog, quad, fbo, canvas, overlay, octx, dirty = true;
let fieldPrefix, subPrefix, grid;
const GRID = 512;

// ---------- loading ----------
async function load() {
  M = await (await fetch('/data/meta.json')).json();
  const b = '?b=' + encodeURIComponent(M.id);
  const bin = async (name) => (await fetch('/data/' + name + b)).arrayBuffer();
  const [p, y, f, s] = await Promise.all([bin('points.f32'), bin('year.u16'), bin('field.u8'), bin('subfield.u16')]);
  XY = new Float32Array(p); YEAR = new Uint16Array(y); FIELD = new Uint8Array(f); SUB = new Uint16Array(s);
  N = YEAR.length;
  // Each dot gets its own moment inside its year (dots are stored in random order within a year),
  // so a year's papers trickle in instead of appearing at once. The moment is a display device:
  // many OpenAlex dates are only a year.
  TIME = new Float32Array(N);
  for (let j = 0; j < M.years.length; j++) {
    const a0 = M.offsets[j], c = M.offsets[j + 1] - a0;
    for (let k = 0; k < c; k++) TIME[a0 + k] = M.years[j] - 1 + (k + .5) / c;
  }
  const prefix = (list) => list.map((l) => { const a = new Int32Array(l.per_year.length + 1);
    l.per_year.forEach((c, i) => a[i + 1] = a[i] + c); return a; });
  fieldPrefix = prefix(M.fields); subPrefix = prefix(M.subfields);
  buildGrid();
  readHash();
  setupGL(); layerSetup(); setupUI(); resize();
  $('loading').remove();
  requestAnimationFrame(frame);
}

function buildGrid() {
  const cell = new Int32Array(N), count = new Int32Array(GRID * GRID + 1);
  for (let i = 0; i < N; i++) {
    const gx = Math.min(GRID - 1, Math.max(0, (XY[2 * i] * GRID) | 0));
    const gy = Math.min(GRID - 1, Math.max(0, (XY[2 * i + 1] * GRID) | 0));
    cell[i] = gy * GRID + gx; count[cell[i] + 1]++;
  }
  for (let c = 0; c < GRID * GRID; c++) count[c + 1] += count[c];
  const fill = count.slice(0, GRID * GRID), idx = new Int32Array(N);
  for (let i = 0; i < N; i++) idx[fill[cell[i]]++] = i;
  grid = { start: count, idx };
}

// ---------- year range ----------
const yi = (y) => Math.max(0, Math.min(M.years.length - 1, y - M.years[0]));
const win = () => (S.show !== 'dots' && !S.window ? 5 : S.window);
function range() {
  const i = yi(S.year), end = M.offsets[i + 1];
  const start = win() ? M.offsets[Math.max(0, i - win() + 1)] : 0;
  return [start, end];
}
// dot index range of the w years ending at yEnd (empty before the first year)
function rangeFor(yEnd, w) {
  const i = yEnd - M.years[0];
  if (i < 0) return [0, 0];
  const j = Math.min(i, M.years.length - 1);
  return [M.offsets[Math.max(0, i - w + 1)], M.offsets[j + 1]];
}
function yearSpan() { return [win() ? Math.max(M.years[0], S.year - win() + 1) : M.years[0], S.year]; }
// number of dots (per prefix array over years) whose moment is ≤ x; continuous in x
function cumAt(pre, x) {
  const Y = Math.ceil(x - 1e-9), i = Y - M.years[0];
  if (i < 0) return 0;
  if (i >= M.years.length) return pre[M.years.length];
  const f = Math.min(1, Math.max(0, x - (Y - 1)));
  return pre[i] + f * (pre[i + 1] - pre[i]);
}
// dots currently in the window (continuous), e.g. windowAt(M.offsets) or a field's prefix
const windowAt = (pre) => cumAt(pre, S.t) - (win() ? cumAt(pre, S.t - win()) : 0);
// dot index range that can be (partly) visible at time S.t
function drawRange() {
  const at = (Y) => M.offsets[Math.min(M.years.length, Math.max(0, Y - M.years[0]))];
  const end = at(Math.ceil(S.t - 1e-9) + 1);
  return [win() ? at(Math.ceil(S.t - win() - 1e-9)) : 0, end];
}
const visible = (i, start, end) => i >= start && i < end && (S.focus < 0 || FIELD[i] === S.focus);

// ---------- WebGL ----------
const VS = `#version 300 es
in vec2 a_pos; in float a_t; in float a_field;
uniform vec2 u_center, u_scale; uniform float u_size, u_t, u_window, u_mode, u_focus;
uniform vec3 u_pal[32];
out vec4 v_col;
vec3 ramp(float t) {               // dark blue -> teal -> yellow (perceptually ordered)
  vec3 a = vec3(.23,.30,.75), b = vec3(.10,.70,.65), c = vec3(.99,.85,.25);
  return t < .5 ? mix(a, b, t * 2.) : mix(b, c, t * 2. - 1.);
}
void main() {
  vec2 p = (a_pos - u_center) * u_scale;
  gl_Position = vec4(p.x, -p.y, 0., 1.);
  float age = u_t - a_t;                      // years since this dot's moment
  float w = smoothstep(0., .4, age);           // fades in over ~5 months
  if (u_window > .5) {                         // ... and out as it leaves the window
    w *= 1. - smoothstep(u_window - .6, u_window, age);
    w *= mix(1., .45, clamp(age / u_window, 0., 1.));
  }
  if (w <= 0.) { gl_Position = vec4(2., 2., 2., 1.); gl_PointSize = 0.; v_col = vec4(0.); return; }
  gl_PointSize = u_size;
  vec3 c = u_mode < .5 ? u_pal[int(a_field)] : u_mode < 1.5 ? ramp(clamp((a_t + 1. - 1900.) / 126., 0., 1.)) : vec3(.62, .66, .74);
  if (u_mode < 1.5) c = mix(c, vec3(1.), .3 * (1. - smoothstep(0., 1., age)));   // newest glow
  if (u_focus > -.5 && abs(a_field - u_focus) > .5) w *= .05;
  v_col = vec4(c * w, w);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 v_col; uniform float u_size; out vec4 o;
void main() {
  float k = 1.;
  if (u_size > 2.5) { vec2 d = gl_PointCoord - .5; float r = dot(d, d) * 4.; if (r > 1.) discard; k = 1. - .6 * r * r; }
  o = v_col * k;
}`;
const QVS = `#version 300 es
in vec2 a; out vec2 uv; void main() { uv = a * .5 + .5; gl_Position = vec4(a, 0., 1.); }`;
const QFS = `#version 300 es
precision highp float;
in vec2 uv; uniform sampler2D t; uniform float u_k; out vec4 o;
void main() {
  vec4 acc = texture(t, uv);
  if (acc.a <= 0.) { o = vec4(0., 0., 0., 1.); return; }
  vec3 mean = acc.rgb / acc.a;
  float b = 1. - exp(-u_k * pow(acc.a, .7));
  vec3 c = mean * b + vec3(1.) * smoothstep(12., 80., acc.a * u_k * 4.) * .18;
  o = vec4(pow(c, vec3(.92)), 1.);
}`;

function compile(vs, fs) {
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name.replace('[0]', ''); u[name] = gl.getUniformLocation(p, name); }
  return { p, u };
}

function setupGL() {
  canvas = $('gl'); overlay = $('overlay'); octx = overlay.getContext('2d');
  gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
  if (!gl) { document.body.innerHTML = '<p style="padding:20px">This page needs WebGL2.</p>'; throw new Error('no webgl2'); }
  const floatOk = gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float');
  prog = compile(VS, FS);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buf = (data, loc, size, type) => {
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, false, 0, 0);
  };
  buf(XY, gl.getAttribLocation(prog.p, 'a_pos'), 2, gl.FLOAT);
  buf(TIME, gl.getAttribLocation(prog.p, 'a_t'), 1, gl.FLOAT);
  buf(FIELD, gl.getAttribLocation(prog.p, 'a_field'), 1, gl.UNSIGNED_BYTE);
  prog.vao = vao;
  const pal = new Float32Array(96);
  M.fields.forEach((f, i) => { const h = f.colour; for (let k = 0; k < 3; k++) pal[3 * i + k] = parseInt(h.substr(1 + 2 * k, 2), 16) / 255; });
  prog.pal = pal;
  if (floatOk) {
    quad = compile(QVS, QFS);
    quad.vao = gl.createVertexArray(); gl.bindVertexArray(quad.vao);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(quad.p, 'a'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }
}

function makeFbo(w, h) {
  if (!quad) return null;
  if (fbo) { gl.deleteFramebuffer(fbo.f); gl.deleteTexture(fbo.t); }
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (!ok) { quad = null; return null; }
  return { f, t };
}

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const c of [canvas, overlay]) { c.width = Math.round(innerWidth * dpr); c.height = Math.round(innerHeight * dpr); }
  fbo = makeFbo(canvas.width, canvas.height);
  const tl = $('timeline'); tl.width = Math.round(tl.clientWidth * dpr); tl.height = Math.round(tl.clientHeight * dpr);
  dirty = true;
}

const scalePx = () => S.view.zoom * Math.min(innerWidth, innerHeight) * .92;   // CSS px per world unit
const toScreen = (x, y) => [(x - S.view.cx) * scalePx() + innerWidth / 2, (y - S.view.cy) * scalePx() + innerHeight / 2];
const toWorld = (sx, sy) => [(sx - innerWidth / 2) / scalePx() + S.view.cx, (sy - innerHeight / 2) / scalePx() + S.view.cy];

function drawPoints() {
  const [start, end] = drawRange();
  const dpr = canvas.width / innerWidth, s = scalePx();
  const shown = Math.max(1, windowAt(M.offsets));     // continuous, so exposure changes smoothly
  // Sparse periods: larger dots when equalizing, so a few hundred papers remain visible.
  const sparse = S.eq ? Math.min(3, Math.max(1, Math.pow(200000 / shown, .25))) : 1;
  const size = Math.max(1, dpr * Math.min(7, sparse * (.9 + .55 * Math.log2(Math.max(1, S.view.zoom)))));
  // Exposure: aim for a similar look whatever the number of dots on screen (unless equalize is off).
  const reference = S.eq ? shown : M.n;
  const density = reference / (Math.pow(S.view.zoom, 2) * .3 * canvas.width * canvas.height) * size * size;
  const k = Math.min(1.6, .9 / Math.pow(Math.max(density, 1e-6), .55));
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo ? fbo.f : null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, fbo ? 0 : 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(prog.p); gl.bindVertexArray(prog.vao);
  gl.uniform2f(prog.u.u_center, S.view.cx, S.view.cy);
  gl.uniform2f(prog.u.u_scale, 2 * s / innerWidth, 2 * s / innerHeight);
  gl.uniform1f(prog.u.u_size, size);
  gl.uniform1f(prog.u.u_t, S.t);
  gl.uniform1f(prog.u.u_window, win());
  gl.uniform1f(prog.u.u_mode, S.show === 'growth' ? 2 : S.colour === 'age' ? 1 : 0);
  gl.uniform1f(prog.u.u_focus, S.focus);
  gl.uniform3fv(prog.u.u_pal, prog.pal);
  if (!fbo) { gl.blendFunc(gl.SRC_ALPHA, gl.ONE); }
  gl.drawArrays(gl.POINTS, start, end - start);
  gl.disable(gl.BLEND);
  if (fbo) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(quad.p); gl.bindVertexArray(quad.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, fbo.t);
    gl.uniform1i(quad.u.t, 0); gl.uniform1f(quad.u.u_k, k);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

// ---------- overlay: labels, selection, search ----------
function drawOverlay() {
  const dpr = overlay.width / innerWidth;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, innerWidth, innerHeight);
  if (S.show === 'growth') drawGrowth();
  if (S.show === 'terr') drawTerritories();
  drawLabels();
  if (window.drawWalk) drawWalk();                    // citation walk overlay (walk.js)
  const ring = (i, r, colour, width = 1.5) => {
    const [x, y] = toScreen(XY[2 * i], XY[2 * i + 1]);
    octx.beginPath(); octx.arc(x, y, r, 0, 7); octx.strokeStyle = colour; octx.lineWidth = width; octx.stroke();
    return [x, y];
  };
  if (S.search) {
    const [qx, qy] = toScreen(S.search.x, S.search.y);
    S.search.results.slice(0, 10).forEach((r) => ring(r.i, 5, 'rgba(140,200,255,.85)'));
    octx.strokeStyle = '#8cc8ff'; octx.lineWidth = 2;
    octx.beginPath(); octx.arc(qx, qy, 11, 0, 7); octx.moveTo(qx - 17, qy); octx.lineTo(qx + 17, qy);
    octx.moveTo(qx, qy - 17); octx.lineTo(qx, qy + 17); octx.stroke();
    octx.font = '600 13px system-ui'; octx.fillStyle = '#8cc8ff'; octx.fillText('your query', qx + 14, qy - 12);
  }
  if (S.raceHover >= 0 && M.subfields[S.raceHover].anchor) {
    const sf = M.subfields[S.raceHover], [hx, hy] = toScreen(sf.anchor.x, sf.anchor.y);
    octx.strokeStyle = '#fff'; octx.lineWidth = 2.5;
    octx.beginPath(); octx.arc(hx, hy, 26, 0, 7); octx.stroke();
    octx.font = '28px system-ui'; octx.textAlign = 'center'; octx.textBaseline = 'middle';
    octx.fillText(iconFor(sf.name, sf.field), hx, hy);
    octx.font = '600 13px system-ui'; octx.lineWidth = 4; octx.strokeStyle = '#000';
    octx.strokeText(sf.name, hx, hy + 42); octx.fillStyle = '#fff'; octx.fillText(sf.name, hx, hy + 42);
  }
  if (S.selected >= 0) {
    const [sx, sy] = toScreen(XY[2 * S.selected], XY[2 * S.selected + 1]);
    if (S.similar) {
      for (const n of S.similar.true) {
        const [x, y] = toScreen(XY[2 * n.i], XY[2 * n.i + 1]);
        octx.beginPath(); octx.moveTo(sx, sy); octx.lineTo(x, y);
        octx.strokeStyle = 'rgba(255,209,102,.35)'; octx.lineWidth = 1; octx.stroke();
        ring(n.i, 5, '#ffd166');
      }
    }
    ring(S.selected, 9, '#fff', 2);
  }
}

function drawLabels() {
  const z = S.view.zoom, boxes = [];
  const items = [];
  // A label shows once its field is a real share of the dots shown (not a handful of
  // classifier hits), so a field's name appears when the field becomes visible.
  const shown = Math.max(1, windowAt(M.offsets));
  const rules = { f: { min: 12, share: .005 }, s: { min: 6, share: .0012 } };
  const addAll = (list, prefix, kind) => list.forEach((l, i) => {
    if (!l.anchor || l.name === 'Unclassified') return;
    if (S.focus >= 0 && (kind === 'f' ? i !== S.focus : l.field !== M.fields[S.focus].name)) return;
    const c = windowAt(prefix[i]), share = c / shown, r = rules[kind];
    if (c < r.min || share < r.share) return;
    items.push({ l, c, kind, a: Math.min(1, .35 + .65 * (share - r.share) / (3 * r.share)) });
  });
  if (z < 5) addAll(M.fields, fieldPrefix, 'f');
  if (z >= 1.8) addAll(M.subfields, subPrefix, 's');
  items.sort((a, b) => (a.kind === b.kind ? b.c - a.c : (a.kind === 'f' ? -1 : 1)));
  octx.textAlign = 'center'; octx.textBaseline = 'middle'; octx.lineJoin = 'round';
  for (const it of items) {
    let anchor = it.l.anchor, fade = 1;
    if (S.show === 'terr' && it.kind === 'f') {            // field names sit inside their current territory
      anchor = terrAnchor(M.fields.indexOf(it.l));
      if (!anchor) continue;
      fade = anchor.a;
    }
    const [x, y] = toScreen(anchor.x, anchor.y);
    if (x < -100 || y < -30 || x > innerWidth + 100 || y > innerHeight + 30) continue;
    const big = it.kind === 'f';
    octx.font = big ? '650 14px system-ui' : '500 11.5px system-ui';
    const text = big ? it.l.name.toUpperCase() : it.l.name;
    const w = octx.measureText(text).width + 8, h = big ? 18 : 15;
    const box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
    if (boxes.some((b) => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]))) continue;
    boxes.push(box);
    octx.globalAlpha = it.a * fade * (typeof WALK !== 'undefined' && WALK.data ? .4 : 1);   // a walk is on top
    octx.strokeStyle = 'rgba(0,0,0,.85)'; octx.lineWidth = 3.5; octx.strokeText(text, x, y);
    octx.fillStyle = big ? '#f3f5fa' : '#cdd5e4'; octx.fillText(text, x, y);
  }
  octx.globalAlpha = 1;
}

// ---------- timeline ----------
function drawTimeline() {
  const c = $('timeline'), ctx = c.getContext('2d'), dpr = c.width / c.clientWidth;
  const W = c.clientWidth, H = c.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const ys = M.years, n = ys.length, bw = W / n;
  const lmax = Math.log10(Math.max(...M.eligible_per_year)), lmin = 1;   // 10 .. max papers
  const [a, b] = yearSpan();
  for (let i = 0; i < n; i++) {
    const c = M.eligible_per_year[i];
    const v = c > 10 ? (Math.log10(c) - lmin) / (lmax - lmin) : 0, hh = Math.max(c ? 1 : 0, v * (H - 16));
    const inside = ys[i] >= a && ys[i] <= b;
    const base = S.show === 'growth' && ys[i] >= a - S.lag && ys[i] <= b - S.lag;
    ctx.fillStyle = ys[i] === S.year ? '#ffd166' : inside ? '#7f95c4' : base ? '#58667f' : '#2e3850';
    ctx.fillRect(i * bw, H - 14 - hh, Math.max(1, bw - .5), hh);
  }
  const head = Math.max(0, Math.min(W, (S.t - ys[0] + 1) * bw));       // playhead at the exact moment
  ctx.fillStyle = '#ffd166'; ctx.fillRect(head - 1, 12, 2, H - 26);
  ctx.fillStyle = '#97a1b5'; ctx.font = '11px system-ui'; ctx.textBaseline = 'bottom';
  for (let y = Math.ceil(ys[0] / 25) * 25; y <= ys[n - 1]; y += 25) {
    const x = (y - ys[0] + .5) * bw; ctx.fillRect(x, H - 13, 1, 3);
    if (x > 16 && x < W - 16) ctx.fillText(String(y), x - 13, H);
  }
  ctx.textBaseline = 'top'; ctx.fillText('publications per year in the corpus (log scale: each step up = 10× more)' +
    (S.show === 'growth' ? ` · grey: the period compared against (${S.lag} years earlier)` : ''), 4, 0);
}

function updateText() {
  $('year').textContent = S.year;
  const [start, end] = range(), [a, b] = yearSpan();
  let corpus = 0;
  for (let y = a; y <= b; y++) corpus += M.eligible_per_year[yi(y)] || 0;
  const span = a === b ? `${b}` : `${a}–${b}`;
  $('count').textContent = `${(end - start).toLocaleString()} dots for ${span} · ${corpus.toLocaleString()} papers in the corpus` +
    (b >= 2026 ? ' · 2026 is incomplete' : '');
}

// ---------- interaction ----------
let pointers = new Map(), dragStart = null, moved = false;
function pick(sx, sy, radius = 8) {
  const [wx, wy] = toWorld(sx, sy), r = radius / scalePx();
  const [start, end] = range();
  let best = -1, bd = r * r;
  const g0x = Math.max(0, ((wx - r) * GRID) | 0), g1x = Math.min(GRID - 1, ((wx + r) * GRID) | 0);
  const g0y = Math.max(0, ((wy - r) * GRID) | 0), g1y = Math.min(GRID - 1, ((wy + r) * GRID) | 0);
  for (let gy = g0y; gy <= g1y; gy++) for (let gx = g0x; gx <= g1x; gx++) {
    const c = gy * GRID + gx;
    for (let k = grid.start[c]; k < grid.start[c + 1]; k++) {
      const i = grid.idx[k];
      if (!visible(i, start, end)) continue;
      const dx = XY[2 * i] - wx, dy = XY[2 * i + 1] - wy, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
  }
  return best;
}

function zoomAt(sx, sy, factor) {
  const [wx, wy] = toWorld(sx, sy);
  S.view.zoom = Math.max(.6, Math.min(400, S.view.zoom * factor));
  const [nx, ny] = toWorld(sx, sy);
  S.view.cx += wx - nx; S.view.cy += wy - ny; changed();
}

function flyTo(x, y, zoom) {
  const from = { ...S.view }, t0 = performance.now(), dur = 700;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    const z = Math.exp(Math.log(from.zoom) + (Math.log(zoom) - Math.log(from.zoom)) * e);
    S.view = { cx: from.cx + (x - from.cx) * e, cy: from.cy + (y - from.cy) * e, zoom: z }; changed();
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setupUI() {
  window.addEventListener('resize', resize);
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * .0015)); }, { passive: false });
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
    dragStart = [e.clientX, e.clientY]; moved = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) { hover(e); return; }
    const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (Math.hypot(e.clientX - dragStart[0], e.clientY - dragStart[1]) > 4) moved = true;
    if (pointers.size === 1) {
      S.view.cx -= (e.clientX - prev[0]) / scalePx(); S.view.cy -= (e.clientY - prev[1]) / scalePx(); changed();
    } else if (pointers.size === 2) {
      const [p, q] = [...pointers.values()];
      const other = p === pointers.get(e.pointerId) ? q : p;
      const d0 = Math.hypot(prev[0] - other[0], prev[1] - other[1]), d1 = Math.hypot(e.clientX - other[0], e.clientY - other[1]);
      if (d0 > 0) zoomAt((e.clientX + other[0]) / 2, (e.clientY + other[1]) / 2, d1 / d0);
    }
  });
  const up = (e) => {
    pointers.delete(e.pointerId);
    if (!moved && pointers.size === 0) {
      if (window.walkClick && walkClick(e)) return;      // a click on a walk marker
      const i = pick(e.clientX, e.clientY, e.pointerType === 'touch' ? 16 : 8); select(i);
    }
  };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', (e) => pointers.delete(e.pointerId));
  canvas.addEventListener('pointerleave', () => { $('tip').style.display = 'none'; });

  $('play').onclick = togglePlay;
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowLeft') setYear(Math.round(S.target) - 1);
    if (e.key === 'ArrowRight') setYear(Math.round(S.target) + 1);
    if (e.key === 'Escape') select(-1);
  });
  $('mode').value = String(S.window); $('colour').value = S.colour; $('eq').classList.toggle('on', S.eq);
  $('mode').onchange = () => { S.window = +$('mode').value; changed(); };
  $('show').value = S.show; $('lag').value = String(S.lag);
  $('show').onchange = () => {
    S.show = $('show').value;
    if (S.playing) { togglePlay(); togglePlay(); }       // restart at the new speed
    if (S.show !== 'dots' && !S.window) { S.window = 5; $('mode').value = '5'; }
    changed();
  };
  $('lag').onchange = () => { S.lag = +$('lag').value; changed(); };
  $('colour').onchange = () => { S.colour = $('colour').value; changed(); };
  $('eq').onclick = () => { S.eq = !S.eq; $('eq').classList.toggle('on', S.eq); changed(); };
  $('legendBtn').onclick = () => $('legend').classList.toggle(innerWidth <= 760 ? 'show' : 'hidden');
  window.addEventListener('hashchange', () => {     // typed or pasted links, back/forward
    readHash(); $('mode').value = String(S.window); $('colour').value = S.colour;
    $('show').value = S.show; $('lag').value = String(S.lag);
    $('eq').classList.toggle('on', S.eq); if (S.selected >= 0) select(S.selected); else changed();
  });
  $('aboutBtn').onclick = showAbout;
  $('search').onsubmit = (e) => { e.preventDefault(); search($('q').value); };
  const tl = $('timeline');
  const scrub = (e) => { const r = tl.getBoundingClientRect();
    setYear(M.years[0] + Math.floor((e.clientX - r.left) / r.width * M.years.length)); };
  tl.addEventListener('pointerdown', (e) => { tl.setPointerCapture(e.pointerId); scrub(e); });
  tl.addEventListener('pointermove', (e) => { if (e.buttons) scrub(e); });
  buildLegend();
  const placeLegend = () => { $('legend').style.top = ($('top').getBoundingClientRect().bottom + 10) + 'px'; };
  placeLegend(); window.addEventListener('resize', placeLegend);
  if (S.selected >= 0) select(S.selected, false);
}

function hover(e) {
  if (window.walkHover && walkHover(e)) return;        // walk markers take precedence over dots
  const i = pick(e.clientX, e.clientY), tip = $('tip');
  if (i < 0) { tip.style.display = 'none'; canvas.style.cursor = 'grab'; return; }
  canvas.style.cursor = 'pointer';
  const f = M.fields[FIELD[i]].name, s = M.subfields[SUB[i]].name;
  tip.innerHTML = `<b>${YEAR[i]}</b> · ${esc(f)}${s && s !== 'Unclassified' ? ' › ' + esc(s) : ''}<br><span class="small">click for details</span>`;
  tip.style.display = 'block';
  tip.style.left = Math.min(innerWidth - 350, e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
}

// Changing the year glides there (see frame); playback advances time continuously.
function setYear(y) {
  S.target = Math.max(M.years[0], Math.min(M.years[M.years.length - 1], y)); changed();
}
// with the size race on screen, play slower so it can be read (ms per year)
const playStep = () => (S.show === 'dots' ? 220 : 450);
function togglePlay() {
  S.playing = !S.playing; $('play').textContent = S.playing ? '❚❚' : '▶';
  const last = M.years[M.years.length - 1];
  if (S.playing && S.t >= last - 1e-6) S.t = S.target = 1900;
  if (!S.playing) S.target = Math.min(last, Math.ceil(S.t - 1e-6));   // settle on the year being filled
  changed();
}

function buildLegend() {
  const byDomain = {};
  M.fields.forEach((f, i) => { (byDomain[f.domain || 'Other'] ||= []).push([f, i]); });
  const order = ['Physical Sciences', 'Life Sciences', 'Health Sciences', 'Social Sciences', 'Other'];
  let html = '';
  for (const d of order) {
    if (!byDomain[d]) continue;
    html += `<div class="dom">${esc(d === 'Other' ? 'No field label' : d)}</div>`;
    for (const [f, i] of byDomain[d]) html += `<div class="f" data-i="${i}"><span class="sw" style="background:${f.colour}"></span>${esc(f.name)}</div>`;
  }
  html += '<div class="hint">Click a field to highlight it; click again to show all. Colours are OpenAlex topic labels.</div>';
  $('legend').innerHTML = html;
  $('legend').querySelectorAll('.f').forEach((el) => el.onclick = () => {
    const i = +el.dataset.i; S.focus = S.focus === i ? -1 : i;
    $('legend').querySelectorAll('.f').forEach((x) => x.classList.toggle('dim', S.focus >= 0 && +x.dataset.i !== S.focus));
    changed();
  });
}

// ---------- side panel ----------
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function side(html) { $('side').style.display = html ? 'block' : 'none'; $('side').innerHTML = html ? '<button class="close" onclick="closeSide()">×</button>' + html : ''; }
window.closeSide = () => { S.selected = -1; S.similar = null; S.search = null; side(''); changed(); };

async function select(i, fly = false) {
  S.selected = i; S.similar = null; changed();
  if (i < 0) { if (!S.search) side(''); return; }
  const [p, sim] = await Promise.all([fetch('/api/paper?i=' + i).then((r) => r.json()), fetch('/api/similar?i=' + i).then((r) => r.json())]);
  if (S.selected !== i) return;
  S.similar = sim; changed();
  if (fly) flyTo(p.x, p.y, Math.max(S.view.zoom, 6));
  const [start, end] = range();
  const list = sim.true.map((n) => `<li data-i="${n.i}" class="${visible(n.i, start, end) ? '' : 'off'}">${esc(n.title)} <span class="y">${n.year} · ${n.sim.toFixed(2)}</span></li>`).join('');
  side(`<h2>${esc(p.title || '(no title)')}</h2>
    <div class="meta">${p.year} · ${esc(p.type || '')}${p.venue ? ' · ' + esc(p.venue) : ''}<br>
    ${esc(p.field || 'No field label')}${p.subfield ? ' › ' + esc(p.subfield) : ''}${p.topic ? ' › ' + esc(p.topic) : ''}<br>
    ${p.citations ?? '?'} citations (OpenAlex, today) · <a href="${p.openalex}" target="_blank" rel="noopener">OpenAlex</a></div>
    ${window.startWalk ? `<button class="walkbtn" onclick="startWalk('W${p.work}', {reset: true})">Citation walk from this paper →</button>` : ''}
    <div class="score"><b>${sim.kept} of ${sim.k}</b> of this paper's most similar papers are also among its ${sim.k} nearest dots.
      <div class="small">Similarity measured on the full vectors, among the dots of all years. Gold rings show where they are.
      For comparison, the map's average is ${((M.quality.methods[M.method + '+knn'] || {}).knn10_out * 10).toFixed(1)} of 10, and about
      ${Math.round(((M.quality.methods[M.method + '+knn'] || {}).knn10_in_100_out || 0) * 10)} of 10 are within its 100 nearest dots.</div></div>
    <b>Most similar papers</b> <span class="small">(year · similarity; faded = outside the selected years)</span>
    <ol>${list}</ol>`);
  $('side').querySelectorAll('li').forEach((el) => el.onclick = () => select(+el.dataset.i, true));
}

async function search(text) {
  text = text.trim(); if (!text) return;
  if (window.isWalkQuery && isWalkQuery(text)) { startWalk(text, { reset: true }); return; }   // W… or DOI: walk
  side('<div class="small">Placing your query…</div>');
  const r = await fetch('/api/search?q=' + encodeURIComponent(text)).then((x) => x.json());
  if (r.error) { side(`<div>${esc(r.error)}</div>`); return; }
  S.search = r; S.selected = -1; S.similar = null;
  const list = r.results.map((n) => `<li data-i="${n.i}">${esc(n.title)} <span class="y">${n.year} · ${n.sim.toFixed(2)}</span></li>`).join('');
  side(`<h2>“${esc(r.query)}”</h2>
    <div class="meta">The blue cross marks where the frozen map puts this text. Blue rings: its ten most similar dots (any year).</div>
    <ol>${list}</ol>`);
  $('side').querySelectorAll('li').forEach((el) => el.onclick = () => { S.search = r; select(+el.dataset.i, true); });
  flyTo(r.x, r.y, Math.max(S.view.zoom, 5));
}

function showAbout() {
  const q = M.quality.methods, chosen = q[M.method + '+knn'] || {}, pca = q.pca || {};
  const pct = (v) => v == null ? '?' : Math.round(v * 100) + '%';
  const th = M.quality.time_holdout || {};
  side(`<h2>About this map</h2><div id="aboutText">
  <p><b>What a dot is.</b> One English publication with an abstract, from OpenAlex (articles, reviews, conference
  papers, preprints, books and chapters, theses, reports, letters, editorials). A fixed 1 in 100 of them is drawn
  (${M.n.toLocaleString()} dots), so the brightness of a period follows its real volume.
  Datasets and other non-publication records are left out: in 2024–2026 they were most of OpenAlex's English
  “works” (mostly automatic instrument logs) and would have shown a fake explosion.</p>
  <p><b>Why dots are where they are.</b> A language model (DenseOn) reads each title and abstract and produces a list of
  768 numbers; similar papers get similar numbers. The map squeezes those 768 dimensions into 2 with ${M.method.toUpperCase()},
  fitted once on ${M.references.toLocaleString()} reference papers chosen to give older decades more room than their volume.</p>
  <p><b>The map never moves.</b> The year controls only decide which dots are lit. A paper's position is fixed by its own text.</p>
  <p><b>How faithful is it?</b> For papers the map was not fitted on, ${pct(chosen.knn10_out)} of each paper's 10 most similar
  papers stay among its 10 nearest dots (the previous straight-projection method: ${pct(pca.knn10_out)}).
  The rest are elsewhere: a 2-D picture cannot keep everything. Click any dot to see its own score.</p>
  ${th.fit_without_2020_plus != null ? `<p><b>New science on a frozen map.</b> When the map was fitted without any paper from 2020 or later, 2020s papers still kept
  ${pct(th.fit_without_2020_plus)} of their neighbours (vs ${pct(th.fit_with_all_years)} with them).</p>` : ''}
  <p><b>Read growth with care.</b> This covers only English papers that have an abstract in OpenAlex. Older papers lack abstracts far
  more often, and indexing coverage changed over time, so part of the “growth” is better records, not only more science.
  The map is <i>retrospective</i>: a 1950 paper is placed by how today's model reads it.</p>
  <p><b>Growth view.</b> Red and blue compare a place's <i>share</i> of all research in the selected years with the same
  years a decade (or 5 / 20 years) earlier. Science as a whole grows, so shares are what reveal shifts. Only differences
  clearly larger than the noise of a 1 % sample are coloured; grey means no clear change.</p>
  <p><b>Field territories.</b> Each place takes the colour of the field with the most papers there in the selected
  years, like countries on a map. Colour is pale where two fields are close to a tie, and nothing is drawn where there
  are too few papers to tell. As the years play, territories grow, shrink and change hands. Field names sit at the
  heart of each field's largest current territory.</p>
  <p><b>Colours</b> are OpenAlex's own topic labels (a classifier); about a fifth of papers have none and are grey.</p>
  <p class="small">Build ${esc(M.id)} · embedding plan ${esc(M.embedding_plan.slice(0, 12))} · placement k=${M.placement.k}</p></div>`);
}

// ---------- state, URL, frame ----------
function changed() { dirty = true; }
let hashTimer = null;
function writeHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const v = S.view;
    history.replaceState(null, '', `#y=${S.year}&w=${S.window}&c=${S.colour}&e=${S.eq ? 1 : 0}&s=${S.show}&l=${S.lag}&v=${v.cx.toFixed(4)},${v.cy.toFixed(4)},${v.zoom.toFixed(2)}` + (S.selected >= 0 ? `&p=${S.selected}` : '') + (window.walkHashPart ? walkHashPart() : ''));
  }, 250);
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has('y')) S.year = +h.get('y');
  if (h.has('w')) S.window = +h.get('w');
  if (h.has('c')) S.colour = h.get('c');
  if (h.has('e')) S.eq = h.get('e') === '1';
  if (h.has('s')) S.show = ['growth', 'dots', 'terr'].includes(h.get('s')) ? h.get('s') : 'terr';
  if (h.has('l') && [5, 10, 20].includes(+h.get('l'))) S.lag = +h.get('l');
  if (h.has('v')) { const [cx, cy, z] = h.get('v').split(',').map(Number); if ([cx, cy, z].every(isFinite)) S.view = { cx, cy, zoom: z }; }
  if (h.has('p')) S.selected = +h.get('p');
  S.year = Math.max(M.years[0], Math.min(M.years[M.years.length - 1], S.year));
  S.t = S.target = S.year;
}

let lastFrame = 0;
function frame(now) {
  const dt = lastFrame ? Math.min(.1, (now - lastFrame) / 1000) : 0, last = M.years[M.years.length - 1];
  lastFrame = now;
  if (S.playing) {
    S.t = Math.min(last, S.t + dt * 1000 / playStep()); S.target = S.t; dirty = true;
    if (S.t >= last) togglePlay();
  } else if (Math.abs(S.target - S.t) > 1e-4) {       // glide: eased, but at least 2 years/second
    const d = S.target - S.t;
    S.t += Math.sign(d) * Math.max(Math.abs(d) * (1 - Math.exp(-dt * 8)), Math.min(Math.abs(d), dt * 2));
    if (Math.abs(S.target - S.t) < 1e-3) S.t = S.target;
    dirty = true;
  }
  const y = Math.max(M.years[0], Math.min(last, Math.ceil(S.t - 1e-6)));
  if (y !== S.year) { S.year = y; }
  if (dirty) {
    dirty = false;
    drawPoints(); drawOverlay(); drawTimeline(); updateText(); writeHash(); updateRace();
  }
  requestAnimationFrame(frame);
}

load().catch((e) => { $('loading').textContent = 'Could not load the atlas: ' + e.message; console.error(e); });


// ---------- animated layers: territories and growth as per-year keyframes ----------
// Each layer is computed for whole years (the window ending at that year) by layers.js, in a Web
// Worker so the animation never stalls, and cross-faded while time moves from one year to the
// next. Without a worker, one keyframe is computed per frame on the page.
const keyframes = new Map(), comp = {};
let worker = null, busy = false, wanted = [], terrNow = null, layerArgs = null;

function layerSetup() {
  layerArgs = { NF: M.fields.length, uncl: M.fields.findIndex((f) => f.name === 'Unclassified'),
                rgb: M.fields.map((f) => [1, 3, 5].map((o) => parseInt(f.colour.substr(o, 2), 16))) };
  try {
    worker = new Worker('layers-worker.js');
    worker.onmessage = (ev) => { store(ev.data); busy = false; pump(); changed(); };
    worker.onerror = (e) => { console.warn('layer worker failed; computing on the page', e.message); worker = null; busy = false; changed(); };
    worker.postMessage({ type: 'init', XY, FIELD, ...layerArgs });
  } catch (e) { worker = null; }
}
function runJob(j) {
  return j.kind === 'terr' ? computeTerritory(XY, FIELD, j.s, j.e, layerArgs.NF, layerArgs.uncl, layerArgs.rgb)
                           : computeGrowth(XY, j.a0, j.a1, j.b0, j.b1);
}
function store(r) {
  let canvas = null;
  if (r.rgba) {
    canvas = document.createElement('canvas'); canvas.width = canvas.height = r.size;
    canvas.getContext('2d').putImageData(new ImageData(r.rgba, r.size, r.size), 0, 0);
  }
  keyframes.set(r.key, { canvas, anchor: r.anchor, used: performance.now() });
  if (keyframes.size > 60) keyframes.delete([...keyframes].sort((p, q) => p[1].used - q[1].used)[0][0]);
}
function pump() {
  if (busy) return;
  const job = wanted.find((j) => !keyframes.has(j.key));
  if (!job) return;
  if (worker) { busy = true; worker.postMessage(job); return; }
  const r = runJob(job);
  store({ key: job.key, rgba: r ? r.rgba : null, size: r ? r.size : 0, anchor: r ? r.anchor : null });
  changed();
}
function terrJob(Y) { const [s, e] = rangeFor(Y, win()); return { kind: 'terr', key: `t|${s}|${e}`, s, e }; }
function growthJob(Y) {
  const w = win(), [a0, a1] = rangeFor(Y, w), [b0, b1] = rangeFor(Y - S.lag, w);
  return { kind: 'growth', key: `g|${a0}|${a1}|${b0}|${b1}`, a0, a1, b0, b1 };
}
// keyframes of the two years around the current moment, and how far we are between them
function layerFrames(make) {
  const last = M.years[M.years.length - 1], Y1 = S.year, Y0 = Y1 - 1, f = Math.max(0, Math.min(1, S.t - Y0));
  const j0 = make(Y0), j1 = make(Y1);
  wanted = [j1, j0];
  if (S.playing) wanted.push(make(Math.min(last, Y1 + 1)), make(Math.min(last, Y1 + 2)));   // prefetch
  pump();
  const get = (j) => { const k = keyframes.get(j.key); if (k) k.used = performance.now(); return k || null; };
  return { k0: get(j0), k1: get(j1), f };
}
function drawLayer(kind, make, size) {
  const fr = layerFrames(make), { k0, k1, f } = fr;
  let c = comp[kind];
  if (!c) { c = comp[kind] = document.createElement('canvas'); c.width = c.height = size; }
  if (k0 || k1) {                       // exact linear mix of the two keyframes ('lighter' adds)
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, size, size);
    ctx.globalCompositeOperation = 'lighter';
    if (k0 && k0.canvas && (!k1 || f < 1)) { ctx.globalAlpha = k1 ? 1 - f : 1; ctx.drawImage(k0.canvas, 0, 0); }
    if (k1 && k1.canvas) { ctx.globalAlpha = k0 ? f : 1; ctx.drawImage(k1.canvas, 0, 0); }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  }                                     // neither ready yet: keep showing the last mix
  const [x0, y0] = toScreen(0, 0), [x1, y1] = toScreen(1, 1);
  octx.imageSmoothingEnabled = true;
  octx.drawImage(c, x0, y0, x1 - x0, y1 - y0);
  return fr;
}

function drawGrowth() {
  const key = $('growthKey');
  key.style.display = 'block'; key.style.bottom = ($('bottom').offsetHeight + 22) + 'px';
  const j = growthJob(S.year), ok = j.a1 - j.a0 >= 200 && j.b1 - j.b0 >= 200;
  $('growthNote').textContent = ok ? `${yearSpan().join('–')} vs ${yearSpan()[0] - S.lag}–${S.year - S.lag}. Grey = no clear change.`
                                   : 'Not enough papers in one of the two periods to compare';
  drawLayer('growth', growthJob, HG);
}

function drawTerritories() {
  octx.globalAlpha = S.view.zoom > 6 ? Math.max(.25, 1 - (S.view.zoom - 6) / 10) : 1;   // let dots take over close up
  terrNow = drawLayer('terr', terrJob, TG);
  octx.globalAlpha = 1;
}
// a field label's position: glides between the two keyframes' anchors, or cross-fades if it jumps
function terrAnchor(fi) {
  if (!terrNow) return null;
  const { k0, k1, f } = terrNow;
  const a0 = k0 && k0.anchor ? k0.anchor[fi] : null, a1 = k1 && k1.anchor ? k1.anchor[fi] : null;
  if (a0 && a1) {
    if (Math.hypot(a1.x - a0.x, a1.y - a0.y) < .15) return { x: a0.x + (a1.x - a0.x) * f, y: a0.y + (a1.y - a0.y) * f, a: 1 };
    return f < .5 ? { ...a0, a: 1 - 2 * f } : { ...a1, a: 2 * f - 1 };
  }
  if (a1) return { ...a1, a: k0 ? f : 1 };
  if (a0) return { ...a0, a: k1 ? 1 - f : 1 };
  return null;
}

// ---------- size race: the 16 largest subfields by share of all publications ----------
// Bar length = share of all publications in the selected years (dots of that subfield / all dots,
// unlabelled papers included in the total). Bars are scaled to the current leader, as in a bar
// chart race; gridlines at whole percents move with the scale so years stay comparable.
// Rows are persistent DOM elements: they slide (rank) and stretch (size) between years.
// Measured on this build: 0.5–0.85 subfields enter the top 16 per year (1900–2026), so no
// smoothing of membership is needed. The chip shows the change in share vs. the comparison period
// (estimate shrunk towards "no change", prior 10 dots, hidden below 40 dots).
const RACE_N = 16, ROW_H = 25, RACE_PRIOR = 10, TICKS = 8;
const raceRows = new Map(), changeCache = new Map();
let raceKey = '', raceTicks = null;

function shareChange(Y) {        // per subfield: share now, change in share {r} (NaN = too few dots)
  const w = win(), key = `${Y}|${w}|${S.lag}`;
  if (changeCache.has(key)) return changeCache.get(key);
  const [a0, a1] = rangeFor(Y, w), [b0, b1] = rangeFor(Y - S.lag, w), A = a1 - a0, B = b1 - b0;
  let out = null;
  if (A >= 200) {
    const n = M.subfields.length, r = new Float64Array(n).fill(NaN), share = new Float64Array(n), av = new Float64Array(n);
    const count = (p, y) => { const s = Math.max(M.years[0], y - w + 1); return y < M.years[0] ? 0 : p[yi(y) + 1] - p[yi(s)]; };
    M.subfields.forEach((sf, i) => {
      if (sf.name === 'Unclassified') return;
      const a = count(subPrefix[i], Y);
      av[i] = a; share[i] = a / A;
      if (B >= 200) {
        const e = count(subPrefix[i], Y - S.lag) * A / B;
        if (a + e >= 40) r[i] = Math.log2((a + RACE_PRIOR) / (e + RACE_PRIOR));
      }
    });
    out = { r, share, a: av, A };
  }
  changeCache.set(key, out);
  return out;
}

function raceRow(i) {
  const sf = M.subfields[i], f = M.fields.find((x) => x.name === sf.field), col = f ? f.colour : '#888';
  const el = document.createElement('div');
  el.className = 'rrow';
  el.innerHTML = `<span class="ico" style="background:${col}33;border-color:${col}">${iconFor(sf.name, sf.field)}</span>
    <span class="track"><span class="bar" style="background:${col}"></span><span class="nm">${esc(sf.name)}</span></span>
    <span class="val"></span><span class="chg"></span>`;
  el.onclick = () => { if (sf.anchor) flyTo(sf.anchor.x, sf.anchor.y, Math.max(S.view.zoom, 3)); };
  el.onmouseenter = () => { S.raceHover = i; changed(); };
  el.onmouseleave = () => { if (S.raceHover === i) { S.raceHover = -1; changed(); } };
  $('raceRows').appendChild(el);
  const row = { el, bar: el.querySelector('.bar'), val: el.querySelector('.val'), chg: el.querySelector('.chg'), gone: null };
  raceRows.set(i, row);
  return row;
}

function setTicks(scale) {        // gridlines at 1 %, 2 %, … of all publications
  if (!raceTicks) {
    raceTicks = [];
    for (let p = 1; p <= TICKS; p++) {
      const line = document.createElement('div'); line.className = 'tick';
      line.innerHTML = `<span>${p}%</span>`;
      $('raceGrid').appendChild(line); raceTicks.push(line);
    }
  }
  raceTicks.forEach((t, k) => {
    const x = (k + 1) / 100 / scale;
    t.style.left = (100 * Math.min(x, 1.2)) + '%';
    t.style.opacity = x <= 1.001 ? 1 : 0;
  });
}

function updateRace() {
  const race = $('race');
  if (S.show !== 'growth') $('growthKey').style.display = 'none';
  const visibleRace = S.show !== 'dots' && S.selected < 0 && !S.search;
  race.style.display = visibleRace ? 'block' : 'none';
  if (!visibleRace) { if (S.raceHover >= 0) { S.raceHover = -1; dirty = true; } return; }
  const key = `${S.year}|${win()}|${S.lag}`;
  if (key === raceKey) return;
  raceKey = key;
  race.style.setProperty('--race-dur', S.playing ? (playStep() / 1000) + 's' : '0.55s');
  race.style.setProperty('--race-ease', S.playing ? 'linear' : 'cubic-bezier(.2,.8,.2,1)');
  const [s1, e1] = yearSpan();
  let corpus = 0;
  for (let y = s1; y <= e1; y++) corpus += M.eligible_per_year[yi(y)] || 0;
  $('raceSub').textContent = `${s1 === e1 ? e1 : s1 + '–' + e1} · ${corpus.toLocaleString()} publications`;
  $('raceChgHead').textContent = `vs ${S.lag}y`;

  const cur = shareChange(S.year);
  const shown = cur ? [...cur.share.keys()].filter((i) => cur.share[i] > 0)
    .sort((p, q) => cur.share[q] - cur.share[p]).slice(0, RACE_N) : [];
  $('raceEmpty').style.display = shown.length ? 'none' : 'block';
  $('raceRows').style.height = (shown.length ? RACE_N * ROW_H : 0) + 'px';
  const scale = shown.length ? cur.share[shown[0]] : 1;
  setTicks(scale);
  const target = new Map(shown.map((i, k) => [i, k * ROW_H]));
  for (const [i, y] of target) {
    let row = raceRows.get(i);
    if (!row || row.gone) {                       // (re-)enter at the bottom: fade in, bar grows
      if (!row) row = raceRow(i);
      row.el.classList.add('entering'); row.el.classList.remove('leaving');
      row.el.style.transform = `translateY(${(RACE_N - 1) * ROW_H}px)`;
      row.el.style.opacity = 0; row.bar.style.width = '0%';
      void row.el.offsetWidth;
      row.el.classList.remove('entering');
    }
    clearTimeout(row.gone); row.gone = null;
    const sh = cur.share[i], r = cur.r[i];
    row.el.style.transform = `translateY(${y}px)`;
    row.el.style.opacity = 1;
    row.bar.style.width = (100 * sh / scale) + '%';
    row.val.textContent = (100 * sh).toFixed(1) + '%';
    if (isNaN(r)) { row.chg.textContent = ''; row.chg.className = 'chg'; }
    else {
      const ratio = Math.pow(2, r);
      row.chg.textContent = Math.abs(r) < .2 ? '=' : ratio >= 1 ? `▲×${ratio < 10 ? ratio.toFixed(1) : ratio.toFixed(0)}` : `▼÷${(1 / ratio).toFixed(1)}`;
      row.chg.className = 'chg ' + (Math.abs(r) < .2 ? 'same' : r > 0 ? 'up' : 'down');
    }
    const est = Math.round(sh * corpus / 100) * 100;
    row.el.title = `${M.subfields[i].name} (${M.subfields[i].field})\n${(100 * sh).toFixed(2)} % of all publications in ${s1}–${e1}` +
      ` (≈ ${est.toLocaleString()} publications; ${cur.a[i]} dots)` +
      (isNaN(r) ? '' : `\nShare vs ${S.lag} years earlier: ${Math.pow(2, r) >= 1 ? '×' + Math.pow(2, r).toFixed(2) : '÷' + Math.pow(2, -r).toFixed(2)} (estimate)`) +
      `\nClick to fly there`;
  }
  for (const [i, row] of raceRows) {
    if (target.has(i) || row.gone) continue;
    row.el.classList.add('leaving');
    row.el.style.opacity = 0;
    row.el.style.transform = `translateY(${(RACE_N - 1) * ROW_H}px)`;
    row.gone = setTimeout(() => { row.el.remove(); raceRows.delete(i); if (S.raceHover === i) { S.raceHover = -1; changed(); } }, 500);
  }
}

window.atlasTerritoryChange = (y0, y1) => {      // test hook: share of firm places changing owner
  const T = (Y) => { const [s, e] = rangeFor(Y, win()); return computeTerritory(XY, FIELD, s, e, layerArgs.NF, layerArgs.uncl, layerArgs.rgb); };
  const t0 = T(y0), t1 = T(y1);
  let both = 0, diff = 0;
  for (let k = 0; k < t0.best.length; k++) if (t0.sure[k] && t1.sure[k]) { both++; if (t0.best[k] !== t1.best[k]) diff++; }
  return { cells: both, changed: diff / Math.max(1, both) };
};
