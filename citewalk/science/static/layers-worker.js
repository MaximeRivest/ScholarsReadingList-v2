'use strict';
// Background computation of map-layer keyframes, so the animation never stalls on them.
importScripts('layers.js');
let D = null;
onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'init') { D = m; return; }
  const r = m.kind === 'terr' ? computeTerritory(D.XY, D.FIELD, m.s, m.e, D.NF, D.uncl, D.rgb)
                              : computeGrowth(D.XY, m.a0, m.a1, m.b0, m.b1);
  postMessage({ key: m.key, rgba: r ? r.rgba : null, size: r ? r.size : 0, anchor: r ? r.anchor : null },
              r ? [r.rgba.buffer] : []);
};
