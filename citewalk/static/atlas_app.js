/* UI for the clean local-map experiment. Citation ranking is unchanged. */
'use strict';
const $=id=>document.getElementById(id);
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const compact=(s,n=75)=>String(s||'(untitled)').length>n?String(s).slice(0,n-1)+'…':String(s||'(untitled)');
const fmt=n=>Number.isFinite(n)&&n>=0?n.toLocaleString():'unknown';
let current=null,mapId=new URLSearchParams(location.search).get('atlas'),historyItems=[],selection=null,hits=[],requestNumber=0,controller=null;
let tab='map',yearCut=Infinity;
const plot=new AtlasPlot($('plot'),{
  hover:(n,e,count)=>{
    const t=$('tooltip');if(!n){t.style.display='none';return;}
    t.innerHTML=`<strong>${escapeHTML(n.title)}</strong><span class="small">${escapeHTML(n.year??'Year unknown')} · ${escapeHTML(n.topic||'No topic supplied')}${count>1?` · ${count} overlapping points—click to choose`:''}</span>`;
    t.style.display='block';t.style.left=Math.max(8,Math.min(e.clientX+14,innerWidth-365))+'px';t.style.top=Math.max(8,Math.min(e.clientY+14,innerHeight-t.offsetHeight-8))+'px';
  },
  select:(n,near=[])=>{selection=n;hits=near;renderSide();},
  viewport:({offscreen})=>{$('offscreen').textContent=tab==='map'&&offscreen?`${offscreen} current papers off-screen`:'';}
});

function parameters(){return new URLSearchParams({k:$('k').value,normalize:+$('normalize').checked,dups:+$('copies').checked,max_refs:$('maxrefs').value||0,max_cites:$('maxcites').value||0});}
function updateURL(id,push=false){const u=new URL(location.href);u.hash=id;u.searchParams.delete('select');u.searchParams.set('view','map');if(mapId)u.searchParams.set('atlas',mapId);else u.searchParams.delete('atlas');history[push?'pushState':'replaceState'](null,'',u);}
async function analyze(query,options={}) {
  query=String(query||'').trim();if(!query)return;
  const turn=++requestNumber;controller?.abort();controller=new AbortController();
  const wantedMap=options.newMap?null:mapId;
  const requestedSelection=new URLSearchParams(location.search).get('select');
  const p=parameters();p.set('id',query);if(wantedMap)p.set('atlas',wantedMap);
  $('query').value=query;$('status').textContent='Loading…';$('go').disabled=true;$('error').textContent='';
  try {
    const response=await fetch('/api/analyze?'+p,{signal:controller.signal});const data=await response.json();
    if(turn!==requestNumber)return;
    if(!response.ok||data.error)throw Error(data.error||`Request failed (${response.status})`);
    // Update the drawing before committing state; a coordinate-integrity failure
    // must not silently present a moving map as stable.
    plot.setData(data);current=data;mapId=data.atlas?.id||null;selection=null;hits=[];plot.setSelected(null);
    if(options.newMap)historyItems=[];
    const focus=data.nodes[data.focus];
    if(!options.keepHistory&&historyItems.at(-1)?.id!==focus.id)historyItems.push({id:focus.id,title:focus.title});
    updateURL(focus.id,options.push!==false);
    $('query').value=focus.id;$('status').textContent=`${data.stats.request_seconds??data.stats.seconds}s`;
    const years=Object.values(data.nodes).map(n=>n.year).filter(Number.isFinite);
    if(years.length){$('year').min=Math.min(...years);$('year').max=Math.max(...years);$('year').value=Number.isFinite(yearCut)?Math.max(+$('year').min,Math.min(+$('year').max,yearCut)):$('year').max;}
    plot.cutoff=yearCut;plot.draw();render();
    if(requestedSelection&&data.nodes[requestedSelection])selectPaper(requestedSelection);
    window.dispatchEvent(new CustomEvent('atlas-ready',{detail:{id:mapId,focus:data.focus}}));
  } catch(e) {
    if(turn!==requestNumber||e.name==='AbortError')return;
    $('error').textContent=e.message;$('status').textContent='Not loaded';
  } finally {if(turn===requestNumber)$('go').disabled=false;}
}

function roleText(n){
  if(!current||!current.nodes[n.id])return 'Starting-map landmark; not part of the current citation ranking.';
  const k=current.stats.k;
  if(n.id===current.focus)return 'Paper X.';
  if(n.column.startsWith('refs'))return `X cites this paper; ${n.ref_support??0} of the ${k} similar papers cite it too.`;
  if(n.column.startsWith('similar'))return `Shares ${n.shared??0} reference publications with X${n.outlier?'; excluded as an outlier':'.'}`;
  return `${n.support??0} of the ${k} similar papers cite it; X does not${n.outlier?'; excluded as an outlier':'.'}`;
}
function button(n,detail=''){return `<button class="item" data-paper="${escapeHTML(n.id)}">${escapeHTML(n.title||n.id)}<span class="meta">${escapeHTML(n.year??'?')}${detail?' · '+escapeHTML(detail):''}</span></button>`;}
function selectPaper(id){
  let n=current?.nodes[id]||[...plot.points.values()].find(p=>p.id===id);
  if(!n)return;
  selection=n;hits=[];plot.setSelected(n.map_status==='placed'?n.atlas_key:null);renderSide();
}
function bindPaperButtons(container){container.querySelectorAll('[data-paper]').forEach(b=>b.onclick=()=>selectPaper(b.dataset.paper));}

function render(){
  const f=current.nodes[current.focus],a=current.atlas;
  $('trail').innerHTML='Walk: '+historyItems.map((x,i)=>`<button data-step="${i}" title="${escapeHTML(x.title)}">${escapeHTML(compact(x.title,34))}</button>`).join(' → ');
  $('trail').querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{historyItems=historyItems.slice(0,+b.dataset.step+1);analyze(historyItems.at(-1).id,{keepHistory:true});});
  const answers=current.answers.map(id=>current.nodes[id]);
  $('summary').innerHTML=`<b>${escapeHTML(compact(f.title,100))}</b><br>`+
    answers.slice(0,2).map(n=>`<button class="answer-link" data-paper="${n.id}">${escapeHTML(compact(n.title,110))}</button>`).join(' · ')+
    `<span class="small"> — cited by ${current.stats.best_support}/${current.stats.k} similar papers; not by X.${answers.length>2?` ${answers.length} tied answers; all are in Papers.`:''}</span>`+
    `<div class="subtitle">${a?.id?`Fixed local map · ${a.plotted}/${a.total} result papers placed · axes anchored on ${a.anchor_count} papers. Only “Start new map” changes the axes.`:escapeHTML(a?.reason||'No projection available.')}</div>`;
  bindPaperButtons($('summary'));
  $('empty').textContent=a?.plotted?'':(a?.reason||'No result papers have usable vectors. They remain in the Papers list.');
  renderPapers();renderSide();
}
function renderPapers(){
  const nodes=Object.values(current.nodes);
  $('papers').innerHTML='<table><thead><tr><th>Publication</th><th>Year</th><th>Map</th><th>Role</th></tr></thead><tbody>'+nodes.map(n=>
    `<tr><td><button data-paper="${n.id}">${escapeHTML(n.title||n.id)}</button></td><td>${escapeHTML(n.year??'?')}</td><td>${n.map_status==='placed'?'Placed':'No placement'}</td><td>${escapeHTML(roleText(n))}</td></tr>`).join('')+'</tbody></table>';
  bindPaperButtons($('papers'));
}
function renderSide(){
  if(!current)return;
  const a=current.atlas,n=selection,s=current.stats;let h='';
  if(n){
    h+=`<h2>Selected publication</h2><h3>${escapeHTML(n.title||n.id)}</h3><p class="small">${escapeHTML(n.year??'Year unknown')} · ${fmt(n.cited_by)} citations · ${escapeHTML(n.topic||'Topic unknown')}</p><p>${escapeHTML(roleText(n))}</p>`;
    h+=`<button id="walk-here" class="primary">Walk here →</button> <a href="https://openalex.org/${encodeURIComponent(n.id)}" target="_blank" rel="noopener">Source record ↗</a>`;
    if(n.map_status!=='placed')h+=`<p class="warning">Not placed: ${escapeHTML(n.map_status==='no_projection'?(a?.reason||'a projection could not be fitted'):'no usable vector is available in this map')}. No position was guessed. You can still walk from this paper.</p>`;
    if(n.outlier)h+=`<p class="small">Outlier: ${escapeHTML(n.outlier)}. ${n.id===current.focus?'Kept as the starting paper because you chose it.':'Shown for context, excluded from the walk calculation.'}</p>`;
    if(n.vector_source_id)h+=`<p class="small">Coordinates use the saved vector for ${escapeHTML(n.vector_source_id)}.</p>`;
    if(hits.length>1){h+='<h2>Overlapping points</h2><p class="small">They were not pushed apart. Choose the paper you meant:</p>'+hits.slice(0,20).map(p=>button(p)).join('');}
    const nearest=n.semantic_neighbors||[];
    if(nearest.length){h+='<h2>Closest in the original vectors</h2><p class="small">Cosine similarity, among the evaluated papers in this result—not inferred from screen distance.</p>'+nearest.map(v=>button(current.nodes[v.id],`${v.similarity.toFixed(3)} cosine similarity`)).join('');}
    const outgoing=new Map(),incoming=new Map();
    for(const l of current.links){const other=l.s===n.id?current.nodes[l.t]:l.t===n.id?current.nodes[l.s]:null;if(other)(l.s===n.id?outgoing:incoming).set(other.id,other);}
    h+='<h2>Citations inside this result</h2><p class="small">Solid arrows point to references; dashed arrows arrive from citing papers. Only links whose endpoints are plotted and pass the year display filter can be drawn.</p>';
    if(outgoing.size)h+='<strong class="small">→ Cites</strong>'+[...outgoing.values()].slice(0,15).map(p=>button(p,p.map_status==='placed'?'plotted':'not placed')).join('');
    if(incoming.size)h+='<strong class="small">← Cited by</strong>'+[...incoming.values()].slice(0,15).map(p=>button(p,p.map_status==='placed'?'plotted':'not placed')).join('');
    if(!incoming.size&&!outgoing.size)h+='<p class="small">No citation connections supplied for this paper in the current result.</p>';
  } else {
    h+='<h2>The clean test</h2><p>No buildings, terrain, cluster-pulling, or citation-driven positions. One flat point per title-group with a genuine vector.</p><p>Select a point for its citation arrows and exact vector neighbors. Blank space means nothing by itself; this is not a geographic scale.</p>';
  }
  if(a?.id){const q=a.quality||{};h+=`<h2>Projection check</h2><dl class="facts quality"><dt>Placed</dt><dd>${a.plotted}/${a.total}</dd><dt>Anchor variance in 2D</dt><dd>${Math.round(100*a.anchor_variance_captured)}%</dd><dt>${q.k||5}-neighbor agreement</dt><dd>${q.neighbor_preservation==null?'Not enough points':Math.round(q.neighbor_preservation*100)+'%'}</dd></dl>`;
    h+=`<p class="small">Neighbor agreement compares this 2D projection with cosine neighbors in the original vectors, within ${q.evaluated||0} evaluated papers${q.evaluated<q.population?' (a bounded subset)':''}. Low agreement means nearby dots may be misleading.</p>`;
    if(q.neighbor_preservation!=null&&q.neighbor_preservation<0.5)h+='<p class="warning">Fewer than half of the original nearest neighbors survive in this projection. This is a diagnostic view—not reliable research geography yet.</p>';
    h+=`<details><summary class="small">Saved-map details</summary><p class="small">Local axes fitted once around “${escapeHTML(a.seed_title)}”. Not a whole-corpus atlas. Existing points and camera stay fixed while walking; Fit changes only the camera. Newly available vectors can add points without moving old ones.</p><p class="small">Map ${escapeHTML(a.id.slice(0,12))} · ${escapeHTML(a.algorithm)}</p></details>`;
  }
  const unplaced=(a?.unplaced||[]).map(id=>current.nodes[id]);
  if(unplaced.length)h+=`<h2>Not plotted (${unplaced.length})</h2><p class="small">Missing/invalid vectors, or insufficient anchors to fit a map. They still participate in the citation result; only their map positions are withheld.</p>`+unplaced.slice(0,12).map(p=>button(p,current.answers.includes(p.id)?'Answer—no map placement':'no map placement')).join('')+(unplaced.length>12?'<p class="small">The Papers list contains the rest.</p>':'');
  h+=`<h2>Scope</h2><p class="small">The ${s.k} similar-paper calculation and its outlier limits are unchanged. The map uses the existing DenseOn embedding snapshot; it does not repair bad abstract text. Title-based duplicate grouping is still a heuristic, not verified publication identity. The year slider changes visibility only.</p>`;
  $('side').innerHTML=h;bindPaperButtons($('side'));
  if($('walk-here'))$('walk-here').onclick=()=>analyze(n.id);
}
function setTab(value){tab=value;$('offscreen').textContent='';$('stage').style.display=value==='map'?'':'none';$('papers').style.display=value==='papers'?'block':'none';$('map-tab').setAttribute('aria-pressed',value==='map');$('papers-tab').setAttribute('aria-pressed',value==='papers');if(value==='map')plot.resize();}
$('go').onclick=()=>analyze($('query').value);$('query').onkeydown=e=>{if(e.key==='Enter')analyze($('query').value);};
$('new-map').onclick=()=>analyze(current?.focus||$('query').value,{newMap:true});
$('fit').onclick=()=>plot.fit();$('trail-visible').onchange=()=>{plot.showTrail=$('trail-visible').checked;plot.draw();};
$('neighbors-visible').onchange=()=>{plot.showSemantic=$('neighbors-visible').checked;plot.draw();};
$('year').oninput=()=>{yearCut=+$('year').value===+$('year').max?Infinity:+$('year').value;$('year-label').textContent=Number.isFinite(yearCut)?yearCut:'all years';plot.cutoff=yearCut;plot.draw();};
$('map-tab').onclick=()=>setTab('map');$('papers-tab').onclick=()=>setTab('papers');
for(const id of ['k','maxrefs','maxcites','normalize','copies'])$(id).onchange=()=>{if(current)analyze(current.focus,{keepHistory:true});};
document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>analyze(b.dataset.example));
window.addEventListener('popstate',()=>{mapId=new URLSearchParams(location.search).get('atlas');const q=decodeURIComponent(location.hash.slice(1));if(q)analyze(q,{push:false,keepHistory:true});});
const initial=decodeURIComponent(location.hash.slice(1));if(initial)analyze(initial,{push:false});
// Read-only diagnostic hooks used by browser tests, not by the computation.
window.atlasDebug={get data(){return current;},plot,analyze,select:selectPaper};
