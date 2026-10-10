/* Flat, fixed-coordinate diagnostic. No layout forces or moving points to avoid overlaps.
 * Camera zoom is uniform. Positions come only from the persisted server projection.
 * Coincident markers stay coincident; the click handler exposes the hit list.
 */
(function (global) {
  'use strict';
  const finitePoint = n => n && Number.isFinite(n.mx) && Number.isFinite(n.my) && n.map_status === 'placed';
  const worldToScreen = (p, camera, size) => ({x: (p.mx-camera.x)*camera.scale+size.w/2, y: size.h/2-(p.my-camera.y)*camera.scale});
  const screenToWorld = (p, camera, size) => ({x: (p.x-size.w/2)/camera.scale+camera.x, y: (size.h/2-p.y)/camera.scale+camera.y});
  const role = n => n.column === 'x' ? 'focus' : n.answer ? 'answer' : n.column?.startsWith('similar') ? 'similar' : n.column?.startsWith('candidates') ? 'candidate' : 'reference';

  class AtlasPlot {
    constructor(canvas, hooks={}) {
      this.canvas=canvas;this.ctx=canvas.getContext('2d');this.hooks=hooks;
      this.points=new Map();this.result=null;this.mapId=null;this.camera={x:0,y:0,scale:500};
      this.size={w:1,h:1};this.selected=null;this.hover=null;this.showTrail=true;this.showSemantic=false;
      this.cutoff=Infinity;this.frame=0;this.drag=null;
      this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(canvas);
      canvas.addEventListener('pointerdown',e=>this.down(e));
      canvas.addEventListener('pointermove',e=>this.move(e));
      canvas.addEventListener('pointerup',e=>this.up(e));
      canvas.addEventListener('pointercancel',()=>{this.drag=null;canvas.style.cursor='grab'});
      canvas.addEventListener('pointerleave',()=>{if(!this.drag){this.hover=null;this.hooks.hover?.(null);this.draw()}});
      canvas.addEventListener('wheel',e=>this.zoom(e),{passive:false});
      canvas.addEventListener('keydown',e=>{if(e.key==='Escape'){this.setSelected(null);this.hooks.select?.(null)}});
      this.resize();
    }
    resize() {
      const r=this.canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
      if(r.width<2||r.height<2)return; // switching to the Papers tab must not destroy the camera geometry
      this.size={w:r.width,h:r.height};
      this.canvas.width=Math.round(this.size.w*dpr);this.canvas.height=Math.round(this.size.h*dpr);
      this.ctx.setTransform(dpr,0,0,dpr,0,0);this.draw();
    }
    setData(data) {
      const id=data.atlas?.id||null,fresh=id!==this.mapId;
      if(fresh){this.points.clear();this.mapId=id;this.selected=null;this.hover=null;}
      for(const p of this.points.values())p.current=false;
      for(const n of [...(data.atlas?.landmarks||[]),...Object.values(data.nodes)]) {
        if(!finitePoint(n))continue;
        const old=this.points.get(n.atlas_key);
        if(old && (Math.abs(old.mx-n.mx)>1e-12 || Math.abs(old.my-n.my)>1e-12))throw Error('A saved map coordinate changed. Start a new map.');
        const current=Boolean(data.nodes[n.id]) && data.nodes[n.id].atlas_key===n.atlas_key;
        this.points.set(n.atlas_key,{...old,...n,current});
      }
      this.result=data;
      // Never refit automatically while walking. New points may be off-screen;
      // the user can explicitly fit them without changing their world positions.
      if(fresh || !this.fitted) {
        let saved=null;
        try{saved=JSON.parse(sessionStorage.getItem('atlas-camera:'+id)||'null')}catch{}
        if(saved && [saved.x,saved.y,saved.scale].every(Number.isFinite) && saved.scale>0){this.camera=saved;this.fitted=true;}
        else this.fit();
      }
      this.draw();
    }
    visible(p) {return (p.current||this.showTrail) && (p.year==null||p.year<=this.cutoff);}
    currentPoints(){return [...this.points.values()].filter(p=>p.current&&this.visible(p));}
    fit() {
      const pts=this.currentPoints().length?this.currentPoints():[...this.points.values()];
      if(!pts.length){this.draw();return;}
      const xs=pts.map(p=>p.mx),ys=pts.map(p=>p.my);
      const loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys);
      const dx=Math.max(hiX-loX,0.02),dy=Math.max(hiY-loY,0.02);
      this.camera={x:(loX+hiX)/2,y:(loY+hiY)/2,scale:Math.max(10,Math.min((this.size.w-100)/dx,(this.size.h-100)/dy))};
      this.fitted=true;this.saveCamera();this.draw();
    }
    saveCamera(){if(this.mapId)try{sessionStorage.setItem('atlas-camera:'+this.mapId,JSON.stringify(this.camera))}catch{}}
    setSelected(key){this.selected=key;this.hover=null;this.draw();}
    screen(p){return worldToScreen(p,this.camera,this.size);}
    point(key){return this.points.get(key);}
    hits(x,y) {
      return [...this.points.values()].filter(p=>this.visible(p)).map(p=>{
        const s=this.screen(p);return {p,d:(s.x-x)**2+(s.y-y)**2};
      }).filter(o=>o.d<=100).sort((a,b)=>a.d-b.d||Number(b.p.current)-Number(a.p.current)||a.p.id.localeCompare(b.p.id)).map(o=>o.p);
    }
    eventPosition(e){const r=this.canvas.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};}
    down(e) {
      if(e.button!==0)return;
      const p=this.eventPosition(e);this.drag={x:p.x,y:p.y,camera:{...this.camera},moved:false};
      this.canvas.setPointerCapture(e.pointerId);
    }
    move(e) {
      const p=this.eventPosition(e);
      if(this.drag) {
        const dx=p.x-this.drag.x,dy=p.y-this.drag.y;
        if(Math.hypot(dx,dy)>4)this.drag.moved=true;
        if(this.drag.moved){this.camera.x=this.drag.camera.x-dx/this.drag.camera.scale;this.camera.y=this.drag.camera.y+dy/this.drag.camera.scale;
          this.canvas.style.cursor='grabbing';this.hooks.hover?.(null);this.draw();}
      } else {
        const hits=this.hits(p.x,p.y);this.hover=hits[0]?.atlas_key||null;
        this.canvas.style.cursor=hits.length?'pointer':'grab';this.hooks.hover?.(hits[0]||null,e,hits.length);this.draw();
      }
    }
    up(e) {
      if(!this.drag)return;
      const wasDrag=this.drag.moved;this.drag=null;this.canvas.style.cursor='grab';
      this.canvas.releasePointerCapture(e.pointerId);
      if(wasDrag){this.saveCamera();return;}
      const p=this.eventPosition(e),hits=this.hits(p.x,p.y);
      this.setSelected(hits[0]?.atlas_key||null);this.hooks.select?.(hits[0]||null,hits);
    }
    zoom(e) {
      e.preventDefault();const p=this.eventPosition(e),at=screenToWorld(p,this.camera,this.size);
      const scale=Math.max(10,Math.min(100000,this.camera.scale*Math.exp(-Math.max(-150,Math.min(150,e.deltaY))*0.005)));
      this.camera={x:at.x-(p.x-this.size.w/2)/scale,y:at.y+(p.y-this.size.h/2)/scale,scale};
      this.saveCamera();this.draw();
    }
    draw(){if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=0;this.paint()});}
    paint() {
      const c=this.ctx,w=this.size.w,h=this.size.h,css=getComputedStyle(this.canvas);
      const color=name=>css.getPropertyValue('--'+name).trim();
      this.colors={focus:color('ink'),answer:color('answer'),similar:color('similar'),candidate:color('candidate'),reference:color('muted'),landmark:color('border')};
      c.clearRect(0,0,w,h);
      const active=this.hover||this.selected,neighborhood=new Set(active?[active]:[]),drawn=[];
      if(active&&this.result) {
        for(const l of this.result.links) {
          const a=this.result.nodes[l.s],b=this.result.nodes[l.t];
          if(!finitePoint(a)||!finitePoint(b))continue;
          const ak=a.atlas_key,bk=b.atlas_key;
          if(ak!==active&&bk!==active)continue;
          const p=this.points.get(ak),q=this.points.get(bk);
          if(!p||!q||!this.visible(p)||!this.visible(q))continue;
          const edgeKey=ak+'\0'+bk;
          if(drawn.includes(edgeKey))continue;
          drawn.push(edgeKey);neighborhood.add(ak);neighborhood.add(bk);
          this.arrow(this.screen(p),this.screen(q),ak===active?color('outgoing'):color('incoming'),ak!==active);
        }
      }
      // Equal-size dots: citation counts, category labels and selected role do not
      // affect geometry. Red diamonds and the focus ring retain task semantics.
      const points=[...this.points.values()].filter(p=>this.visible(p));
      points.sort((a,b)=>Number(a.current)-Number(b.current));
      let offscreen=0;
      for(const p of points) {
        const s=this.screen(p);if(s.x<0||s.x>w||s.y<0||s.y>h){if(p.current)offscreen++;continue;}
        const r=role(p),fade=active&&!neighborhood.has(p.atlas_key);
        c.globalAlpha=p.current?(fade?0.22:(p.column?.endsWith('_out')?0.5:1)):0.25;
        c.fillStyle=p.current?this.colors[r]:this.colors.landmark;
        c.beginPath();
        if(p.current&&r==='answer'){c.moveTo(s.x,s.y-7);c.lineTo(s.x+7,s.y);c.lineTo(s.x,s.y+7);c.lineTo(s.x-7,s.y);c.closePath();}
        else c.arc(s.x,s.y,p.current?5:3,0,Math.PI*2);
        c.fill();
        if(p.current&&r==='focus'){c.strokeStyle=color('ink');c.lineWidth=1.5;c.beginPath();c.arc(s.x,s.y,9,0,Math.PI*2);c.stroke();}
        if(p.atlas_key===this.selected){c.globalAlpha=1;c.strokeStyle=color('selected');c.lineWidth=2;c.beginPath();c.arc(s.x,s.y,12,0,Math.PI*2);c.stroke();}
      }
      c.globalAlpha=1;
      // Optional exact-neighbor rings are annotations, not links or layout forces.
      const selected=this.points.get(active);
      if(this.showSemantic&&selected) {
        for(const n of selected.semantic_neighbors||[]) {
          const node=this.result.nodes[n.id],p=node&&this.points.get(node.atlas_key);
          if(!p||!this.visible(p))continue;
          const s=this.screen(p);c.strokeStyle=color('semantic');c.lineWidth=1.8;c.beginPath();c.arc(s.x,s.y,10,0,Math.PI*2);c.stroke();
        }
      }
      // At most focus + selected + two answers get labels; the rest use hover.
      const labels=points.filter(p=>p.current&&(role(p)==='focus'||role(p)==='answer'||p.atlas_key===active));
      const boxes=[];
      for(const p of labels.slice(0,5)) {
        const s=this.screen(p);if(s.x<0||s.x>w||s.y<0||s.y>h)continue;
        const title=(p.title||p.id),text=title.length>53?title.slice(0,52)+'…':title;
        c.font='12px system-ui';const tw=c.measureText(text).width;
        const x=Math.max(8,Math.min(w-tw-10,s.x+14));let y=s.y-10;
        while(boxes.some(b=>x<b.x+b.w&&x+tw>b.x&&y-12<b.y&&y>b.y-14))y+=16;
        boxes.push({x,y,w:tw});c.fillStyle=color('plot-bg');c.fillRect(x-3,y-12,tw+6,16);
        c.fillStyle=role(p)==='answer'?color('answer'):color('ink');c.fillText(text,x,y);
      }
      this.canvas.dataset.points=String(points.length);this.canvas.dataset.currentPoints=String(points.filter(p=>p.current).length);
      this.canvas.dataset.visibleEdges=String(drawn.length);this.canvas.dataset.mapId=this.mapId||'';
      this.hooks.viewport?.({offscreen,shown:points.filter(p=>p.current).length});
    }
    arrow(a,b,color,dashed) {
      const c=this.ctx,dx=b.x-a.x,dy=b.y-a.y,len=Math.hypot(dx,dy);
      if(len<20)return;
      const ux=dx/len,uy=dy/len,start={x:a.x+9*ux,y:a.y+9*uy},end={x:b.x-9*ux,y:b.y-9*uy};
      c.strokeStyle=color;c.fillStyle=color;c.lineWidth=1.5;c.globalAlpha=0.8;c.setLineDash(dashed?[4,3]:[]);
      c.beginPath();c.moveTo(start.x,start.y);c.lineTo(end.x,end.y);c.stroke();c.setLineDash([]);
      c.beginPath();c.moveTo(end.x,end.y);c.lineTo(end.x-7*ux+3.5*uy,end.y-7*uy-3.5*ux);
      c.lineTo(end.x-7*ux-3.5*uy,end.y-7*uy+3.5*ux);c.closePath();c.fill();c.globalAlpha=1;
    }
  }
  global.AtlasPlot=AtlasPlot;
  if(typeof module!=='undefined')module.exports={worldToScreen,screenToWorld,finitePoint};
})(typeof window!=='undefined'?window:globalThis);
