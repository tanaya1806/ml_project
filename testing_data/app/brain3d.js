/* Interactive 3D glass-brain view for the screening report (plain canvas, no libraries).
   Data: /brain.json (cortex shell points + ROI positions) and the per-scan explanation returned by /api/screen. */
const Brain3D=(()=>{
  let data=null,canvas,ctx,W=0,H=0,scene=null,yaw=0.7,pitch=0.3,zoom=1,auto=true,drag=null,hl=-1,raf=0,lastTouch=0,hits=[];
  const shell=[];                                   // flat [x,y,z,...] of both hemispheres
  const COL={more:'#ff7a2e',less:'#5b8cff',node:'#22c3dc',bg:'rgba(60,190,215,0.22)'};

  async function load(){
    if(data)return;
    data=await (await fetch('/brain.json')).json();
    for(const s of ['left','right'])for(const p of data[s])shell.push(p[0],p[1],p[2]);
  }
  function resize(){
    const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;
    W=r.width;H=r.height;canvas.width=W*d;canvas.height=H*d;ctx.setTransform(d,0,0,d,0,0);
  }
  function project(x,y,z){                           // MNI: x right, y front, z up
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const X=cy*x+sy*y,Y=-sy*x+cy*y;                  // spin around the vertical axis
    const Z=sp*Y+cp*z,D=-cp*Y+sp*z;                  // viewer sits behind and above; D = distance towards the viewer
    const s=Math.min(W,H)/232*zoom;
    return [W/2+X*s,H/2-Z*s,D];
  }
  function draw(){
    const r=canvas.getBoundingClientRect();if(Math.abs(r.width-W)>1||Math.abs(r.height-H)>1)resize();
    ctx.clearRect(0,0,W,H);
    if(!data)return;
    // cortex shell as a depth-shaded point cloud (three alpha bands: far / middle / near)
    const bands=[[],[],[]];
    for(let i=0;i<shell.length;i+=3){const p=project(shell[i],shell[i+1],shell[i+2]);bands[p[2]<-25?0:p[2]<25?1:2].push(p)}
    [['rgba(120,175,205,0.14)',1.3],['rgba(120,175,205,0.28)',1.6],['rgba(150,205,230,0.46)',1.9]].forEach((st,b)=>{ctx.fillStyle=st[0];for(const p of bands[b])ctx.fillRect(p[0],p[1],st[1],st[1])});
    if(!scene)return;
    const P=n=>{const c=data.roi[n];return project(c[0],c[1],c[2])};
    ctx.lineCap='round';
    ctx.strokeStyle=COL.bg;ctx.lineWidth=0.7;ctx.beginPath();
    for(const e of scene.background_edges){const a=P(e.a),b=P(e.b);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}
    ctx.stroke();
    ctx.fillStyle=COL.node;
    for(const n of scene.nodes){const p=P(n);ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,6.283);ctx.fill()}
    hits=[];
    scene.top_edges.forEach((e,i)=>{
      const a=P(e.a),b=P(e.b),col=e.delta>=0?COL.more:COL.less,dim=hl>=0&&hl!==i;
      ctx.globalAlpha=dim?0.18:1;ctx.strokeStyle=col;ctx.lineWidth=(hl===i?3.5:2)+e.influence*2.5;
      ctx.shadowColor=col;ctx.shadowBlur=dim?0:10;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.shadowBlur=0;
      for(const p of [a,b]){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(p[0],p[1],5,0,6.283);ctx.fill();ctx.fillStyle=col;ctx.beginPath();ctx.arc(p[0],p[1],3.6,0,6.283);ctx.fill()}
      hits.push([i,a,b]);
    });
    ctx.globalAlpha=1;
  }
  function loop(){
    if(canvas.offsetParent===null){raf=0;return}     // hidden: stop until shown again
    if(auto&&!drag)yaw+=0.004;
    draw();raf=requestAnimationFrame(loop);
  }
  function start(){if(!raf)raf=requestAnimationFrame(loop)}
  function init(cv){
    canvas=cv;ctx=canvas.getContext('2d');resize();window.addEventListener('resize',()=>{resize();draw()});
    const down=(x,y)=>{drag=[x,y,yaw,pitch];auto=false;setAuto()};
    canvas.addEventListener('mousedown',e=>down(e.clientX,e.clientY));
    window.addEventListener('mousemove',e=>{if(drag){yaw=drag[2]+(e.clientX-drag[0])*0.008;pitch=Math.max(-1.4,Math.min(1.4,drag[3]+(e.clientY-drag[1])*0.008))}});
    window.addEventListener('mouseup',()=>{drag=null});
    canvas.addEventListener('touchstart',e=>{const t=e.touches[0];down(t.clientX,t.clientY)},{passive:true});
    canvas.addEventListener('touchmove',e=>{const t=e.touches[0];if(drag){yaw=drag[2]+(t.clientX-drag[0])*0.008;pitch=Math.max(-1.4,Math.min(1.4,drag[3]+(t.clientY-drag[1])*0.008));e.preventDefault()}},{passive:false});
    canvas.addEventListener('touchend',()=>{drag=null});
    canvas.addEventListener('mousemove',e=>{
      if(drag||!hoverCb)return;
      const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;let best=-1,bd=10;
      for(const [i,a,b] of hits){const dx=b[0]-a[0],dy=b[1]-a[1],L=dx*dx+dy*dy||1,t=Math.max(0,Math.min(1,((mx-a[0])*dx+(my-a[1])*dy)/L)),d=Math.hypot(mx-(a[0]+t*dx),my-(a[1]+t*dy));if(d<bd){bd=d;best=i}}
      hl=best;canvas.style.cursor=best>=0?'pointer':'grab';hoverCb(best,e.clientX,e.clientY);
    });
    canvas.addEventListener('mouseleave',()=>{hl=-1;hoverCb&&hoverCb(-1)});
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.6,Math.min(2.6,zoom*(e.deltaY<0?1.08:0.93)))},{passive:false});
  }
  let setAuto=()=>{},hoverCb=null;
  return {
    async show(cv,explain,onAuto){
      if(!canvas)init(cv);
      setAuto=()=>onAuto&&onAuto(auto);
      await load();scene=explain;hl=-1;resize();start();
    },
    view(name){auto=false;setAuto();({left:[-1.5708,0.05],right:[1.5708,0.05],top:[0,1.5],front:[3.1416,0.05],back:[0,0.05]})[name]&&([yaw,pitch]=({left:[-1.5708,0.05],right:[1.5708,0.05],top:[0,1.5],front:[3.1416,0.05],back:[0,0.05]})[name]);zoom=1},
    toggleAuto(){auto=!auto;setAuto();return auto},
    highlight(i){hl=i},
    onHover(cb){hoverCb=cb},
  };
})();
