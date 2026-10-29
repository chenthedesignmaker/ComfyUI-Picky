// Local schematic camera planner. The server compiles saved paths into H3 prose.
const point = (azimuth = 0, elevation = 0, distance = 1, time = 1) => ({ time, azimuth, elevation, distance });
export const CAMERA_PRESETS = {
  right: { label: "Orbit right 90°", end: point(90) },
  left: { label: "Orbit left 90°", end: point(-90) },
  push: { label: "Push in", end: point(0, 0, .6) },
  pull: { label: "Pull back", end: point(0, 0, 1.6) },
  rise: { label: "Rise 20°", end: point(0, 20) },
  arc: { label: "Rising orbit", end: point(60, 15) },
  circle: { label: "Full orbit 360°", end: point(360) },
  fall: { label: "Lower 20°", end: point(0, -20) },
  hold: { label: "Static camera", end: point() },
};
export function newCameraPath(preset = "right") {
  return { target: "", interpolation: "smooth", freeze: false,
    keyframes: [point(0, 0, 1, 0), { ...CAMERA_PRESETS[preset].end }] };
}
export function samplePath(path, time) {
  const points = path.keyframes;
  let i = points.findIndex((p, j) => j && p.time >= time);
  if (i < 1) i = points.length - 1;
  const a = points[i - 1], b = points[i];
  // Ease only on the path's entry/exit or an axis reversal. Intermediate
  // points retain velocity; a cubic Hermite interval makes that visible.
  const span = b.time - a.time;
  const t = Math.max(0, Math.min(1, (time - a.time) / span));
  const out = { time };
  for (const key of ["azimuth", "elevation", "distance"]) {
    const slope = (b[key] - a[key]) / span;
    if (path.interpolation === "linear") { out[key] = a[key] + (b[key] - a[key]) * t; continue; }
    const tangent = j => {
      if (j === 0 || j === points.length - 1) return 0;
      const prev = (points[j][key] - points[j - 1][key]) / (points[j].time - points[j - 1].time);
      const next = (points[j + 1][key] - points[j][key]) / (points[j + 1].time - points[j].time);
      if (prev * next <= 0) return 0;
      const h0=points[j].time-points[j-1].time, h1=points[j+1].time-points[j].time;
      const w0=2*h1+h0, w1=h1+2*h0;
      return (w0+w1)/(w0/prev+w1/next);
    };
    out[key] = Math.max(Math.min(a[key],b[key]),Math.min(Math.max(a[key],b[key]),
      (2*t*t*t - 3*t*t + 1)*a[key] + (t*t*t - 2*t*t + t)*span*tangent(i-1)
      + (-2*t*t*t + 3*t*t)*b[key] + (t*t*t - t*t)*span*tangent(i)));
  }
  return out;
}

export function unwrapPath(path) {
  const result=structuredClone(path), points=result.keyframes;
  for(let i=1;i<points.length;i++) {
    const raw=path.keyframes[i].azimuth-path.keyframes[i-1].azimuth;
    // Explicit complete turns must survive this editing aid.
    const delta=Math.abs(raw)>=359.999 ? raw : ((raw+180)%360+360)%360-180;
    points[i].azimuth=points[i-1].azimuth+delta;
  }
  if(points.some(p=>Math.abs(p.azimuth)>720)) throw new Error("Unwrapping would exceed the ±720° range.");
  return result;
}
export function redistributePath(path) {
  const result=structuredClone(path), points=result.keyframes;
  const weights=points.slice(1).map((p,i)=>Math.hypot((p.azimuth-points[i].azimuth)*Math.PI/180,
    (p.elevation-points[i].elevation)*Math.PI/180,Math.log(p.distance/points[i].distance)));
  const hold=weights.map((w,i)=>w<1e-9?points[i+1].time-points[i].time:0);
  const remaining=1-hold.reduce((a,b)=>a+b,0), total=weights.reduce((a,b)=>a+b,0);
  if(total<1e-9)return result;
  let time=0;
  points.slice(1).forEach((p,i)=>{time+=hold[i]||remaining*weights[i]/total;p.time=time;});
  points.at(-1).time=1;
  return result;
}
export function addHolds(path,duration,seconds=.5) {
  if(duration<=seconds*2 || path.keyframes.length>10) throw new Error("Holds need room for two points and more than 1 second.");
  const result=structuredClone(path), fraction=seconds/duration;
  result.keyframes=result.keyframes.map(p=>({...p,time:fraction+p.time*(1-2*fraction)}));
  result.keyframes.unshift({...result.keyframes[0],time:0});
  result.keyframes.push({...result.keyframes.at(-1),time:1});
  return result;
}
export function pathDiagnostics(path,{whole=true,hasFirst=false,hasReferences=false,hasDifferentLast=false,duration=1}={}) {
  const points=path.keyframes, first=points[0], last=points.at(-1);
  const travel=points.slice(1).reduce((s,p,i)=>s+Math.abs(p.azimuth-points[i].azimuth),0);
  const holds=points.slice(1).filter((p,i)=>["azimuth","elevation","distance"].every(k=>p[k]===points[i][k])).length;
  const notes=[`${travel.toFixed(1)}° total travel · ${last.azimuth-first.azimuth}° net orbit · ${holds} hold interval${holds===1?"":"s"}.`];
  const reversals=points.slice(1,-1).filter((p,i)=>(p.azimuth-points[i].azimuth)*(points[i+2].azimuth-p.azimuth)<0);
  if(reversals.length)notes.push(`Orbit reverses at ${reversals.map(p=>(p.time*duration).toFixed(2)+"s").join(", ")}.`);
  if(holds===points.length-1)notes.push("The camera stays static throughout this path.");
  if(points.slice(1).some((p,i)=>p.azimuth*points[i].azimuth<0))notes.push("The orbit crosses its opening angle between keyframes.");
  if(path.frame_mode==="motion") notes.push("Motion Frame uses the selected clip; action continues during camera holds. Saved still-frame anchors are inactive in this mode. Source audio is not copied.");
  if(path.loop_closure==="auto") {
    const eligible=whole&&path.frame_mode!=="motion"&&path.freeze&&hasFirst&&!hasReferences&&!hasDifferentLast
      &&Math.abs(Math.abs(last.azimuth-first.azimuth)-360)<1e-6&&Math.abs(last.elevation-first.elevation)<1e-6&&Math.abs(last.distance-first.distance)<1e-6;
    notes.push(eligible?"Loop closure ON: first image also anchors the final frame. This does not guarantee a full orbit.":
      "Loop closure OFF: requires one full-generation frozen path, a first image, exactly ±360°, matching end height/distance, no references and no conflicting last image.");
  }
  if(points.some(p=>Math.abs(p.elevation)>30)) notes.push("Steep elevation may put the horizon outside the frame.");
  if(path.experiment && path.experiment!=="off") notes.push("Experimental text recipe; no proven improvement in camera accuracy. Compare with the same seed.");
  return notes;
}
export function cameraPathSummary(path) {
  if (!path?.keyframes?.length) return "Camera path";
  const last = path.keyframes.at(-1);
  return `${path.frame_mode==="motion"?"Motion Frame · ":""}${path.keyframes.length} keyframes · ${last.azimuth}° orbit · ${last.elevation}° height · ${last.distance}× distance`;
}

export function openCameraPlanner({ path, duration, imageURL, title, apply, videos=[], uploadVideo, mediaURL=p=>p, whole=true,
  hasFirst=!!imageURL, hasReferences=false, hasDifferentLast=false }) {
  let draft = structuredClone(path || newCameraPath());
  // Malformed imported paths remain blocked by the server. Offer a fresh
  // editable draft without touching them until Apply is pressed.
  if (!Array.isArray(draft.keyframes) || draft.keyframes.length < 2
    || draft.keyframes.some(p => !p || [p.time, p.azimuth, p.elevation, p.distance].some(n => !Number.isFinite(n)))) draft = newCameraPath();
  duration = Math.max(.001, duration);
  let selected = draft.keyframes.length - 1, playhead = 1, playing = false, raf = 0;
  let returnedFocus = document.activeElement;
  const dialog = document.createElement("dialog");
  dialog.id = "camera-planner";
  dialog.setAttribute("aria-labelledby", "cp-title");
  dialog.innerHTML = `
    <header class="cp-head"><div><span class="cp-eyebrow">SHOT DIRECTION</span><h2 id="cp-title">Camera path</h2><p id="cp-context"></p></div><button type="button" id="cp-close" class="ghost" aria-label="Close camera planner">Close ×</button></header>
    <div class="cp-body">
      <div class="cp-stage-column">
        <div class="cp-stage"><canvas id="cp-canvas" aria-label="Schematic three-dimensional camera path"></canvas><span class="cp-stage-label">PATH PREVIEW · SCHEMATIC</span><p>Drag to orbit or raise the selected keyframe. Scroll to change distance.</p></div>
        <div class="cp-transport"><button type="button" id="cp-play" aria-label="Play path preview">▶ Play</button><input id="cp-scrub" type="range" min="0" max="1" step="0.001" value="1" aria-label="Camera preview time"><output id="cp-time"></output></div>
        <div class="cp-reference" id="cp-reference"><img id="cp-image" alt="Scene first-frame reference"><div><strong>Scene first frame</strong><p id="cp-reference-note">The diagram shows camera direction, not a reconstruction of this image.</p></div></div>
        <div class="cp-box-preview" id="cp-box-preview"><img id="cp-box-image" alt="Draw a box around the camera target"><video id="cp-box-video" muted playsinline preload="metadata"></video><div id="cp-box-overlay" aria-label="Drag to select subject"><span id="cp-box-rect"></span></div></div>
        <p class="cp-note">Drag a subject box on the reference, or enter its normalized bounds below. It identifies the opening target; it does not crop the picture.</p>
        <div class="cp-controls" id="cp-box-fields"></div><button type="button" class="small ghost" id="cp-clear-box">Clear subject box</button>
        <p class="cp-note">H3 follows the path as written direction. Exact angles and timing can vary.</p>
        <div class="cp-note" id="cp-diagnostics" role="status"></div>
      </div>
      <div class="cp-edit-column">
        <label for="cp-mode">Camera pass<select id="cp-mode"><option value="scene">Scene camera path</option><option value="motion">Motion Frame · video reference (preview)</option></select></label>
        <div id="cp-motion-controls"><label for="cp-video">Reference clip<select id="cp-video"><option value="">Choose a reference clip…</option></select></label><button type="button" id="cp-upload" class="small ghost">Upload reference video</button><p class="cp-note">Use a 2–15s clip and a Ref2VA / H3 Hybrid render profile. Match the generation duration to the action you want to preserve: reference frames beyond it are omitted. The reference is converted to 24 fps; up to 16 trailing frames may be trimmed.</p><p class="cp-note">Preview feature: H3 can reinterpret poses, room layout and camera angles. Check a short render before relying on action continuity.</p></div>
        <label for="cp-preset">Start with a move<select id="cp-preset"><option value="">Choose a preset…</option></select></label>
        <label for="cp-target">Aim at<input id="cp-target" maxlength="300" placeholder="e.g. the kitchen island, the woman in the red coat"></label>
        <div class="cp-key-head"><strong>Keyframes</strong><button type="button" class="small ghost" id="cp-add">+ Add keyframe</button></div>
        <div id="cp-keyframes" class="cp-keyframes" role="group" aria-label="Select camera keyframe"></div>
        <div class="cp-path-actions"><button type="button" class="small ghost" id="cp-retime">Even travel timing</button><button type="button" class="small ghost" id="cp-unwrap">Unwrap angles</button><button type="button" class="small ghost" id="cp-holds">Add 0.5s end holds</button><button type="button" class="small ghost" id="cp-close-orbit">Close orbit</button></div>
        <div class="cp-selected"><strong id="cp-selected"></strong><button type="button" id="cp-delete" class="small ghost">Remove point</button></div>
        <div class="cp-controls">
          <label for="cp-point-time">Time in this beat <span>seconds</span><input id="cp-point-time" type="number" step="0.01"></label>
          <label for="cp-orbit">Orbit <span>degrees · + right / − left</span><input id="cp-orbit" type="number" min="-720" max="720" step="1"><input id="cp-orbit-range" type="range" min="-720" max="720" step="1" aria-label="Orbit degrees"></label>
          <label for="cp-height">Height angle <span>degrees · + up / − down</span><input id="cp-height" type="number" min="-89" max="89" step="1"><input id="cp-height-range" type="range" min="-89" max="89" step="1" aria-label="Height angle degrees"></label>
          <label for="cp-distance">Distance <span>× opening distance · closer to 1 = gentler move</span><input id="cp-distance" type="number" min="0.05" max="4" step="0.05"><input id="cp-distance-range" type="range" min="0.05" max="4" step="0.05" aria-label="Distance multiplier"></label>
        </div>
        <label for="cp-easing">Motion<select id="cp-easing"><option value="smooth">Smooth through keyframes</option><option value="linear">Constant speed per interval</option></select></label>
        <label class="cp-check"><input id="cp-freeze" type="checkbox">Freeze scene while the camera moves</label>
        <label for="cp-loop">Loop closure<select id="cp-loop"><option value="off">Off</option><option value="auto">Auto · anchor final frame when compatible</option></select></label>
        <details><summary>Advanced · prompt experiments</summary><label for="cp-experiment">Prompt recipe<select id="cp-experiment"><option value="off">Standard camera direction</option><option value="compact">Direction + numeric keyframes</option><option value="numbers">Numeric keyframes · experimental</option></select></label><p class="cp-note">These recipes still send text to H3. They do not add geometric conditioning.</p></details>
        <p class="cp-note" id="cp-lock-note"></p>
      </div>
    </div>
    <footer class="cp-foot"><p id="cp-status" role="status" aria-live="polite"></p><button type="button" id="cp-remove" class="ghost">Use written camera</button><button type="button" id="cp-cancel" class="ghost">Cancel</button><button type="button" id="cp-apply" class="primary">Apply camera path</button></footer>`;
  document.body.append(dialog);
  const el = key => dialog.querySelector(`#cp-${key}`);
  el("context").textContent = `${title} · ${duration.toFixed(2)}s · follows this beat when retimed`;
  for (const [key, value] of Object.entries(CAMERA_PRESETS)) el("preset").append(new Option(value.label, key));
  el("target").value = draft.target || "";
  el("easing").value = draft.interpolation || "smooth";
  el("freeze").checked = draft.freeze === true;
  el("mode").value=draft.frame_mode||"scene";
  el("loop").value=draft.loop_closure||"off";
  el("experiment").value=draft.experiment||"off";
  const videoChoices=new Map(videos.map(v=>[v.path,v.label||v.path]));
  if(draft.reference_video)videoChoices.set(draft.reference_video,videoChoices.get(draft.reference_video)||draft.reference_video);
  for(const [value,label] of videoChoices)el("video").append(new Option(label,value));
  el("video").value=draft.reference_video||"";
  el("upload").disabled=!uploadVideo;
  for(const key of ["left","top","width","height"]) {
    const label=document.createElement("label");label.textContent=key[0].toUpperCase()+key.slice(1);
    const input=document.createElement("input");input.type="number";input.min="0";input.max="1";input.step=".01";input.id=`cp-box-${key}`;
    input.setAttribute("aria-label",`Subject box ${key}`);label.append(input);el("box-fields").append(label);
    input.onchange=()=>{const value=input.valueAsNumber;if(!Number.isFinite(value))return;
      const box={left:0,top:0,width:1,height:1,...draft.subject_box,[key]:Math.max(0,Math.min(1,value))};
      box.left=Math.min(.99,box.left);box.top=Math.min(.99,box.top);
      box.width=Math.max(.001,Math.min(box.width,1-box.left));box.height=Math.max(.001,Math.min(box.height,1-box.top));
      draft.subject_box=box;updateBox();};
  }
  function updateBox() {
    const box=draft.subject_box;el("box-rect").hidden=!box;
    if(box)Object.assign(el("box-rect").style,{left:`${box.left*100}%`,top:`${box.top*100}%`,width:`${box.width*100}%`,height:`${box.height*100}%`});
    for(const key of ["left","top","width","height"])el(`box-${key}`).value=box?box[key].toFixed(3):"";
  }
  function syncReference() {
    const motion=draft.frame_mode==="motion", video=el("box-video"), img=el("box-image");
    el("motion-controls").hidden=!motion;el("freeze").disabled=motion;
    el("reference").hidden=motion;
    img.hidden=motion||!imageURL;video.hidden=!motion||!draft.reference_video;
    if(motion&&draft.reference_video) {const url=mediaURL(draft.reference_video);if(video.getAttribute("src")!==url)video.src=url;}
    if(!motion&&imageURL)img.src=imageURL;
    el("box-preview").hidden=motion?!draft.reference_video:!imageURL;
    updateBox();
  }
  el("box-image").onerror=()=>{el("box-preview").hidden=true;};
  el("box-video").onerror=()=>{el("status").textContent="Reference video preview unavailable. Check the file before rendering.";};
  el("clear-box").onclick=()=>{delete draft.subject_box;updateBox();};
  let boxDrag=null;
  const boxPoint=event=>{const r=el("box-overlay").getBoundingClientRect();return {x:Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(event.clientY-r.top)/r.height))};};
  el("box-overlay").onpointerdown=event=>{boxDrag=boxPoint(event);el("box-overlay").setPointerCapture(event.pointerId);};
  el("box-overlay").onpointermove=event=>{if(!boxDrag)return;const p=boxPoint(event);const width=Math.abs(p.x-boxDrag.x),height=Math.abs(p.y-boxDrag.y);
    if(width<.001||height<.001)return;draft.subject_box={left:Math.min(p.x,boxDrag.x),top:Math.min(p.y,boxDrag.y),width,height};updateBox();};
  el("box-overlay").onpointerup=el("box-overlay").onpointercancel=()=>{boxDrag=null;};
  el("remove").hidden = !path;
  el("retime").title="Redistribute time by angular and relative-radius travel, preserving explicit holds. This is a planning estimate of even speed.";
  el("unwrap").title="Take the short arc across an angle boundary; explicit complete turns stay intact. Inspect the result before applying.";
  if (imageURL) el("image").src = imageURL;
  else { el("image").hidden = true; el("reference").querySelector("strong").textContent = "No first frame attached"; el("reference-note").textContent = "You can plan a move now. Add a first frame in Media to anchor the scene's opening view."; }
  el("image").onerror = () => { el("image").hidden = true; el("reference-note").textContent = "First-frame preview unavailable. Your attached reference is unchanged."; };
  const canvas = el("canvas"), ctx = canvas.getContext("2d");
  let hitPoints = [];
  function draw() {
    const box = canvas.getBoundingClientRect(), width = box.width, height = box.height;
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0c0c0e"; ctx.fillRect(0, 0, width, height);
    const scale = Math.min(width / 8.5, height / 5.8), cx = width*.5, cy = height*.54;
    // Oblique projection of a ground plane and a vertical camera axis.
    const project = (x, y, z) => [cx + scale * (x*.84 - z*.48), cy + scale * (x*.23 + z*.42 - y)];
    const pose = p => { const a = p.azimuth*Math.PI/180, e = p.elevation*Math.PI/180;
      return project(Math.sin(a)*Math.cos(e)*p.distance*2, Math.sin(e)*p.distance*2, Math.cos(a)*Math.cos(e)*p.distance*2); };
    const line = (a, b, color, stroke = 1) => { ctx.strokeStyle=color; ctx.lineWidth=stroke; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); };
    for (let n=-4;n<=4;n++) { line(project(n,0,-4),project(n,0,4),"#26262a"); line(project(-4,0,n),project(4,0,n),"#26262a"); }
    ctx.strokeStyle="#38383e"; ctx.setLineDash([4,5]); ctx.beginPath();
    for(let i=0;i<=100;i++){ const p=pose(point(i*3.6)); i ? ctx.lineTo(...p) : ctx.moveTo(...p); } ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle="#e9b45c";ctx.lineWidth=2.5;ctx.beginPath();
    for(let i=0;i<=180;i++){const p=pose(samplePath(draft,i/180));i?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.stroke();
    const origin=project(0,0,0); line(origin,project(0,1,0),"#b78af5",3);
    ctx.fillStyle="#b78af5";ctx.beginPath();ctx.arc(...origin,7,0,Math.PI*2);ctx.fill();
    ctx.font="12px system-ui";ctx.fillStyle="#d9c7f5";ctx.fillText("TARGET",origin[0]+11,origin[1]-9);
    hitPoints=draft.keyframes.map(pose);
    hitPoints.forEach((p,i)=>{ctx.fillStyle=i===selected?"#e9b45c":"#848179";ctx.beginPath();ctx.arc(...p,i===selected?8:5,0,Math.PI*2);ctx.fill();ctx.fillStyle="#f1efe9";ctx.fillText(String(i+1),p[0]+10,p[1]-8);});
    const current=pose(samplePath(draft,playhead));line(current,origin,"#f2c577",1);
    ctx.fillStyle="#f1efe9";ctx.fillRect(current[0]-8,current[1]-6,16,12);
    ctx.fillStyle="#aeaba4";ctx.fillText("Opening camera",hitPoints[0][0]-45,hitPoints[0][1]+25);
    el("scrub").value=String(playhead);el("time").textContent=`${(playhead*duration).toFixed(2)} / ${duration.toFixed(2)}s`;
  }
  function stop() { playing=false;cancelAnimationFrame(raf);el("play").textContent="▶ Play";el("play").setAttribute("aria-label","Play path preview"); }
  function update() {
    const p=draft.keyframes[selected], last=draft.keyframes.length-1;
    el("keyframes").replaceChildren();
    draft.keyframes.forEach((k,i)=>{const b=document.createElement("button");b.type="button";b.textContent=`${i+1} · ${(k.time*duration).toFixed(2)}s`;b.className=i===selected?"active":"";b.setAttribute("aria-pressed",String(i===selected));b.onclick=()=>{stop();selected=i;playhead=k.time;update();};el("keyframes").append(b);});
    el("selected").textContent=selected===0?"Opening camera · locked":`Keyframe ${selected+1}`;
    el("delete").disabled=selected===0||selected===last;
    el("add").disabled=draft.keyframes.length>=12;
    el("point-time").value=(p.time*duration).toFixed(3);el("point-time").disabled=selected===0||selected===last;
    el("point-time").min=selected?String(draft.keyframes[selected-1].time*duration+.001):"0";
    el("point-time").max=selected<last?String(draft.keyframes[selected+1].time*duration-.001):String(duration);
    for(const [id,key] of [["orbit","azimuth"],["height","elevation"],["distance","distance"]]) for(const suffix of ["","-range"]){el(id+suffix).value=String(p[key]);el(id+suffix).disabled=selected===0;}
    el("lock-note").textContent=selected===0?"The opening camera stays at 0°, 0°, 1×. Select another keyframe to move it.":Math.abs(p.elevation)>30?"A steep height angle can move the horizon outside the frame.":"Height and distance are relative to the opening view. The target stays fixed.";
    el("status").textContent=cameraPathSummary(draft);
    const motion=draft.frame_mode==="motion";
    const problem=motion&&!whole?"Motion Frame needs a single beat covering the generation. Use a separate source for this camera pass.":motion&&!draft.reference_video?"Choose a reference video before applying Motion Frame.":"";
    el("apply").disabled=!!problem;
    el("diagnostics").replaceChildren(...[...(problem?[problem]:[]),...pathDiagnostics(draft,{whole,hasFirst,hasReferences,hasDifferentLast,duration})].map(text=>{const p=document.createElement("p");p.textContent=text;return p;}));
    draw();
  }
  el("preset").onchange=()=>{if(!el("preset").value)return;stop();draft.keyframes=newCameraPath(el("preset").value).keyframes;selected=1;playhead=1;el("preset").value="";update();};
  el("target").oninput=()=>{draft.target=el("target").value;};
  el("easing").onchange=()=>{draft.interpolation=el("easing").value;draw();};
  el("freeze").onchange=()=>{draft.freeze=el("freeze").checked;update();};
  el("mode").onchange=()=>{draft.frame_mode=el("mode").value;if(draft.frame_mode==="motion"){draft.freeze=false;el("freeze").checked=false;}syncReference();update();};
  el("video").onchange=()=>{draft.reference_video=el("video").value;delete draft.subject_box;syncReference();update();};
  el("upload").onclick=async()=>{el("upload").disabled=true;try {const uploaded=await uploadVideo();if(!dialog.isConnected||!uploaded)return;
    if(!Array.from(el("video").options).some(o=>o.value===uploaded.path))el("video").append(new Option(uploaded.label||uploaded.path,uploaded.path));
    draft.reference_video=uploaded.path;el("video").value=uploaded.path;delete draft.subject_box;syncReference();update();
  }catch(error){el("status").textContent=error.message;}finally{el("upload").disabled=false;}};
  el("loop").onchange=()=>{draft.loop_closure=el("loop").value;update();};
  el("experiment").onchange=()=>{draft.experiment=el("experiment").value;update();};
  const transform=fn=>{stop();try{draft=fn(draft);selected=Math.min(selected,draft.keyframes.length-1);playhead=draft.keyframes[selected].time;update();}catch(error){el("status").textContent=error.message;}};
  el("unwrap").onclick=()=>transform(unwrapPath);
  el("retime").onclick=()=>transform(redistributePath);
  el("holds").onclick=()=>transform(p=>addHolds(p,duration));
  el("close-orbit").onclick=()=>transform(p=>{const result=structuredClone(p),last=result.keyframes.at(-1);last.azimuth=last.azimuth<0?-360:360;last.elevation=0;last.distance=1;return result;});
  el("point-time").onchange=()=>{const value=el("point-time").valueAsNumber;const p=draft.keyframes[selected];if(Number.isFinite(value)&&value>Number(el("point-time").min)-.0001&&value<Number(el("point-time").max)+.0001)p.time=value/duration;playhead=p.time;update();};
  for(const [id,key] of [["orbit","azimuth"],["height","elevation"],["distance","distance"]]) for(const suffix of ["","-range"]){const input=el(id+suffix);input.addEventListener(suffix?"input":"change",()=>{stop();const value=input.valueAsNumber;if(Number.isFinite(value))draft.keyframes[selected][key]=Math.max(Number(input.min),Math.min(Number(input.max),value));playhead=draft.keyframes[selected].time;update();});}
  el("add").onclick=()=>{stop();if(draft.keyframes.length>=12)return;let i=1;for(let j=2;j<draft.keyframes.length;j++)if(draft.keyframes[j].time-draft.keyframes[j-1].time>draft.keyframes[i].time-draft.keyframes[i-1].time)i=j;const t=(draft.keyframes[i-1].time+draft.keyframes[i].time)/2;const p=samplePath(draft,t);draft.keyframes.splice(i,0,p);selected=i;playhead=t;update();};
  el("delete").onclick=()=>{if(selected===0||selected===draft.keyframes.length-1)return;stop();draft.keyframes.splice(selected,1);selected=Math.min(selected,draft.keyframes.length-1);playhead=draft.keyframes[selected].time;update();};
  el("scrub").oninput=()=>{stop();playhead=Number(el("scrub").value);draw();};
  el("play").onclick=()=>{if(playing){stop();return;}if(playhead>=1)playhead=0;playing=true;el("play").textContent="Ⅱ Pause";el("play").setAttribute("aria-label","Pause path preview");let previous=performance.now();const tick=now=>{playhead=Math.min(1,playhead+(now-previous)/(duration*1000));previous=now;draw();if(playhead>=1){stop();return;}raf=requestAnimationFrame(tick);};raf=requestAnimationFrame(tick);};
  let drag=null;
  canvas.onpointerdown=event=>{const r=canvas.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;const hit=hitPoints.findIndex(p=>Math.hypot(p[0]-x,p[1]-y)<16);if(hit>=0)selected=hit;stop();playhead=draft.keyframes[selected].time;update();if(selected===0)return;drag={x:event.clientX,y:event.clientY,point:{...draft.keyframes[selected]}};canvas.setPointerCapture(event.pointerId);};
  canvas.onpointermove=event=>{if(!drag)return;const p=draft.keyframes[selected];p.azimuth=Math.round(Math.max(-720,Math.min(720,drag.point.azimuth+(event.clientX-drag.x)*.6)));p.elevation=Math.round(Math.max(-89,Math.min(89,drag.point.elevation-(event.clientY-drag.y)*.3)));update();};
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  canvas.addEventListener("wheel",event=>{if(selected===0)return;event.preventDefault();stop();const p=draft.keyframes[selected];p.distance=Math.round(Math.max(.05,Math.min(4,p.distance+(event.deltaY>0?.05:-.05)))*100)/100;playhead=p.time;update();},{passive:false});
  const resize=new ResizeObserver(draw);resize.observe(canvas);
  dialog.addEventListener("keydown",event=>{if(!((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"))event.stopPropagation();});
  dialog.addEventListener("close",()=>{stop();resize.disconnect();dialog.remove();if(returnedFocus?.isConnected)returnedFocus.focus();},{once:true});
  el("close").onclick=el("cancel").onclick=()=>dialog.close();
  const commit=value=>{if(apply(value)!==false)dialog.close();else el("status").textContent="This shot changed while the planner was open. Close and reopen the planner before applying.";};
  el("apply").onclick=()=>commit(structuredClone(draft));el("remove").onclick=()=>commit(null);
  dialog.showModal();syncReference();update();
}
