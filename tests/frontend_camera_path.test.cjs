const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const load = async () => import('data:text/javascript;base64,' + fs.readFileSync(path.join(__dirname,'../js/standalone/camera-path.js')).toString('base64'));

test('presets keep the opening pose fixed and do not share mutable keyframes', async () => {
  const {newCameraPath} = await load();
  const a = newCameraPath('circle'), b = newCameraPath('circle');
  assert.deepEqual(a.keyframes[0], {time:0,azimuth:0,elevation:0,distance:1});
  assert.equal(a.keyframes[1].azimuth,360);
  a.keyframes[1].azimuth=20;
  assert.equal(b.keyframes[1].azimuth,360);
});

test('playback respects endpoints, direction reversals and distance without overshoot', async () => {
  const {newCameraPath,samplePath} = await load();
  const p = newCameraPath();
  p.keyframes.splice(1,0,{time:.5,azimuth:120,elevation:15,distance:.5});
  for(const point of p.keyframes) assert.deepEqual(samplePath(p,point.time),point);
  for(let i=0;i<=100;i++) {
    const pose=samplePath(p,i/100);
    assert.ok(pose.azimuth>=0 && pose.azimuth<=120);
    assert.ok(pose.elevation>=0 && pose.elevation<=15);
    assert.ok(pose.distance>=.5 && pose.distance<=1);
  }
  p.interpolation='linear';
  assert.equal(samplePath(p,.25).azimuth,60);
});

test('timing redistribution preserves holds and whole turns survive unwrapping', async()=>{
  const {newCameraPath,addHolds,redistributePath,unwrapPath}=await load();
  const full=newCameraPath('circle');
  assert.equal(unwrapPath(full).keyframes.at(-1).azimuth,360);
  const p=newCameraPath();
  p.keyframes.splice(1,0,{time:.8,azimuth:30,elevation:0,distance:1});
  const timed=redistributePath(p);
  assert.ok(Math.abs(timed.keyframes[1].time-1/3)<1e-8);
  assert.equal(p.keyframes[1].time,.8);
  const held=redistributePath(addHolds(p,5));
  assert.equal(held.keyframes[1].time,.1);
  assert.ok(Math.abs(held.keyframes.at(-2).time-.9)<1e-8);
  assert.throws(()=>addHolds(p,1));
  const wrapping=newCameraPath();wrapping.keyframes.splice(1,0,{time:.5,azimuth:350,elevation:0,distance:1});wrapping.keyframes.at(-1).azimuth=10;
  assert.deepEqual(unwrapPath(wrapping).keyframes.map(p=>p.azimuth),[0,-10,10]);
});

test('uneven smooth segments keep velocity continuous, holds stay fixed',async()=>{
  const {newCameraPath,samplePath,addHolds}=await load();
  const p=newCameraPath();p.keyframes.splice(1,0,{time:.2,azimuth:10,elevation:0,distance:.8});
  const eps=1e-6, at=samplePath(p,.2).azimuth;
  assert.ok(Math.abs((at-samplePath(p,.2-eps).azimuth)/eps-(samplePath(p,.2+eps).azimuth-at)/eps)<.01);
  const held=addHolds(p,5);
  assert.equal(samplePath(held,.05).azimuth,0);
  assert.equal(samplePath(held,.95).azimuth,90);
});

test('loop diagnostic requires real first-image eligibility',async()=>{
  const {newCameraPath,pathDiagnostics}=await load();
  const p=newCameraPath('circle');p.freeze=true;p.loop_closure='auto';
  assert.match(pathDiagnostics(p,{hasFirst:true}).join(' '),/closure ON/);
  assert.match(pathDiagnostics(p).join(' '),/closure OFF/);
  p.frame_mode='motion';
  assert.match(pathDiagnostics(p,{hasFirst:true}).join(' '),/closure OFF/);
});
