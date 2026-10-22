const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const code = fs.readFileSync(path.join(__dirname, '../js/standalone/main.js'), 'utf8');
function section(start, end) { return code.slice(code.indexOf(start), code.indexOf(end, code.indexOf(start))); }

test('picker hides unrelated files from an older running server', () => {
  const ctx={};vm.createContext(ctx);
  vm.runInContext(section('function videoLoraPickerNames(', 'async function loadVideoLoraFiles('),ctx);
  const people='h3-realism-people-t2v-i2v-r2v.safetensors';
  assert.deepEqual(Array.from(ctx.videoLoraPickerNames({loras:['krea2.safetensors','h3-unknown.safetensors','turbo.safetensors',people]})),[people]);
  assert.deepEqual(Array.from(ctx.videoLoraPickerNames({model_family:'minimax-h3',loras:['renamed-character.safetensors']})),['renamed-character.safetensors']);
});

test('queued payload records the prepared stack, including a deliberate empty stack', () => {
  const ctx = {state:{clientId:'client',project:{name:'Test'}},uuid:()=> 'id'};
  vm.createContext(ctx);
  vm.runInContext(section('function buildPromptPayload(', 'async function holdPreparedRender('),ctx);
  for(const stack of [[],[{name:'people.safetensors',strength_model:0.6,enabled:true}]]) {
    const payload=ctx.buildPromptPayload({workflow:{},loras:stack},{id:'source'},'t2v');
    assert.deepEqual(payload.extra_data.video_prompter.loras,stack);
  }
});

test('render preparation waits for pending LoRA settings to save', async () => {
  let release; const calls=[];
  const ctx={videoLoraSave:new Promise(resolve=>{release=resolve;}),post:async(...args)=>{calls.push(args);return {ok:true};},toast:()=>{},showFindings:()=>{}};
  vm.createContext(ctx);
  vm.runInContext(section('async function preparePost(', '// Save the source, so a take'),ctx);
  const pending=ctx.preparePost('/render/prepare',{project:'Test'});
  await Promise.resolve();assert.equal(calls.length,0);
  release();await pending;assert.equal(calls.length,1);
});

test('fresh, batch, edited-prompt and reshoot requests identify their own shot', async () => {
  const calls=[];
  const ctx={API:'/api',state:{project:{name:'Test'},seedPin:null},
    preparePost:async(path,body)=>{calls.push({path,body});return path.endsWith('/batch')?{renders:[{}]}:{};},
    $:()=>({value:'profile'}),queueTake:async()=>{},QUALITY_LABELS:{}};
  vm.createContext(ctx);
  vm.runInContext(section('async function prepareRenders(', '// Refine into a new take'),ctx);
  const target={id:'shot-two',prompt:{width:1344,height:736}};
  await ctx.prepareRenders(target,'profile','final',1);
  await ctx.prepareRenders(target,'profile','final',2);
  ctx.state.previewDirty=true;ctx.state.previewEditedText='Changed words';
  await ctx.prepareRenders(target,'profile','final',1);
  await ctx.reshoot({prompt_text:'Frozen text',frames:124,seconds:5},target,{quality:'final'});
  assert.equal(calls.length,4);
  for(const {body} of calls) assert.equal(body.source_id,'shot-two');
});

test('a failed LoRA save prevents preparation', async () => {
  let called=false;
  const ctx={videoLoraSave:Promise.reject(new Error('Save failed')),post:async()=>{called=true;},toast:()=>{},showFindings:()=>{}};
  vm.createContext(ctx);
  vm.runInContext(section('async function preparePost(', '// Save the source, so a take'),ctx);
  assert.equal(await ctx.preparePost('/render/prepare',{}),null);
  assert.equal(called,false);
});
