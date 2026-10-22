const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '../js/standalone/main.js'), 'utf8');
function section(start, end) {
  const from = script.indexOf(start);
  const to = script.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `Missing application function: ${start}`);
  return script.slice(from, to);
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('a prompt link reveals its folded shot and voice details before focusing the field', () => {
  const events = [];
  const panel = { dataset: { panel: 'scene' } };
  const details = { tagName: 'DETAILS', open: false, parentElement: panel };
  const target = {
    parentElement: { tagName: 'DIV', parentElement: details },
    closest: selector => selector === '.tabpanel' ? panel : { dataset: { beat: '2' } },
    scrollIntoView() { assert.equal(details.open, true); events.push('scroll'); },
    focus() { events.push('focus'); },
    classList: { remove() {}, add() {} },
  };
  const context = {
    findField: () => target,
    $: () => ({ querySelector: () => ({ click: () => events.push('scene') }) }),
    setSceneShotOpen: (index, open) => {
      assert.equal(index, 2);
      assert.equal(open, true);
      events.push('shot');
    },
    refitGrowFields: () => events.push('fit'),
  };
  vm.createContext(context);
  vm.runInContext(section('function focusField(address)', '/** `screen_text.0`'), context);
  context.focusField('line.2.0.delivery');
  assert.deepEqual(events, ['scene', 'shot', 'fit', 'scroll', 'focus']);
});

test('a delayed take preview cannot replace the monitor after changing sources', async () => {
  let resolve;
  const played = [];
  const context = {
    state: {projectEpoch: 1, selectedSource: 'a'},
    lineageTo: take => [take.id],
    resolveLineage: () => new Promise(done => { resolve = done; }),
    playPreview: (...args) => played.push(args),
  };
  vm.createContext(context);
  vm.runInContext(section('async function playSegment(take)', '/** Ask the server which files'), context);
  const stale = context.playSegment({id: 'take-a'});
  context.state.selectedSource = 'b';
  resolve([{id: 'take-a', start: 8}]);
  await stale;
  assert.equal(played.length, 0);
  const current = context.playSegment({id: 'take-b'});
  resolve([{id: 'take-b', start: 8}]);
  await current;
  assert.equal(played.length, 1);
  assert.equal(played[0][0][0].start, 0);
  vm.runInContext(section('function ensureTimeline()', '// The chain ending'), context);
  const cancelled = context.playSegment({id: 'take-b'});
  context.ensureTimeline();
  resolve([{id: 'take-b', start: 8}]);
  await cancelled;
  assert.equal(played.length, 1, 'Returning to the timeline cancels a pending preview');
});

function editor() {
  const state = {
    project: {name: 'Audit', sources: [{id: 's', text: 'Before save'}]},
    selectedSource: 's', editorOpen: true, dirty: true,
    editRevision: 1, projectEpoch: 1, sequenceRequest: 0,
    jobs: new Map([['job', {}]]),
  };
  const requests = [], notices = [];
  const context = {
    state, API: '/video_prompter', structuredClone,
    $: () => ({classList: {add() {}, remove() {}}}),
    currentSource: () => state.project?.sources.find(s => s.id === state.selectedSource),
    readForm() {}, restoreSelection() {}, keepingCaret(fn) { fn(); }, fillForm() {},
    buildPlaylist() {}, renderBin() {}, renderTimeline() {}, renderCandidates() {}, seekTo() {},
    updateReviewContext() {},
    toast: (...args) => notices.push(args), syncQueue() {},
    post(url, payload) {
      const project = structuredClone(state.project);
      project.sources = [structuredClone(payload.source)];
      return new Promise((resolve, reject) => requests.push({
        url, payload, resolve: () => resolve({project}), reject,
      }));
    },
  };
  vm.createContext(context);
  vm.runInContext(section('function markDirty()', '// -- model'), context);
  vm.runInContext(section('function adoptProject(project)', '// `fillForm` throws'), context);
  vm.runInContext(section('async function reloadSequence()', '// Local edits live'), context);
  vm.runInContext(section('let sourceSaveQueue', 'async function save()'), context);
  return {state, requests, notices, context};
}

test('typing during save survives and is saved before a waiting switch can proceed', async () => {
  const {state, requests, context} = editor();
  const saving = context.pushSource();
  await tick();
  state.project.sources[0].text = 'Typed during save';
  context.markDirty();
  requests.shift().resolve();
  await tick();
  assert.equal(state.project.sources[0].text, 'Typed during save');
  assert.equal(state.dirty, true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].payload.source.text, 'Typed during save');
  requests.shift().resolve();
  await saving;
  assert.equal(state.project.sources[0].text, 'Typed during save');
  assert.equal(state.dirty, false);
});

test('overlapping saves serialize and a queued save reads the latest source', async () => {
  const {state, requests, context} = editor();
  const first = context.pushSource();
  await tick();
  const second = context.pushSource();
  await tick();
  assert.equal(requests.length, 1);
  requests.shift().resolve();
  await first;
  await tick();
  assert.equal(requests.length, 1);
  requests.shift().resolve();
  await second;
  assert.equal(state.dirty, false);
});

test('saving the docked inspector reads its draft while the expanded editor is closed', async () => {
  const {state, requests, context} = editor();
  state.editorOpen = false;
  state.inspectorOpen = true;
  context.readForm = () => { state.project.sources[0].text = 'Inspector draft'; };
  const saving = context.pushSource();
  await tick();
  assert.equal(requests[0].payload.source.text, 'Inspector draft');
  requests.shift().resolve();
  await saving;
  assert.equal(state.dirty, false);
});

test('a server refresh rebinds docked inspector fields to the new source objects', () => {
  const {state, context} = editor();
  state.editorOpen = false;
  state.inspectorOpen = true;
  let boundSource = null;
  context.keepingCaret = redraw => redraw();
  context.fillForm = () => { boundSource = state.project.sources[0]; };
  const refreshed = {name: 'Audit', sources: [{id: 's', text: 'Fresh'}]};
  context.adoptProject(refreshed);
  assert.equal(boundSource, refreshed.sources[0]);
});

test('save responses cannot replace a different project or clear its dirty flag', async () => {
  const {state, requests, context} = editor();
  const saving = context.pushSource();
  await tick();
  state.projectEpoch++;
  state.project = {name: 'Other project', sources: [{id: 's', text: 'Other draft'}]};
  context.markDirty();
  requests.shift().resolve();
  await saving;
  assert.equal(state.project.name, 'Other project');
  assert.equal(state.project.sources[0].text, 'Other draft');
  assert.equal(state.dirty, true);
});

test('failed saves retain the draft and do not poison later retries', async () => {
  const {state, requests, context} = editor();
  const saving = context.pushSource();
  await tick();
  requests.shift().reject(new Error('offline'));
  await assert.rejects(saving, /offline/);
  assert.equal(state.dirty, true);
  const retry = context.pushSource();
  await tick();
  requests.shift().resolve();
  await retry;
  assert.equal(state.dirty, false);
});

test('background refresh cannot overwrite edits made while its reads are pending', async () => {
  const {state, context} = editor();
  state.dirty = false;
  const reads = [];
  context.call = url => new Promise(resolve => reads.push({url, resolve}));
  const reload = context.reloadSequence();
  assert.equal(reads.length, 3);
  state.project.sources[0].text = 'Typed during refresh';
  context.markDirty();
  reads[0].resolve({stale: true});
  reads[1].resolve({takes: []});
  reads[2].resolve({project: {name: 'Audit', sources: [{id: 's', text: 'Stale'}]}});
  await reload;
  assert.equal(state.project.sources[0].text, 'Typed during refresh');
  assert.equal(state.report, undefined);
  assert.equal(state.dirty, true);
});

test('an old project refresh cannot replace the current project', async () => {
  const {state, context} = editor();
  state.dirty = false;
  const reads = [];
  context.call = url => new Promise(resolve => reads.push(resolve));
  const reload = context.reloadSequence();
  state.projectEpoch++;
  state.project = {name: 'Other', sources: []};
  reads[0]({stale: true});
  reads[1]({takes: []});
  reads[2]({project: {name: 'Audit', sources: []}});
  await reload;
  assert.equal(state.project.name, 'Other');
  assert.equal(state.report, undefined);
});

test('queue deletion does not claim success or remove local jobs after an HTTP error', async () => {
  const {state, context, notices} = editor();
  context.fetch = async () => ({ok: false, status: 500, json: async () => ({error: 'queue failed'})});
  vm.runInContext(section('async function call(path', '// Every write carries'), context);
  vm.runInContext(section('async function deleteQueuedJob', 'async function clearPendingQueue'), context);
  await context.deleteQueuedJob('job');
  assert.equal(state.jobs.has('job'), true);
  assert.equal(notices.length, 1);
  assert.match(notices[0][0], /Could not delete queued job: queue failed/);
  assert.equal(notices[0][1], true);
});

test('refresh hides stored input previews even when an older server returns them', async () => {
  const {state, context} = editor();
  state.dirty = false;
  context.call = async url => url.includes('/sequence') ? {}
    : url.includes('/gallery') ? {takes: [
      {id: 'result', type: 'output', seed: 42},
      {id: 'input-preview', type: 'input', seed: 42},
      {id: 'temp-render', type: 'temp', seed: 42},
    ]} : {project: state.project};
  await context.reloadSequence();
  assert.deepEqual(state.takes.map(take => take.id), ['result', 'temp-render']);
});

test('queue history chooses the generated video when the input preview appears first', () => {
  const {context} = editor();
  context.sourceById = () => null;
  vm.runInContext(section('function extractJobFromHistoryEntry', 'function getCombinedHistory'), context);
  const job = context.extractJobFromHistoryEntry('extend', {
    prompt: [1, 'extend', {}, {video_prompter: {seed: 42, mode: 'Extend'}}],
    outputs: {
      vp_extend_src: {images: [{filename: 'parent.mp4', type: 'input'}]},
      save: {images: [{filename: 'new.mp4', type: 'output'}]},
    }, status: {status_str: 'success'},
  });
  assert.equal(job.filename, 'new.mp4');
  assert.equal(job.type, 'output');
  assert.equal(job.seed, 42);
  assert.match(job.url, /filename=new.mp4/);
});
