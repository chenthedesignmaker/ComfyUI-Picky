const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '../js/standalone/main.js'), 'utf8');
const functions = script.slice(script.indexOf('async function copyTakeContext('), script.indexOf('function candidateCard('));

function harness(clipboard) {
  const notices = [], fields = [], requests = [];
  const context = {
    state: { project: { name: 'Film & one' }, dirty: true },
    currentClip: () => ({ id: 'clip-1', source_id: 'shot-1' }),
    API: '/video_prompter', URL, URLSearchParams,
    window: { location: { href: 'http://localhost:8188/video_prompter/app' } },
    navigator: { clipboard },
    document: {
      activeElement: { focus() {} },
      createElement() {
        const field = { style: {}, setAttribute() {}, select() {}, remove() { this.removed = true; } };
        fields.push(field);
        return field;
      },
      body: { append() {} },
      execCommand: () => true,
    },
    toast: text => notices.push(text),
    post: async (url, body) => {
      requests.push({ url, body });
      return { take: { id: body.take, seed: 0 }, notes: [] };
    },
  };
  vm.createContext(context);
  vm.runInContext(functions, context);
  return { context, notices, fields, requests };
}

test('copies the clicked take and original project despite selection changing during fetch', async () => {
  let copied, finish;
  const h = harness({ writeText: async text => { copied = text; } });
  h.context.post = async (url, body) => {
    h.requests.push({ url, body });
    return new Promise(resolve => { finish = resolve; });
  };
  const pending = h.context.copyTakeContext({ id: 'exact:take.mp4' }, { id: 'shot-1' });
  h.context.state.project.name = 'Different film';
  assert.equal(h.notices.length, 0);
  finish({ take: { id: 'exact:take.mp4', seed: 0 }, notes: [] });
  await pending;
  const data = JSON.parse(copied.slice(copied.indexOf('{')));
  assert.equal(h.requests[0].body.project, 'Film & one');
  assert.equal(h.requests[0].body.clip, 'clip-1');
  assert.equal(data.take.id, 'exact:take.mp4');
  assert.equal(new URL(data.lookup.gallery_url).searchParams.get('project'), 'Film & one');
  assert.match(data.notes[0], /unsaved changes/);
  assert.equal(h.notices.length, 1);
});

test('clipboard denial falls back and removes the temporary field', async () => {
  const h = harness({ writeText: async () => { throw new Error('denied'); } });
  await h.context.copyTakeContext({ id: 'take' }, { id: 'shot-1' });
  assert.match(h.fields[0].value, /generation context/);
  assert.equal(h.fields[0].removed, true);
  assert.equal(h.notices.length, 1);
});

test('failed copy cleans up and never claims success', async () => {
  const h = harness(undefined);
  h.context.document.execCommand = () => false;
  await assert.rejects(h.context.copyTakeContext({ id: 'take' }, { id: 'shot-1' }), /Copy was blocked/);
  assert.equal(h.fields[0].removed, true);
  assert.equal(h.notices.length, 0);
});

test('server failure never copies incomplete context', async () => {
  const h = harness({ writeText: () => assert.fail('must not copy') });
  h.context.post = async () => { throw new Error('Take removed'); };
  await assert.rejects(h.context.copyTakeContext({ id: 'take' }, { id: 'shot-1' }), /Take removed/);
  assert.equal(h.notices.length, 0);
});

test('an older running server gives restart instructions instead of a bare 405', async () => {
  const h = harness({ writeText: () => assert.fail('must not copy') });
  h.context.post = async () => { throw Object.assign(new Error('405: Method Not Allowed'), { status: 405 }); };
  await assert.rejects(h.context.copyTakeContext({ id: 'take' }, { id: 'shot-1' }), /Restart ComfyUI/);
  assert.equal(h.notices.length, 0);
});
