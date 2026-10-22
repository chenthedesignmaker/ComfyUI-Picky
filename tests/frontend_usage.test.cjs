const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../js/standalone/usage.js'), 'utf8');
const moduleReady = import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
function memoryStorage() {
  const values = new Map();
  return { get length() { return values.size; }, key: i => [...values.keys()][i],
    getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
}
async function fixture(storage = memoryStorage()) {
  const { createUsageTracker } = await moduleReady;
  let time = 1000, serial = 0;
  const tracker = createUsageTracker({ storage, now: () => time, id: () => `test-${++serial}` });
  return { tracker, storage, advance: ms => time += ms };
}
test('navigation separates manual, contextual and initial visits without counting repeated active-tab clicks', async () => {
  const { tracker } = await fixture();
  tracker.visit('scene', 'initial'); tracker.visit('audio', 'manual'); tracker.visit('audio', 'manual');
  tracker.visit('scene', 'contextual');
  const [row] = tracker.records();
  assert.equal(row.panels.audio.visits.manual, 1);
  assert.equal(row.panels.scene.visits.manual, 0);
  assert.equal(row.panels.scene.visits.contextual, 1);
  assert.equal(row.panels.scene.visits.initial, 1);
  assert.deepEqual(row.transitions, { 'scene>audio:manual': 1, 'audio>scene:contextual': 1 });
});
test('active time stops at the idle threshold and excludes hidden-window time', async () => {
  const { tracker, advance } = await fixture();
  tracker.visit('scene', 'initial'); advance(90000); tracker.tick();
  tracker.visibility(false); advance(120000); tracker.tick();
  tracker.visibility(true); advance(10000); tracker.tick();
  assert.equal(tracker.records()[0].panels.scene.activeMs, 70000);
});
test('returning after 30 idle minutes starts a new session', async () => {
  const { tracker, advance } = await fixture();
  tracker.visit('scene', 'initial'); advance(31 * 60000); tracker.visit('clip', 'manual');
  assert.equal(tracker.records().length, 2);
});
test('disabled collection does not record actions, outcomes or dwell time', async () => {
  const { tracker, advance } = await fixture();
  tracker.visit('scene', 'initial'); tracker.setEnabled(false);
  advance(30000); tracker.visit('audio'); tracker.action('audio', 'field_change'); tracker.outcome('take_chosen');
  const [row] = tracker.records();
  assert.equal(row.panels.audio, undefined); assert.deepEqual(row.outcomes, {});
  assert.equal(row.panels.scene.activeMs, 0);
});
test('separate tabs preserve their counts and honor a shared reset and opt-out', async () => {
  const { createUsageTracker } = await moduleReady;
  const storage = memoryStorage();
  const a = createUsageTracker({ storage, now: () => 1000, id: () => 'a' });
  const b = createUsageTracker({ storage, now: () => 1001, id: () => 'b' });
  a.visit('scene', 'initial'); b.visit('audio');
  assert.equal(a.records().length, 2);
  a.reset(); b.flush();
  assert.equal(a.records().length, 0, 'Another tab must not resurrect reset data');
  a.setEnabled(false); b.action('audio', 'field_change'); b.outcome('take_chosen');
  assert.equal(b.records().length, 0);
});
test('reset preserves unrelated application preferences and project data', async () => {
  const { tracker, storage } = await fixture();
  storage.setItem('vp:lastProject', 'My project'); tracker.visit('scene', 'initial'); tracker.reset();
  assert.equal(tracker.records().length, 0);
  assert.equal(storage.getItem('vp:lastProject'), 'My project');
});
test('storage failure and malformed records do not prevent tracking or export', async () => {
  const { tracker } = await fixture({ getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); } });
  tracker.visit('audio'); tracker.action('audio', 'field_change');
  assert.equal(tracker.records()[0].panels.audio.actions.field_change, 1);
  assert.equal(tracker.durable, false);
  const other = await fixture(); other.storage.setItem('vp:usage:v1:damaged', '{bad');
  other.tracker.visit('scene', 'initial'); assert.equal(other.tracker.records().length, 1);
});
test('only allowlisted aggregate categories are persisted, with expired sessions removed', async () => {
  const { tracker, storage, advance } = await fixture();
  tracker.visit('private project name'); tracker.action('scene', 'secret prompt text'); tracker.outcome('/private/file');
  tracker.action('scene', 'text_edit'); tracker.outcome('render_submitted');
  let rows = tracker.records();
  assert.equal(rows[0].outcomes.render_submitted, 1);
  assert.ok(!JSON.stringify(rows).includes('private'));
  assert.ok(!JSON.stringify(rows).includes('secret'));
  advance(91 * 86400000); tracker.visit('audio'); rows = tracker.records();
  assert.equal(rows.length, 1); assert.equal(rows[0].panels.audio.visits.manual, 1);
});
