const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const data = fs.readFileSync(path.join(__dirname, '../js/standalone/shot-ideas-data.js'), 'utf8');
const library = import(`data:text/javascript;base64,${Buffer.from(data).toString('base64')}`);

const beat = () => ({
  start: 0, end: 4, cut: false, transition: '', angle: 'A two-shot',
  action: 'She puts the mug down.\nHe waits.', camera: 'The camera tracks left.',
  lines: [{ speakers: ['anna'], text: 'Please stay.', delivery: 'softly' }],
});

test('applying an approach preserves the story, dialogue and timing; repetition adds no duplicate light/color', async () => {
  const { FEELINGS, QUALITIES, buildIdeaPatch } = await library;
  for (const emotion of FEELINGS) for (const approach of emotion.approaches) {
    const before = beat();
    const patch = buildIdeaPatch(before, approach, QUALITIES.map(([key]) => key));
    const after = { ...before, ...patch };
    assert.deepEqual(after.lines, before.lines);
    assert.equal(after.cut, false);
    assert.equal(after.start, before.start);
    assert.equal(after.end, before.end);
    assert.ok(after.action.startsWith(before.action + '\n\n'));
    assert.deepEqual(buildIdeaPatch(after, approach, QUALITIES.map(([key]) => key)), {});
    assert.deepEqual(before, beat(), 'Preview must not mutate the beat');
  }
});

test('camera-only selection leaves existing framing and environment untouched', async () => {
  const { FEELINGS, buildIdeaPatch, directionText } = await library;
  const approach = FEELINGS[0].approaches[0];
  assert.deepEqual(buildIdeaPatch(beat(), approach, ['camera']), { camera: approach.camera });
  assert.equal(directionText(approach, ['camera']), approach.camera);
  assert.deepEqual(buildIdeaPatch(beat(), approach, []), {});
});

test('reference context stays inside the search parameter, including punctuation and Unicode', async () => {
  const { referenceURL } = await library;
  const query = 'anger café & office #2 / "rain" ?a=1';
  const url = new URL(referenceURL(query));
  assert.equal(url.origin, 'https://frameset.app');
  assert.equal(url.pathname, '/search');
  assert.equal(url.searchParams.get('search'), query);
  assert.equal([...url.searchParams].length, 1);
  assert.equal(url.hash, '');
});

test('actual editor commit rejects stale/reordered/different-source previews and edits only the chosen beat', async () => {
  const { previewMatches } = await library;
  const prompt = { beats: [beat(), { ...beat(), action: 'He leaves.' }] };
  let key = 'project/source/generation';
  let dirty = 0;
  const snapshot = () => ({ key, beats: structuredClone(prompt.beats) });
  const context = {
    previewMatches, shotIdeasSnapshot: snapshot, currentPrompt: () => prompt,
    sceneOpenShots: () => new Set(), markDirty: () => dirty++, renderBeats() {}, refresh() {},
  };
  const main = fs.readFileSync(path.join(__dirname, '../js/standalone/main.js'), 'utf8');
  vm.createContext(context);
  vm.runInContext(main.slice(main.indexOf('function commitShotIdea('), main.indexOf('function sceneOpenShots(')), context);
  const makePreview = () => ({ key, beats: JSON.stringify(prompt.beats), index: 0, patch: { camera: 'New direction' } });
  const pending = makePreview();
  prompt.beats.reverse();
  assert.equal(context.commitShotIdea(pending), false);
  prompt.beats.reverse();
  key = 'another-source';
  assert.equal(context.commitShotIdea(pending), false);
  key = pending.key;
  prompt.beats[0].lines[0].text = 'New dialogue';
  assert.equal(context.commitShotIdea(pending), false);
  assert.equal(dirty, 0);
  const fresh = makePreview();
  assert.equal(context.commitShotIdea({ ...fresh, patch: { lines: [] } }), false);
  const otherBeat = structuredClone(prompt.beats[1]);
  assert.equal(context.commitShotIdea(fresh), true);
  assert.equal(dirty, 1);
  assert.equal(prompt.beats[0].camera, 'New direction');
  assert.equal(prompt.beats[0].lines[0].text, 'New dialogue');
  assert.deepEqual(prompt.beats[1], otherBeat);
});
