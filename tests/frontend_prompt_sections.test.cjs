const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

test('prompt-sections module parses H3 sections and diffs words accurately', async () => {
  const modUrl = pathToFileURL(path.join(__dirname, '../js/standalone/prompt-sections.js')).href;
  const {
    PROMPT_FIELDS,
    promptSections,
    diffWords,
  } = await import(modUrl);

  assert.ok(Array.isArray(PROMPT_FIELDS) && PROMPT_FIELDS.length >= 10);

  const raw = [
    'integrated_multimodal_description: [Shot 1] Wide shot of an office cubicle. [Shot 2] Close-up on keyboard.',
    'overall_soundscape: Quiet fluorescent hum and keyboard clicks.',
    'summary: A quiet office sequence.',
  ].join('\n');

  const sections = promptSections(raw);
  assert.ok(Array.isArray(sections), 'Expected sections array');
  assert.equal(sections.length, 4);
  assert.equal(sections[0].label, 'Shot 1');
  assert.equal(sections[1].label, 'Shot 2');
  assert.equal(sections[2].label, 'Soundscape');
  assert.equal(sections[3].label, 'Summary');

  const diff = diffWords('Wide shot of cubicle', 'Medium shot of cubicle');
  assert.deepEqual(diff, [
    { type: 'del', text: 'Wide' },
    { type: 'add', text: 'Medium' },
    { type: 'same', text: ' shot of cubicle' },
  ]);
});
