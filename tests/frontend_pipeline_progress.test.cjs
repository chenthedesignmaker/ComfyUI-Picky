const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

test('pipeline-progress exports node stage mapping, weighted progress, and stageMarkup', async () => {
  const mod = await import(pathToFileURL(path.join(__dirname, '../js/standalone/pipeline-progress.js')).href);
  const {
    getStageDescription,
    getNodeStage,
    resolveWorkflowNodeId,
    calculateOverallProgress,
    getJobStageInfo,
    stageMarkup,
  } = mod;

  const workflow = {
    '1': { class_type: 'CLIPTextEncode', _meta: { title: 'Positive Prompt' }, inputs: { text: 'hello' } },
    '2': { class_type: 'SamplerCustomAdvanced', _meta: { title: 'Sampler' }, inputs: { positive: ['1', 0] } },
    '3': { class_type: 'VAEDecode', _meta: { title: 'VAE Decode' }, inputs: { samples: ['2', 0] } },
    '4': { class_type: 'SaveVideo', _meta: { title: 'Save Video' }, inputs: { images: ['3', 0] } },
  };

  assert.equal(getStageDescription(workflow, '1'), 'Encoding prompt and conditioning');
  assert.deepEqual(getNodeStage(workflow, '2'), {
    key: 'sampling',
    label: 'Generating motion and audio',
    detail: 'Sampler',
  });
  assert.equal(getNodeStage(workflow, '4').key, 'save');
  assert.equal(resolveWorkflowNodeId(workflow, '999', '2'), '2');

  const job = {
    workflow,
    currentNode: '2',
    progress: { node: '2', value: 10, max: 20 },
  };

  // During sampling at 50% step completion
  const pct = calculateOverallProgress(job, '2', { value: 10, max: 20 });
  assert.ok(typeof pct === 'number' && pct > 10 && pct < 90, `Expected weighted progress in range, got ${pct}`);

  const info = getJobStageInfo(job);
  assert.equal(info.key, 'sampling');
  assert.equal(info.label, 'Generating motion and audio');
  assert.equal(info.metric, 'Step 10/20 (50%)');
  assert.ok(info.overallPct > 0);

  const html = stageMarkup(info);
  assert.ok(html.includes('Generating motion and audio'));
  assert.ok(html.includes('Step 10/20 (50%)'));
});
