const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const html = fs.readFileSync(path.join(__dirname, '../js/standalone/index.html'), 'utf8');

test('index.html contains the full-screen compare modal and all required controls', () => {
  assert.ok(html.includes('id="compare-modal"'), 'Missing #compare-modal element');
  assert.ok(html.includes('id="compare-stage"'), 'Missing #compare-stage element');
  assert.ok(html.includes('id="compare-swiper"'), 'Missing #compare-swiper element');
  assert.ok(html.includes('id="compare-divider"'), 'Missing #compare-divider element');
  assert.ok(html.includes('id="compare-handle"'), 'Missing #compare-handle element');
  assert.ok(html.includes('id="cmp-mode-swiper"'), 'Missing #cmp-mode-swiper button');
  assert.ok(html.includes('id="cmp-mode-grid"'), 'Missing #cmp-mode-grid button');
  assert.ok(html.includes('id="cmp-mode-diff"'), 'Missing #cmp-mode-diff button');
  assert.ok(html.includes('id="cmp-roi-btn"'), 'Missing #cmp-roi-btn button');
  assert.ok(html.includes('id="cmp-roi-reset"'), 'Missing #cmp-roi-reset button');
  assert.ok(html.includes('id="compare-minimap"'), 'Missing #compare-minimap element');
  assert.ok(html.includes('id="cmp-minimap-roi"'), 'Missing #cmp-minimap-roi element');
  assert.ok(html.includes('id="compare-close"'), 'Missing #compare-close button');
  assert.ok(html.includes('id="cmp-play"'), 'Missing #cmp-play button');
  assert.ok(html.includes('id="cmp-scrub"'), 'Missing #cmp-scrub scrubber');
});

test('compareState tracks swiper, takes, playback, and ROI zoom accurately', async () => {
  const elements = new Map();
  function makeEl(id) {
    const el = {
      id,
      style: {},
      classList: {
        _classes: new Set(),
        add(c) { this._classes.add(c); },
        remove(c) { this._classes.delete(c); },
        toggle(c, v) { if (v) this._classes.add(c); else this._classes.delete(c); return this._classes.has(c); },
        contains(c) { return this._classes.has(c); },
      },
      querySelectorAll() { return []; },
      append() {},
      pause() {},
      play() { return Promise.resolve(); },
      innerHTML: '',
      textContent: '',
      hidden: false,
      value: '',
      options: [],
      matches() { return false; },
    };
    elements.set(id, el);
    return el;
  }

  const $ = (id) => elements.get(id) || makeEl(id);

  globalThis.document = {
    createElement: (tag) => makeEl(tag),
    querySelectorAll: () => [],
    fullscreenElement: null,
    exitFullscreen: async () => {},
  };
  globalThis.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
  globalThis.requestAnimationFrame = () => 1;
  globalThis.cancelAnimationFrame = () => {};

  const compareMod = await import(pathToFileURL(path.join(__dirname, '../js/standalone/compare-modal.js')).href);
  const {
    compareState,
    configureCompareModal,
    openCompareModal,
    updateSplitDivider,
    setCompareMode,
    applyRoiTransform,
    resetRoiZoom,
    closeCompareModal,
  } = compareMod;

  const mockState = { player: null, depth: 0, selectedClip: 'c1', compare: new Set() };
  configureCompareModal({
    $,
    getState: () => mockState,
    currentClip: () => ({ source_in: 0 }),
    clipReport: () => ({ source_out: 5 }),
    currentSource: () => ({ id: 'src1', name: 'Scene 1' }),
    pathOf: () => ['take-1'],
    stopPlayer: () => {},
    label: (icon, text) => text,
    iconHTML: () => '',
    ratingOf: () => 0,
    ratingLabel: () => '',
    guard: (fn) => fn,
    toast: () => {},
    setPath: async () => {},
    renderCandidates: () => {},
    takeById: () => null,
  });

  const take1 = { id: 'take-1', url: 't1.mp4', seed: 100, filename: 'take_1.mp4' };
  const take2 = { id: 'take-2', url: 't2.mp4', seed: 200, filename: 'take_2.mp4' };

  // Test opening compare modal
  openCompareModal([take1, take2]);
  assert.equal(compareState.open, true);
  assert.equal(compareState.takeA.id, 'take-1');
  assert.equal(compareState.takeB.id, 'take-2');
  assert.equal(compareState.mode, 'swiper');
  assert.equal(compareState.splitPercent, 50);

  // Test updating swiper divider
  updateSplitDivider(75);
  assert.equal(compareState.splitPercent, 75);
  assert.equal($('compare-divider').style.left, '75%');
  assert.equal($('cmp-layer-b').style.clipPath, 'inset(0 0 0 75%)');

  // Test switching mode to grid
  setCompareMode('grid');
  assert.equal(compareState.mode, 'grid');
  assert.equal($('compare-swiper').hidden, true);
  assert.equal($('compare-grid').hidden, false);

  // Test ROI transform calculations
  const dummyRoi = { x: 0.2, y: 0.3, w: 0.25, h: 0.25, scale: 4, cx: 0.325, cy: 0.425 };
  applyRoiTransform(dummyRoi);
  assert.equal($('cmp-roi-reset').hidden, false);
  assert.ok($('cmp-roi-zoom-level').textContent.includes('4.0×'));

  // Test ROI zoom reset
  resetRoiZoom();
  assert.equal(compareState.roi, null);
  assert.equal($('cmp-roi-reset').hidden, true);

  // Test closing compare modal
  closeCompareModal();
  assert.equal(compareState.open, false);
});
