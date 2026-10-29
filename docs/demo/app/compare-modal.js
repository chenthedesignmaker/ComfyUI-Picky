// Full-screen take comparison workspace:
// - Swiper mode: interactive draggable A/B split wipe divider
// - Side-by-Side grid mode: multi-column comparison of 2-4 takes
// - Difference mode: blend overlay & flicker to expose pixel differences
// - ROI Detail Box zoom: draw a box to zoom into fine details (face, hands, textures)
//   with synchronized magnification and a Picture-in-Picture Mini-Map navigator.

const deps = {
  $: (id) => document.getElementById(id),
  getState: () => ({}),
  currentClip: () => null,
  clipReport: () => null,
  currentSource: () => null,
  pathOf: () => [],
  stopPlayer: () => {},
  label: (_icon, text) => text,
  iconHTML: () => "",
  ratingOf: () => 0,
  ratingLabel: () => "",
  guard: (fn) => fn,
  toast: () => {},
  setPath: async () => {},
  renderCandidates: () => {},
  takeById: () => null,
};

export function configureCompareModal(nextDeps = {}) {
  Object.assign(deps, nextDeps);
}

export const compareState = {
  open: false,
  mode: "swiper", // "swiper" | "grid" | "diff"
  takes: [],
  takeA: null,
  takeB: null,
  splitPercent: 50,
  isPlaying: false,
  speed: 1,
  loop: true,
  muted: true,
  drawingRoi: false,
  roi: null, // { x, y, w, h, scale, cx, cy }
  isDraggingDivider: false,
  isPanningMiniMap: false,
  syncHandle: null,
  from: 0,
  to: null,
};

export function allCompareVideos() {
  const $ = deps.$;
  const modal = $("compare-modal");
  if (!modal) return [];
  const list = [
    $("cmp-vid-a"),
    $("cmp-vid-b"),
    $("cmp-diff-vid-a"),
    $("cmp-diff-vid-b"),
    ...(modal.querySelectorAll?.(".cmp-card video") || []),
  ].filter(Boolean);
  return [...new Set(list)];
}

export function activeCompareVideos() {
  const $ = deps.$;
  if (compareState.mode === "swiper") {
    return [$("cmp-vid-a"), $("cmp-vid-b")].filter(Boolean);
  }
  if (compareState.mode === "grid") {
    return [...($("compare-grid")?.querySelectorAll?.("video") || [])];
  }
  if (compareState.mode === "diff") {
    return [$("cmp-diff-vid-a"), $("cmp-diff-vid-b")].filter(Boolean);
  }
  return [];
}

export function openCompareModal(chosenTakes) {
  if (!chosenTakes || chosenTakes.length < 2) return;
  const $ = deps.$;
  const state = deps.getState() || {};

  if (state.player?.playing) deps.stopPlayer();
  if (typeof document !== "undefined" && document.querySelectorAll) {
    for (const node of document.querySelectorAll("video, audio")) {
      if (!node.paused && !node.closest?.("#compare-modal")) node.pause();
    }
  }

  compareState.open = true;
  compareState.takes = [...chosenTakes];
  compareState.takeA = chosenTakes[0];
  compareState.takeB = chosenTakes[1] || chosenTakes[0];
  compareState.splitPercent = 50;
  compareState.isPlaying = false;
  compareState.drawingRoi = false;
  compareState.roi = null;

  const clip = deps.currentClip();
  compareState.from = clip?.source_in || 0;
  compareState.to = deps.clipReport(state.selectedClip)?.source_out ?? null;

  const source = deps.currentSource();
  const titleEl = $("compare-source-badge");
  if (titleEl) titleEl.textContent = source?.name ? `${source.name} (Gen ${(state.depth ?? 0) + 1})` : "";
  const countEl = $("compare-count-badge");
  if (countEl) countEl.textContent = `${chosenTakes.length} takes`;

  populateCompareSelects();
  resetRoiZoom();
  updateSplitDivider(50);
  setCompareMode(compareState.mode || "swiper");

  const modal = $("compare-modal");
  if (modal) {
    modal.hidden = false;
    modal.classList.add("open");
  }

  if ($("cmp-speed")) $("cmp-speed").value = String(compareState.speed || 1);
  if ($("cmp-loop")) $("cmp-loop").classList.toggle("active", compareState.loop);
  updateComparePlayButton();
  updateCompareSoundButton();
  updateCompareTransportUI(compareState.from, 0);
}

export function closeCompareModal() {
  const $ = deps.$;
  compareState.open = false;
  stopCompareSync();
  allCompareVideos().forEach((v) => {
    if (typeof v.pause === "function") v.pause();
  });
  resetRoiZoom();

  const modal = $("compare-modal");
  if (modal) {
    modal.hidden = true;
    modal.classList.remove("open");
  }
  if (typeof document !== "undefined" && document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
}

export function populateCompareSelects() {
  const $ = deps.$;
  const selA = $("cmp-select-a");
  const selB = $("cmp-select-b");
  if (!selA || !selB) return;
  selA.innerHTML = "";
  selB.innerHTML = "";

  const OptionCtor = typeof Option !== "undefined"
    ? Option
    : class { constructor(text, value) { this.text = text; this.value = value; } };

  compareState.takes.forEach((take) => {
    const labelText = take.seed != null ? `Seed #${take.seed}` : take.filename;
    selA.append(new OptionCtor(labelText, take.id));
    selB.append(new OptionCtor(labelText, take.id));
  });

  if (compareState.takeA) selA.value = compareState.takeA.id;
  if (compareState.takeB) selB.value = compareState.takeB.id;
}

export function setCompareMode(mode) {
  const $ = deps.$;
  compareState.mode = mode;
  $("cmp-mode-swiper")?.classList.toggle("active", mode === "swiper");
  $("cmp-mode-grid")?.classList.toggle("active", mode === "grid");
  $("cmp-mode-diff")?.classList.toggle("active", mode === "diff");

  const swiperBox = $("compare-swiper");
  const gridBox = $("compare-grid");
  const diffBox = $("compare-diff");
  const abSelectors = $("compare-ab-selectors");

  if (swiperBox) swiperBox.hidden = mode !== "swiper";
  if (gridBox) gridBox.hidden = mode !== "grid";
  if (diffBox) diffBox.hidden = mode !== "diff";
  if (abSelectors) abSelectors.hidden = mode === "grid";

  const prevTime = activeCompareVideos()[0]?.currentTime ?? compareState.from;

  if (mode === "swiper") {
    setupSwiperVideos(prevTime);
  } else if (mode === "grid") {
    renderCompareGrid(prevTime);
  } else if (mode === "diff") {
    setupDiffVideos(prevTime);
  }

  if (compareState.roi) {
    applyRoiTransform(compareState.roi);
  }

  if (compareState.isPlaying) {
    startCompareSync();
  }
}

export function setupSwiperVideos(seekTo = compareState.from) {
  const $ = deps.$;
  const vidA = $("cmp-vid-a");
  const vidB = $("cmp-vid-b");
  if (!vidA || !vidB || !compareState.takeA || !compareState.takeB) return;

  if (vidA.src !== compareState.takeA.url) vidA.src = compareState.takeA.url;
  if (vidB.src !== compareState.takeB.url) vidB.src = compareState.takeB.url;

  vidA.currentTime = seekTo;
  vidB.currentTime = seekTo;
  vidA.playbackRate = compareState.speed;
  vidB.playbackRate = compareState.speed;
  vidA.muted = compareState.muted;
  vidB.muted = true;

  const nameA = $("cmp-name-a");
  const nameB = $("cmp-name-b");
  if (nameA) nameA.textContent = compareState.takeA.seed != null ? `Seed #${compareState.takeA.seed}` : compareState.takeA.filename;
  if (nameB) nameB.textContent = compareState.takeB.seed != null ? `Seed #${compareState.takeB.seed}` : compareState.takeB.filename;

  updateCompareUseButtons();
}

export function setupDiffVideos(seekTo = compareState.from) {
  const $ = deps.$;
  const vidA = $("cmp-diff-vid-a");
  const vidB = $("cmp-diff-vid-b");
  if (!vidA || !vidB || !compareState.takeA || !compareState.takeB) return;

  if (vidA.src !== compareState.takeA.url) vidA.src = compareState.takeA.url;
  if (vidB.src !== compareState.takeB.url) vidB.src = compareState.takeB.url;

  vidA.currentTime = seekTo;
  vidB.currentTime = seekTo;
  vidA.playbackRate = compareState.speed;
  vidB.playbackRate = compareState.speed;
  vidA.muted = compareState.muted;
  vidB.muted = true;
}

export function updateCompareUseButtons() {
  const $ = deps.$;
  const state = deps.getState() || {};
  const source = deps.currentSource();
  const inEdit = source ? (deps.pathOf(source.id) || [])[state.depth ?? 0] : null;

  const btnA = $("cmp-use-a");
  if (btnA && compareState.takeA) {
    const isChosen = inEdit === compareState.takeA.id;
    btnA.textContent = isChosen ? "In the edit" : "Use this";
    btnA.classList.toggle("in-edit", isChosen);
    btnA.onclick = deps.guard(async () => {
      await deps.setPath(source.id, compareState.takeA.id);
      updateCompareUseButtons();
      deps.renderCandidates();
    });
  }

  const btnB = $("cmp-use-b");
  if (btnB && compareState.takeB) {
    const isChosen = inEdit === compareState.takeB.id;
    btnB.textContent = isChosen ? "In the edit" : "Use this";
    btnB.classList.toggle("in-edit", isChosen);
    btnB.onclick = deps.guard(async () => {
      await deps.setPath(source.id, compareState.takeB.id);
      updateCompareUseButtons();
      deps.renderCandidates();
    });
  }
}

export function renderCompareGrid(seekTo = compareState.from) {
  const $ = deps.$;
  const state = deps.getState() || {};
  const grid = $("compare-grid");
  if (!grid) return;
  grid.innerHTML = "";
  grid.className = `compare-grid n${compareState.takes.length}`;

  const source = deps.currentSource();
  const inEdit = source ? (deps.pathOf(source.id) || [])[state.depth ?? 0] : null;

  for (const take of compareState.takes) {
    const card = document.createElement("div");
    card.className = "cmp-card";

    const wrap = document.createElement("div");
    wrap.className = "cmp-card-video-wrap";
    const video = document.createElement("video");
    video.src = take.url;
    video.preload = "auto";
    video.playsInline = true;
    video.muted = true;
    video.currentTime = seekTo;
    video.playbackRate = compareState.speed;
    wrap.append(video);

    const foot = document.createElement("div");
    foot.className = "cmp-card-foot";

    const title = document.createElement("span");
    title.textContent = take.seed != null ? `Seed #${take.seed}` : take.filename;

    const stars = document.createElement("span");
    stars.className = "stars";
    stars.textContent = deps.ratingLabel(deps.ratingOf(take));

    const spacer = Object.assign(document.createElement("span"), { className: "spacer" });

    const pick = document.createElement("button");
    pick.className = `small pick${inEdit === take.id ? " in-edit" : ""}`;
    pick.innerHTML = deps.label("check", inEdit === take.id ? "In the edit" : "Use this", "sm");
    pick.onclick = deps.guard(async () => {
      await deps.setPath(source.id, take.id);
      renderCompareGrid(video.currentTime);
      updateCompareUseButtons();
      deps.renderCandidates();
    });

    foot.append(title, stars, spacer, pick);
    card.append(wrap, foot);
    grid.append(card);
  }

  if (compareState.roi) {
    applyRoiTransform(compareState.roi);
  }
}

export function updateSplitDivider(percent) {
  const $ = deps.$;
  percent = Math.max(0, Math.min(100, percent));
  compareState.splitPercent = percent;
  const divider = $("compare-divider");
  const layerB = $("cmp-layer-b");
  if (divider) divider.style.left = `${percent}%`;
  if (layerB) layerB.style.clipPath = `inset(0 0 0 ${percent}%)`;
}

export function startCompareSync() {
  const $ = deps.$;
  stopCompareSync();
  const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : () => 1;
  const tick = () => {
    if (!compareState.open) return;
    const vids = activeCompareVideos();
    if (!vids.length) return;
    const lead = vids[0];

    if (compareState.loop) {
      if (compareState.to !== null && lead.currentTime >= compareState.to) {
        vids.forEach((v) => { v.currentTime = compareState.from; });
      } else if (lead.ended) {
        vids.forEach((v) => {
          v.currentTime = compareState.from;
          v.play().catch(() => {});
        });
      }
    }

    for (const v of vids.slice(1)) {
      if (Math.abs(v.currentTime - lead.currentTime) > 0.04) {
        v.currentTime = lead.currentTime;
      }
    }

    updateCompareTransportUI(lead.currentTime, lead.duration || 0);

    if (compareState.roi && !$("compare-minimap")?.hidden) {
      drawMiniMapCanvas(lead);
    }

    compareState.syncHandle = raf(tick);
  };
  compareState.syncHandle = raf(tick);
}

export function stopCompareSync() {
  if (compareState.syncHandle !== null) {
    if (typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(compareState.syncHandle);
    }
    compareState.syncHandle = null;
  }
}

export function stopSync() {
  stopCompareSync();
}

export function startSync(_videos, _from, _to) {
  startCompareSync();
}

export function updateCompareTransportUI(currentTime, duration) {
  const $ = deps.$;
  const timeEl = $("cmp-time");
  if (timeEl) {
    const curSec = formatTimeSec(currentTime);
    const durSec = formatTimeSec(duration);
    timeEl.textContent = `${curSec} / ${durSec}`;
  }

  const frameEl = $("cmp-frame");
  if (frameEl) {
    const f = Math.floor(currentTime * 24);
    frameEl.textContent = `f: ${f}`;
  }

  const scrub = $("cmp-scrub");
  if (scrub && duration > 0 && !scrub.matches?.(":active")) {
    scrub.value = String((currentTime / duration) * 100);
  }
}

export function formatTimeSec(sec) {
  sec = Math.max(0, sec || 0);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const cs = Math.floor((sec % 1) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

export function updateComparePlayButton() {
  const btn = deps.$("cmp-play");
  if (!btn) return;
  btn.innerHTML = deps.label(compareState.isPlaying ? "pause" : "play", "", "fill");
}

export function updateCompareSoundButton() {
  const btn = deps.$("cmp-sound");
  if (!btn) return;
  btn.innerHTML = deps.iconHTML(compareState.muted ? "volume-x" : "volume", "sm");
}

export function toggleComparePlay() {
  const vids = activeCompareVideos();
  if (!vids.length) return;
  const lead = vids[0];

  if (lead.paused || !compareState.isPlaying) {
    vids.forEach((v) => {
      v.playbackRate = compareState.speed;
      v.play().catch(() => {});
    });
    compareState.isPlaying = true;
    updateComparePlayButton();
    startCompareSync();
  } else {
    vids.forEach((v) => v.pause());
    compareState.isPlaying = false;
    updateComparePlayButton();
    stopCompareSync();
  }
}

export function stepCompareFrame(deltaFrames) {
  const vids = activeCompareVideos();
  if (!vids.length) return;

  if (compareState.isPlaying) {
    vids.forEach((v) => v.pause());
    compareState.isPlaying = false;
    updateComparePlayButton();
    stopCompareSync();
  }

  const lead = vids[0];
  const frameDur = 1 / 24;
  const targetTime = Math.max(0, lead.currentTime + deltaFrames * frameDur);

  vids.forEach((v) => {
    v.currentTime = targetTime;
  });

  updateCompareTransportUI(targetTime, lead.duration || 0);
  if (compareState.roi) drawMiniMapCanvas(lead);
}

// -- ROI / Detail Box Zoom -----------------------------------------------

export function getVideoDisplayBounds() {
  const $ = deps.$;
  const vids = activeCompareVideos();
  const video = vids[0] || $("cmp-vid-a");
  const stage = $("compare-stage");
  if (!stage || !stage.getBoundingClientRect) return { left: 0, top: 0, width: 800, height: 450 };

  const sRect = stage.getBoundingClientRect();
  const vW = video?.videoWidth || 1920;
  const vH = video?.videoHeight || 1080;
  const sAspect = sRect.width / (sRect.height || 1);
  const vAspect = vW / vH;

  let width, height, left, top;
  if (sAspect > vAspect) {
    height = sRect.height;
    width = height * vAspect;
    top = 0;
    left = (sRect.width - width) / 2;
  } else {
    width = sRect.width;
    height = width / vAspect;
    left = 0;
    top = (sRect.height - height) / 2;
  }
  return { left, top, width, height, right: left + width, bottom: top + height };
}

export function applyRoiTransform(roi) {
  const $ = deps.$;
  const modal = $("compare-modal");
  const targets = modal?.querySelectorAll
    ? [
        ...modal.querySelectorAll(".cmp-video-layer"),
        ...modal.querySelectorAll(".cmp-card-video-wrap"),
        ...modal.querySelectorAll(".cmp-diff-stage"),
      ]
    : [];

  if (!roi) {
    targets.forEach((el) => {
      el.style.transform = "";
      el.style.transformOrigin = "center center";
    });
    const resetBtn = $("cmp-roi-reset");
    const minimap = $("compare-minimap");
    if (resetBtn) resetBtn.hidden = true;
    if (minimap) minimap.hidden = true;
    return;
  }

  const { scale, cx, cy } = roi;
  targets.forEach((el) => {
    el.style.transformOrigin = `${(cx * 100).toFixed(2)}% ${(cy * 100).toFixed(2)}%`;
    el.style.transform = `scale(${scale.toFixed(2)})`;
  });

  const resetBtn = $("cmp-roi-reset");
  const zoomLevel = $("cmp-roi-zoom-level");
  if (resetBtn) resetBtn.hidden = false;
  if (zoomLevel) zoomLevel.textContent = `(${scale.toFixed(1)}×)`;

  updateMiniMapUI(roi);
}

export function resetRoiZoom() {
  compareState.roi = null;
  applyRoiTransform(null);
}

export function updateMiniMapUI(roi) {
  const $ = deps.$;
  const minimap = $("compare-minimap");
  const roiBox = $("cmp-minimap-roi");
  if (!minimap || !roiBox) return;

  minimap.hidden = false;
  roiBox.style.left = `${(roi.x * 100).toFixed(2)}%`;
  roiBox.style.top = `${(roi.y * 100).toFixed(2)}%`;
  roiBox.style.width = `${(roi.w * 100).toFixed(2)}%`;
  roiBox.style.height = `${(roi.h * 100).toFixed(2)}%`;

  const lead = activeCompareVideos()[0] || $("cmp-vid-a");
  drawMiniMapCanvas(lead);
}

export function drawMiniMapCanvas(video) {
  const canvas = deps.$("cmp-minimap-canvas");
  if (!canvas || !video || !video.videoWidth || typeof canvas.getContext !== "function") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  } catch {
    // Media may not yet be drawn
  }
}

export function toggleRoiDrawing(force) {
  const $ = deps.$;
  const next = force !== undefined ? force : !compareState.drawingRoi;
  compareState.drawingRoi = next;
  const btn = $("cmp-roi-btn");
  const layer = $("compare-roi-layer");
  if (btn) btn.classList.toggle("active", next);
  if (layer) layer.hidden = !next;
  const marquee = $("compare-roi-marquee");
  if (marquee) marquee.hidden = true;
}

export function wireCompareModal() {
  const $ = deps.$;
  $("compare-close")?.addEventListener?.("click", closeCompareModal);

  $("cmp-fullscreen-btn")?.addEventListener?.("click", () => {
    const modal = $("compare-modal");
    if (!modal) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      modal.requestFullscreen().catch(() => {});
    }
  });

  $("cmp-mode-swiper")?.addEventListener?.("click", () => setCompareMode("swiper"));
  $("cmp-mode-grid")?.addEventListener?.("click", () => setCompareMode("grid"));
  $("cmp-mode-diff")?.addEventListener?.("click", () => setCompareMode("diff"));

  $("cmp-select-a")?.addEventListener?.("change", (e) => {
    const found = compareState.takes.find((t) => t.id === e.target.value);
    if (found) {
      compareState.takeA = found;
      setupSwiperVideos();
      if (compareState.mode === "diff") setupDiffVideos();
    }
  });
  $("cmp-select-b")?.addEventListener?.("change", (e) => {
    const found = compareState.takes.find((t) => t.id === e.target.value);
    if (found) {
      compareState.takeB = found;
      setupSwiperVideos();
      if (compareState.mode === "diff") setupDiffVideos();
    }
  });

  $("cmp-swap-btn")?.addEventListener?.("click", () => {
    const tmp = compareState.takeA;
    compareState.takeA = compareState.takeB;
    compareState.takeB = tmp;
    populateCompareSelects();
    setupSwiperVideos();
    if (compareState.mode === "diff") setupDiffVideos();
  });

  $("cmp-roi-btn")?.addEventListener?.("click", () => toggleRoiDrawing());
  $("cmp-roi-reset")?.addEventListener?.("click", () => resetRoiZoom());

  const flickBtn = $("cmp-flicker-btn");
  if (flickBtn?.addEventListener) {
    const pressFlick = () => {
      const vidB = $("cmp-diff-vid-b");
      if (vidB) vidB.style.visibility = "hidden";
    };
    const releaseFlick = () => {
      const vidB = $("cmp-diff-vid-b");
      if (vidB) vidB.style.visibility = "visible";
    };
    flickBtn.addEventListener("mousedown", pressFlick);
    flickBtn.addEventListener("mouseup", releaseFlick);
    flickBtn.addEventListener("mouseleave", releaseFlick);
    flickBtn.addEventListener("touchstart", pressFlick);
    flickBtn.addEventListener("touchend", releaseFlick);
  }

  $("cmp-play")?.addEventListener?.("click", toggleComparePlay);
  $("cmp-step-back")?.addEventListener?.("click", () => stepCompareFrame(-1));
  $("cmp-step-fwd")?.addEventListener?.("click", () => stepCompareFrame(1));

  const scrub = $("cmp-scrub");
  if (scrub?.addEventListener) {
    scrub.addEventListener("input", () => {
      const vids = activeCompareVideos();
      if (!vids.length) return;
      const lead = vids[0];
      const duration = lead.duration || 1;
      const targetTime = (parseFloat(scrub.value) / 100) * duration;
      vids.forEach((v) => { v.currentTime = targetTime; });
      updateCompareTransportUI(targetTime, duration);
      if (compareState.roi) drawMiniMapCanvas(lead);
    });
  }

  $("cmp-speed")?.addEventListener?.("change", (e) => {
    compareState.speed = parseFloat(e.target.value) || 1;
    activeCompareVideos().forEach((v) => { v.playbackRate = compareState.speed; });
  });

  $("cmp-loop")?.addEventListener?.("click", (e) => {
    compareState.loop = !compareState.loop;
    e.currentTarget.classList.toggle("active", compareState.loop);
  });

  $("cmp-sound")?.addEventListener?.("click", () => {
    compareState.muted = !compareState.muted;
    const vidA = $("cmp-vid-a");
    if (vidA) vidA.muted = compareState.muted;
    const diffVidA = $("cmp-diff-vid-a");
    if (diffVidA) diffVidA.muted = compareState.muted;
    updateCompareSoundButton();
  });

  wireSwiperDrag();
  wireRoiDrawingDrag();
  wireMiniMapPan();

  $("compare-stage")?.addEventListener?.("dblclick", (e) => {
    if (e.target.closest?.("#compare-minimap") || e.target.closest?.("#compare-divider")) return;
    resetRoiZoom();
  });
}

function wireSwiperDrag() {
  const $ = deps.$;
  const stage = $("compare-stage");
  const divider = $("compare-divider");
  if (!stage?.addEventListener || !divider?.addEventListener) return;

  const onPointerDown = (e) => {
    if (compareState.mode !== "swiper" || compareState.drawingRoi) return;
    if (e.target.closest?.("#compare-minimap") || e.target.closest?.(".cmp-badge")) return;
    compareState.isDraggingDivider = true;
    divider.setPointerCapture?.(e.pointerId);
    updateFromPointer(e);
  };

  const updateFromPointer = (e) => {
    const rect = stage.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = (x / rect.width) * 100;
    updateSplitDivider(pct);
  };

  const onPointerMove = (e) => {
    if (compareState.isDraggingDivider) {
      updateFromPointer(e);
    }
  };

  const onPointerUp = (e) => {
    if (compareState.isDraggingDivider) {
      compareState.isDraggingDivider = false;
      try { divider.releasePointerCapture?.(e.pointerId); } catch {}
    }
  };

  divider.addEventListener("pointerdown", onPointerDown);
  divider.addEventListener("pointermove", onPointerMove);
  divider.addEventListener("pointerup", onPointerUp);
  divider.addEventListener("pointercancel", onPointerUp);

  stage.addEventListener("pointerdown", (e) => {
    if (compareState.mode === "swiper" && !compareState.drawingRoi) {
      if (e.target.closest?.("#compare-minimap") || e.target.closest?.(".cmp-badge") || e.target.closest?.("#compare-divider")) return;
      onPointerDown(e);
    }
  });
}

function wireRoiDrawingDrag() {
  const $ = deps.$;
  const layer = $("compare-roi-layer");
  const marquee = $("compare-roi-marquee");
  if (!layer?.addEventListener || !marquee) return;

  let startX = 0;
  let startY = 0;
  let isDragging = false;

  layer.addEventListener("pointerdown", (e) => {
    isDragging = true;
    layer.setPointerCapture?.(e.pointerId);
    const rect = layer.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;

    marquee.hidden = false;
    marquee.style.left = `${startX}px`;
    marquee.style.top = `${startY}px`;
    marquee.style.width = "0px";
    marquee.style.height = "0px";
  });

  layer.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const rect = layer.getBoundingClientRect();
    const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    const left = Math.min(startX, curX);
    const top = Math.min(startY, curY);
    const width = Math.abs(curX - startX);
    const height = Math.abs(curY - startY);

    marquee.style.left = `${left}px`;
    marquee.style.top = `${top}px`;
    marquee.style.width = `${width}px`;
    marquee.style.height = `${height}px`;
  });

  layer.addEventListener("pointerup", (e) => {
    if (!isDragging) return;
    isDragging = false;
    try { layer.releasePointerCapture?.(e.pointerId); } catch {}

    const rect = layer.getBoundingClientRect();
    const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    const mLeft = Math.min(startX, curX);
    const mTop = Math.min(startY, curY);
    const mWidth = Math.abs(curX - startX);
    const mHeight = Math.abs(curY - startY);

    marquee.hidden = true;
    toggleRoiDrawing(false);

    if (mWidth >= 18 && mHeight >= 18) {
      const vBounds = getVideoDisplayBounds();
      const nx0 = Math.max(0, Math.min(1, (mLeft - vBounds.left) / vBounds.width));
      const ny0 = Math.max(0, Math.min(1, (mTop - vBounds.top) / vBounds.height));
      const nx1 = Math.max(0, Math.min(1, (mLeft + mWidth - vBounds.left) / vBounds.width));
      const ny1 = Math.max(0, Math.min(1, (mTop + mHeight - vBounds.top) / vBounds.height));

      const roiX = Math.min(nx0, nx1);
      const roiY = Math.min(ny0, ny1);
      const roiW = Math.max(0.04, Math.abs(nx1 - nx0));
      const roiH = Math.max(0.04, Math.abs(ny1 - ny0));

      const cx = roiX + roiW / 2;
      const cy = roiY + roiH / 2;
      const scale = Math.min(8, Math.max(1.5, 1 / Math.max(roiW, roiH)));

      compareState.roi = { x: roiX, y: roiY, w: roiW, h: roiH, scale, cx, cy };
      applyRoiTransform(compareState.roi);
    }
  });
}

function wireMiniMapPan() {
  const $ = deps.$;
  const minimap = $("compare-minimap");
  const roiBox = $("cmp-minimap-roi");
  const thumbWrap = $("cmp-minimap-thumb-wrap");
  if (!minimap || !roiBox?.addEventListener || !thumbWrap?.addEventListener) return;

  let isPanning = false;
  let startX = 0;
  let startY = 0;
  let initCx = 0.5;
  let initCy = 0.5;

  roiBox.addEventListener("pointerdown", (e) => {
    if (!compareState.roi) return;
    isPanning = true;
    roiBox.setPointerCapture?.(e.pointerId);
    startX = e.clientX;
    startY = e.clientY;
    initCx = compareState.roi.cx;
    initCy = compareState.roi.cy;
    e.stopPropagation();
  });

  roiBox.addEventListener("pointermove", (e) => {
    if (!isPanning || !compareState.roi) return;
    const rect = thumbWrap.getBoundingClientRect();
    const dx = (e.clientX - startX) / rect.width;
    const dy = (e.clientY - startY) / rect.height;

    const roi = compareState.roi;
    const halfW = roi.w / 2;
    const halfH = roi.h / 2;
    const newCx = Math.max(halfW, Math.min(1 - halfW, initCx + dx));
    const newCy = Math.max(halfH, Math.min(1 - halfH, initCy + dy));

    roi.cx = newCx;
    roi.cy = newCy;
    roi.x = newCx - halfW;
    roi.y = newCy - halfH;

    applyRoiTransform(roi);
  });

  const stopPan = (e) => {
    if (isPanning) {
      isPanning = false;
      try { roiBox.releasePointerCapture?.(e.pointerId); } catch {}
    }
  };
  roiBox.addEventListener("pointerup", stopPan);
  roiBox.addEventListener("pointercancel", stopPan);

  thumbWrap.addEventListener("click", (e) => {
    if (e.target === roiBox || !compareState.roi) return;
    const rect = thumbWrap.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / rect.width;
    const clickY = (e.clientY - rect.top) / rect.height;

    const roi = compareState.roi;
    const halfW = roi.w / 2;
    const halfH = roi.h / 2;
    roi.cx = Math.max(halfW, Math.min(1 - halfW, clickX));
    roi.cy = Math.max(halfH, Math.min(1 - halfH, clickY));
    roi.x = roi.cx - halfW;
    roi.y = roi.cy - halfH;

    applyRoiTransform(roi);
  });
}

export function renderCompare() {
  const state = deps.getState() || {};
  const chosen = [...(state.compare || [])].map(deps.takeById).filter(Boolean);
  if (chosen.length < 2) {
    deps.toast("Select at least 2 takes to compare.");
    return;
  }
  openCompareModal(chosen);
}
