// Picky demo shim: the real editor, answered from a recorded snapshot.
//
// Loaded before the editor's own modules. It replaces `fetch` and `WebSocket`
// so every `/video_prompter/*` call and ComfyUI's `/prompt`, `/queue`,
// `/history` and `/ws` are answered in the page from `data/snapshot.json`.
// Edits, picks and ratings change an in-memory copy for this visit only. A
// render runs a fake progress bar and lands a take that was really rendered
// for that source. Nothing is sent anywhere.
(() => {
  "use strict";

  const BASE = new URL("./", document.currentScript.src).href;
  const REPO = "https://github.com/chenthedesignmaker/ComfyUI-Picky";
  const LOCAL_ONLY = "Available when you run Picky locally.";
  const DEMO_NOTE = {
    rule: "demo.snapshot",
    severity: "info",
    message: "Demo: this is the recorded composition. Run Picky locally for live composing and linting of your edits.",
    where: "",
  };

  const realFetch = window.fetch.bind(window);
  const clone = (value) => (value === undefined ? value : JSON.parse(JSON.stringify(value)));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const uuid = () =>
    (crypto.randomUUID ? crypto.randomUUID()
      : "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16)));

  // -- state ---------------------------------------------------------------

  let snap = null;
  const S = {}; // the mutable copy for this visit
  const sockets = new Set();
  const queue = []; // fake ComfyUI jobs: {id, payload, takeFrom, parent}
  const history = {};

  function demoToast(message) {
    const element = document.getElementById("toast");
    if (!element) return;
    element.textContent = message;
    element.className = "show";
    clearTimeout(demoToast.timer);
    demoToast.timer = setTimeout(() => { element.className = ""; }, 3500);
  }

  const takes = () => S.gallery.takes;
  const takeById = (id) => takes().find((t) => t.id === id);
  const sourceById = (id) => S.project.sources.find((s) => s.id === id);
  const videoTakesOf = (sourceId) =>
    takes().filter((t) => t.shot === sourceId && t.is_video !== false && /\.(mp4|webm|mov)$/i.test(t.filename));

  function forEachNode(fn) {
    const walk = (node) => { fn(node); (node.children || []).forEach(walk); };
    for (const forest of Object.values(S.lineage)) forest.roots.forEach(walk);
  }

  // Everything that follows from a source's path: the sequence clips and
  // source rows, the lineage's on-path marks and the timeline arithmetic.
  function derive() {
    const paths = Object.fromEntries(S.project.sources.map((s) => [s.id, s.path || []]));
    const onPath = new Set(Object.values(paths).flat());
    forEachNode((node) => { node.on_path = onPath.has(node.take_id); });
    for (const [sourceId, forest] of Object.entries(S.lineage)) {
      let at = 0;
      forest.path = [...(paths[sourceId] || [])];
      forest.segments = forest.path.map((id, depth) => {
        const take = takeById(id) || {};
        const start = at;
        at += take.seconds || 0;
        return { take_id: id, depth, filename: take.filename, shot: sourceId, start, end: at, seconds: take.seconds || 0 };
      });
      forest.seconds = Math.round(at * 1000) / 1000;
    }

    const seq = S.sequence;
    for (const row of seq.sources || []) {
      const path = paths[row.id] || [];
      const last = takeById(path[path.length - 1]);
      row.path = [...path];
      row.chosen = path.length > 0;
      row.seconds = row.rendered_seconds = last ? last.seconds : 0;
      row.take_count = videoTakesOf(row.id).length;
      (row.segments || []).forEach((seg, i) => { seg.take_id = path[i] || null; });
      (row.branches || []).forEach((branch) => {
        branch.chosen = path[branch.depth] || null;
        const have = new Set(branch.options.map((o) => o.take_id));
        for (const take of videoTakesOf(row.id)) {
          if (!have.has(take.id) && (take.parent_take || null) === (branch.depth ? path[branch.depth - 1] : null)) {
            branch.options.push({ take_id: take.id, filename: take.filename, rating: take.rating || 0, seconds: take.seconds, children: 0 });
          }
        }
        branch.options.forEach((o) => { o.rating = takeById(o.take_id)?.rating || 0; });
      });
    }

    const clipsById = Object.fromEntries((S.project.timeline?.clips || []).map((c) => [c.id, c]));
    let at = 0;
    for (const clip of seq.clips || []) {
      const own = clipsById[clip.id];
      const path = paths[clip.source_id] || [];
      const last = takeById(path[path.length - 1]);
      clip.path = [...path];
      clip.rendered = !!last;
      clip.source_seconds = last ? last.seconds : 0;
      if (own) {
        clip.source_in = own.source_in || 0;
        clip.source_out = own.source_out ?? clip.source_seconds;
        clip.speed = own.speed || 1;
        clip.enabled = own.enabled !== false;
        clip.label = own.label || "";
        clip.audio = { ...clip.audio, ...(own.audio || {}) };
      } else {
        clip.source_out = clip.source_seconds;
      }
      clip.source_out = Math.min(clip.source_out, clip.source_seconds);
      clip.timeline_seconds = Math.max(0, (clip.source_out - clip.source_in) / clip.speed);
      clip.starts_at = Math.round(at * 1000) / 1000;
      at += clip.timeline_seconds;
    }
    seq.duration = Math.round(at * 1000) / 1000;
  }

  function closestPreview(prompt) {
    const entries = Object.values(snap.previews);
    const text = JSON.stringify(prompt);
    const exact = entries.find((e) => JSON.stringify(e.prompt) === text);
    if (exact) return { preview: clone(exact.preview), exact: true };
    let best = entries[0], score = -1;
    for (const entry of entries) {
      let s = 0;
      for (const [key, value] of Object.entries(entry.prompt || {})) {
        if (JSON.stringify(value) === JSON.stringify(prompt?.[key])) s += 1;
      }
      if (s > score) { best = entry; score = s; }
    }
    return { preview: clone(best.preview), exact: false };
  }

  function previewFor(prompt) {
    const { preview, exact } = closestPreview(prompt);
    if (!exact) preview.findings = [DEMO_NOTE, ...(preview.findings || [])];
    return preview;
  }

  // -- the fake ComfyUI ------------------------------------------------------

  function broadcast(type, data) {
    const message = JSON.stringify({ type, data });
    for (const socket of sockets) socket._deliver(message);
  }

  function queueStatus() {
    broadcast("status", { status: { exec_info: { queue_remaining: queue.length } } });
  }

  let running = false;
  async function runQueue() {
    if (running) return;
    running = true;
    while (queue.length) {
      const job = queue[0];
      job.running = true;
      broadcast("execution_start", { prompt_id: job.id });
      queueStatus();
      const steps = 8;
      const node = Object.keys(job.payload.prompt || {})[0] || "1";
      broadcast("executing", { prompt_id: job.id, node, display_node: node });
      for (let value = 1; value <= steps; value++) {
        await sleep(450);
        broadcast("progress", { prompt_id: job.id, value, max: steps, node, display_node: node });
      }
      await sleep(300);
      queue.shift();
      history[job.id] = { prompt: [0, job.id, job.payload.prompt, job.payload.extra_data, []], outputs: {}, status: { completed: true, status_str: "success" } };
      job.done = true;
      broadcast("executing", { prompt_id: job.id, node: null });
      broadcast("execution_success", { prompt_id: job.id });
      queueStatus();
    }
    running = false;
  }

  // A finished fake job becomes a copy of a real take of that source, so it
  // plays real footage. It rotates through the source's takes.
  const rotation = {};
  function landTake(job) {
    const meta = job.payload.extra_data?.video_prompter || {};
    const sourceId = meta.shot;
    const pool = snap.get["/gallery"].takes.filter((t) => t.shot === sourceId && /\.(mp4|webm|mov)$/i.test(t.filename));
    if (!pool.length) return null;
    rotation[sourceId] = ((rotation[sourceId] ?? -1) + 1) % pool.length;
    const from = pool[rotation[sourceId]];
    const take = {
      ...clone(from),
      id: `${job.id}:${from.filename}`,
      prompt_id: job.id,
      created: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"),
      rating: 0,
      discarded: false,
      seen: false,
      seed: meta.seed ?? from.seed,
      quality: meta.quality ?? from.quality,
      parent_take: meta.parent_take || null,
      prompt_text: meta.prompt || from.prompt_text,
      generation_time: 4.2,
    };
    S.gallery.takes.unshift(take);
    const node = { take_id: take.id, depth: 0, on_path: false, take, children: [] };
    let parent = null;
    if (take.parent_take) forEachNode((n) => { if (n.take_id === take.parent_take) parent = n; });
    if (parent) {
      node.depth = parent.depth + 1;
      parent.children = parent.children || [];
      parent.children.push(node);
    } else if (S.lineage[sourceId]) {
      S.lineage[sourceId].roots.push(node);
    }
    derive();
    return take;
  }

  class DemoSocket extends EventTarget {
    constructor(url) {
      super();
      this.url = url;
      this.readyState = 0;
      this.onopen = this.onmessage = this.onclose = this.onerror = null;
      sockets.add(this);
      setTimeout(() => {
        this.readyState = 1;
        const event = new Event("open");
        this.dispatchEvent(event);
        this.onopen?.(event);
        this._deliver(JSON.stringify({ type: "status", data: { status: { exec_info: { queue_remaining: queue.length } }, sid: "demo" } }));
      }, 50);
    }
    _deliver(data) {
      if (this.readyState !== 1) return;
      const event = new MessageEvent("message", { data });
      this.dispatchEvent(event);
      this.onmessage?.(event);
    }
    send() {}
    close() { this.readyState = 3; sockets.delete(this); }
  }
  DemoSocket.CONNECTING = 0; DemoSocket.OPEN = 1; DemoSocket.CLOSING = 2; DemoSocket.CLOSED = 3;

  // -- routes ----------------------------------------------------------------

  const json = (body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const localOnly = () => { demoToast(LOCAL_ONLY); return json({ error: LOCAL_ONLY, demo: true }, 409); };
  const projectReply = () => json({ project: clone(S.project) });

  function fakePrepared(body) {
    const source = sourceById(body.source_id || body.source || body.shot) || S.project.sources[0];
    const segment = source.segments?.[0];
    const preview = previewFor(body.prompt || segment?.prompt);
    return {
      workflow: { "1": { class_type: "PickyDemo", inputs: { prompt: preview.prompt } } },
      prompt_text: body.prompt_text || preview.prompt,
      frames: body.frames || preview.frames,
      seconds: body.seconds || preview.seconds,
      width: body.width || preview.width,
      height: body.height || preview.height,
      seed: body.seed ?? Math.floor(Math.random() * 2 ** 48),
      steps: 8,
      quality: body.quality || segment?.quality || "preview",
      parent_take: body.parent_take || body.take || null,
      source_id: source.id,
      findings: preview.findings,
    };
  }

  function prepareMany(body) {
    const count = Math.max(1, Math.min(8, Number(body.count) || 1));
    const base = body.seed ?? Math.floor(Math.random() * 2 ** 40);
    return { renders: Array.from({ length: count }, (_, i) => ({ ...fakePrepared(body), seed: base + i })) };
  }

  const GETS = {
    "/projects": () => json(clone(snap.get["/projects"])),
    // Enabled, so pressing Write explains itself instead of sitting greyed out.
    "/llm/status": () => json({ configured: true, provider: "demo", model: "your own model when you run Picky locally" }),
    "/project/load": () => projectReply(),
    "/gallery": (q) => {
      const source = q.get("source") || q.get("shot");
      return json({ takes: clone(source ? takes().filter((t) => t.shot === source) : takes()) });
    },
    "/lineage": (q) => json(clone(S.lineage[q.get("source") || q.get("shot")] || { roots: [], path: [], problems: [] })),
    "/sequence": () => json(clone(S.sequence)),
    "/stills": () => json(clone(S.stills)),
    "/cast": () => json({ ...clone(snap.get["/cast"]), cast: clone(S.project.cast) }),
    "/cast/usage": (q) => json(clone(snap.cast_usage?.[q.get("member")] || { sources: [] })),
    "/profile": (q) => json(clone(snap.profile?.[q.get("name")] || {})),
    "/preset": (q) => json(clone(snap.preset?.[`${q.get("kind")}/${q.get("name")}`] || {})),
    "/source/revisions": (q) => {
      const key = Object.keys(snap.revisions || {}).find((k) => k.startsWith(`${q.get("source")}/`) && (!q.get("segment") || k.endsWith(`/${q.get("segment")}`)));
      return json(clone(snap.revisions?.[key] || { revisions: [] }));
    },
  };

  const POSTS = {
    "/preview": (b) => json(previewFor(b.prompt)),
    "/transform": (b) => { demoToast("Demo: storyboard edits need the local server."); return json({ prompt: b.prompt, preview: previewFor(b.prompt) }); },
    "/source/update": (b) => {
      const i = S.project.sources.findIndex((s) => s.id === b.source?.id);
      if (i >= 0) S.project.sources[i] = clone(b.source);
      return projectReply();
    },
    "/project/update": (b) => { if (b.project) S.project = { ...S.project, ...clone(b.project), name: S.project.name }; derive(); return projectReply(); },
    "/defaults": (b) => { S.project.defaults = { ...S.project.defaults, ...clone(b.defaults || {}) }; return projectReply(); },
    "/gallery/pick": (b) => {
      const source = sourceById(b.source || b.shot);
      if (source) {
        const path = [];
        let take = takeById(b.take);
        while (take) { path.unshift(take.id); take = take.parent_take ? takeById(take.parent_take) : null; }
        source.path = path;
      }
      derive();
      return projectReply();
    },
    "/source/path": (b) => { const s = sourceById(b.source); if (s) s.path = [...(b.path || [])]; derive(); return projectReply(); },
    "/gallery/annotate": (b) => {
      const take = takeById(b.take);
      if (!take) return json({ error: "No such take." }, 404);
      for (const key of ["rating", "discarded", "seen"]) if (b[key] !== undefined && b[key] !== null) take[key] = key === "rating" ? Number(b[key]) : !!b[key];
      forEachNode((n) => { if (n.take_id === take.id) n.take = take; });
      derive();
      return json({ take: clone(take) });
    },
    "/gallery/seen": (b) => {
      const changed = [];
      for (const id of b.takes || []) { const t = takeById(id); if (t) { t.seen = b.seen !== false; changed.push(id); } }
      return json({ changed, count: changed.length });
    },
    "/gallery/record": () => {
      const added = [];
      for (const job of Object.values(pending)) {
        if (!job.done || job.recorded) continue;
        job.recorded = true;
        const take = landTake(job);
        if (take) added.push(take);
      }
      return json({ added: clone(added), count: added.length });
    },
    "/gallery/context": (b) => {
      const take = takeById(b.take || b.id) || {};
      return json({ id: take.id, filename: take.filename, shot: take.shot, seed: take.seed, prompt_text: take.prompt_text, demo: true });
    },
    "/stills/annotate": (b) => {
      const still = S.stills.stills.find((s) => s.id === b.still);
      if (still) for (const key of ["rating", "discarded", "seen", "note"]) if (b[key] !== undefined) still[key] = b[key];
      return json({ still: clone(still || {}) });
    },
    "/stills/record": () => json({ added: [], count: 0 }),
    "/render/prepare": (b) => json(fakePrepared(b)),
    "/render/batch": (b) => json(prepareMany(b)),
    "/render/reshoot": (b) => json(fakePrepared(b)),
    "/render/extend": (b) => json(fakePrepared(b)),
    "/render/extend/batch": (b) => json(prepareMany(b)),
    "/render/hold": () => localOnly(),
    "/clip/edit": (b) => {
      const clip = S.project.timeline?.clips?.find((c) => c.id === b.clip);
      if (clip) {
        for (const [key, value] of Object.entries(b)) {
          if (["project", "clip", "origin"].includes(key)) continue;
          clip[key] = key === "audio" ? { ...clip.audio, ...value } : value;
        }
      }
      derive();
      return projectReply();
    },
  };
  // Routes that need the GPU, the disk, a model key or the OS.
  for (const route of ["/llm/draft", "/llm/draft-image", "/reference/upload", "/export", "/reveal",
    "/render/refine", "/render/t2i", "/render/t2i/batch", "/render/adopt", "/project/create",
    "/project/duplicate", "/project/shelve", "/project/purge", "/project/empty-trash", "/project/delete",
    "/project/restore", "/profile/save", "/profile/delete", "/preset/save", "/preset/delete",
    "/video-loras", "/image_prompt/revisions/file", "/parse"]) {
    POSTS[route] = () => localOnly();
  }

  const pending = {};

  async function answer(url, init) {
    const method = (init.method || "GET").toUpperCase();
    let body = {};
    if (init.body && typeof init.body === "string") {
      try { body = JSON.parse(init.body); } catch { body = {}; }
    }
    const path = url.pathname;

    if (path.endsWith("/prompt") && method === "POST" && !path.includes("/video_prompter/")) {
      const id = body.prompt_id || uuid();
      const job = { id, payload: body };
      pending[id] = job;
      queue.push(job);
      setTimeout(runQueue, 200);
      return json({ prompt_id: id, number: queue.length, node_errors: {} });
    }
    if (path.endsWith("/queue")) {
      const entry = (job) => [0, job.id, job.payload.prompt, job.payload.extra_data, []];
      return json({
        queue_running: queue.filter((j) => j.running).map(entry),
        queue_pending: queue.filter((j) => !j.running).map(entry),
      });
    }
    if (path.endsWith("/interrupt")) return json({});
    const historyAt = path.indexOf("/history");
    if (historyAt >= 0 && !path.includes("/video_prompter/")) {
      const id = path.slice(historyAt + "/history/".length);
      return json(id ? (history[id] ? { [id]: history[id] } : {}) : clone(history));
    }

    const at = path.indexOf("/video_prompter/");
    const route = path.slice(at + "/video_prompter".length).replace(/\/$/, "");
    if (method === "GET") {
      if (GETS[route]) return GETS[route](url.searchParams);
      if (route in snap.get) return json(clone(snap.get[route]));
      console.warn("[picky demo] unrecorded GET", route);
      return json({});
    }
    if (POSTS[route]) return POSTS[route](body);
    // Any other edit: accept it without changing anything, so the UI keeps going.
    console.info("[picky demo] accepted without effect:", route);
    demoToast("Demo: that change is not kept. " + LOCAL_ONLY);
    return projectReply();
  }

  function handles(url) {
    if (url.origin !== location.origin) return false;
    const p = url.pathname;
    if (p.includes("/video_prompter/") && !p.includes("/video_prompter/app/")) return true;
    return /\/(prompt|queue|interrupt)$/.test(p) || /\/history(\/|$)/.test(p);
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (!handles(url)) return realFetch(input, init);
    await ready;
    if (typeof input !== "string" && !init.method) init = { method: input.method, body: init.body };
    await sleep(15);
    return answer(url, init);
  };

  const RealSocket = window.WebSocket;
  window.WebSocket = function (url, protocols) {
    if (/\/ws(\?|$)/.test(String(url))) return new DemoSocket(url);
    return new RealSocket(url, protocols);
  };
  Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });

  // -- boot --------------------------------------------------------------------

  function banner() {
    const bar = document.createElement("div");
    bar.id = "picky-demo-banner";
    bar.innerHTML =
      `<span><b>Picky demo</b> · a recorded snapshot of the LAST ORDER project. ` +
      `Pick, rate, edit and render are simulated; real renders need MiniMax H3 on a 24 GB GPU.</span>` +
      `<span class="links"><a href="${REPO}" target="_blank" rel="noopener">GitHub</a>` +
      `<a href="${REPO}#install" target="_blank" rel="noopener">Install</a>` +
      `<button type="button" aria-label="Hide the demo banner">×</button></span>`;
    bar.querySelector("button").onclick = () => bar.remove();
    document.body.prepend(bar);
  }

  async function loadSnapshot() {
    const [snapshot] = await Promise.all([
      realFetch(BASE + "data/snapshot.json").then((r) => r.json()),
    ]);
    snap = snapshot;
    S.project = clone(snap.get["/project/load"].project);
    S.gallery = clone(snap.get["/gallery"]);
    S.lineage = clone(snap.lineage);
    S.sequence = clone(snap.get["/sequence"]);
    S.stills = clone(snap.get["/stills"]);
    derive();
    try { localStorage.setItem("vp:lastProject", snap.project_name); } catch { /* private mode */ }
  }

  async function serviceWorker() {
    if (!("serviceWorker" in navigator)) return false;
    try {
      await navigator.serviceWorker.register(BASE + "sw.js", { scope: BASE });
      if (navigator.serviceWorker.controller) return true;
      await new Promise((resolve) => {
        navigator.serviceWorker.addEventListener("controllerchange", resolve, { once: true });
        setTimeout(resolve, 4000);
      });
      return !!navigator.serviceWorker.controller;
    } catch (error) {
      console.warn("[picky demo] service worker unavailable:", error);
      return false;
    }
  }

  const ready = Promise.all([loadSnapshot(), serviceWorker()]).then(([, worker]) => {
    if (!worker) console.warn("[picky demo] no service worker: takes and images will not load.");
  });

  function start() {
    banner();
    ready.then(() => {
      for (const src of ["app/screening-room/motion.js", "app/main.js"]) {
        const script = document.createElement("script");
        script.type = "module";
        script.src = BASE + src;
        document.body.append(script);
      }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
  window.__pickyDemo = { state: S, queue };
})();
