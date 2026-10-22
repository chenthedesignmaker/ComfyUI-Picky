// Browser-local aggregate counters. Never store field values, names, paths or prompts.
export const USAGE_PREFIX = "vp:usage:v1:";
export const LAYOUT = "four-tabs-v1";
const AREAS = ["scene", "audio", "inputs", "clip", "prompt", "findings", "cast", "usage", "generation", "review"];
const REASONS = ["manual", "contextual", "initial"];
const IDLE_MS = 60000;
const SESSION_MS = 30 * 60000;
const RETENTION_MS = 90 * 86400000;

export function createUsageTracker({ storage, now = Date.now, id = () => crypto.randomUUID() } = {}) {
  let record, key, area = "scene", lastTick = now(), lastActivity = now(), visible = true;
  let enabled = true, durable = true;
  let resetStamp = null;
  try { enabled = storage.getItem("vp:usage:enabled") !== "false"; } catch { durable = false; }
  try { resetStamp = storage.getItem("vp:usage:reset"); } catch { durable = false; }
  const fresh = () => {
    key = USAGE_PREFIX + id();
    record = { version: 1, layout: LAYOUT, started: now(), updated: now(), panels: {}, transitions: {}, outcomes: {} };
  };
  fresh();
  const sync = () => {
    try {
      const stamp = storage.getItem("vp:usage:reset");
      const preference = storage.getItem("vp:usage:enabled") !== "false";
      if (stamp !== resetStamp) { fresh(); resetStamp = stamp; lastTick = lastActivity = now(); }
      if (preference !== enabled) { enabled = preference; lastTick = lastActivity = now(); }
    } catch { durable = false; }
  };
  const panel = () => record.panels[area] ||= { visits: { manual: 0, contextual: 0, initial: 0 }, actions: {}, activeMs: 0 };
  const flush = () => {
    sync();
    if (!enabled) return;
    if (!Object.keys(record.panels).length && !Object.keys(record.outcomes).length) return;
    record.updated = now();
    try { storage.setItem(key, JSON.stringify(record)); durable = true; } catch { durable = false; }
  };
  const tick = () => {
    sync();
    const time = now();
    const elapsed = Math.max(0, Math.min(time, lastActivity + IDLE_MS) - lastTick);
    if (enabled && visible && elapsed > 0) panel().activeMs += elapsed;
    lastTick = time;
  };
  const activity = () => {
    tick();
    if (now() - lastActivity >= SESSION_MS) { flush(); fresh(); if (enabled) panel().visits.initial++; }
    lastActivity = now();
  };
  const visit = (next, reason = "manual") => {
    if (!AREAS.includes(next) || !REASONS.includes(reason)) return;
    activity();
    if (enabled && (next !== area || reason === "initial")) {
      if (reason !== "initial") {
        const transition = `${area}>${next}:${reason}`;
        record.transitions[transition] = (record.transitions[transition] || 0) + 1;
      }
      area = next;
      panel().visits[reason]++;
      flush();
    } else area = next;
  };
  const action = (next, kind) => {
    if (!AREAS.includes(next) || !["field_change", "text_edit", "command"].includes(kind)) return;
    activity();
    if (!enabled) return;
    area = next;
    panel().actions[kind] = (panel().actions[kind] || 0) + 1;
  };
  const outcome = (name) => {
    tick();
    if (!enabled || !["render_submitted", "render_rejected", "still_submitted", "still_rejected", "take_chosen", "render_blocked", "validation_cleared", "identity_reused", "reference_attached"].includes(name)) return;
    record.outcomes[name] = (record.outcomes[name] || 0) + 1;
    flush();
  };
  const records = () => {
    tick(); flush();
    const rows = [];
    try {
      const keys = Array.from({ length: storage.length }, (_, i) => storage.key(i)).filter(k => k?.startsWith(USAGE_PREFIX));
      for (const candidate of keys) {
        try {
          const value = JSON.parse(storage.getItem(candidate));
          if (value?.version !== 1 || !Number.isFinite(value.started) || !value.panels || !value.outcomes) continue;
          if (value.started < now() - RETENTION_MS) { storage.removeItem(candidate); continue; }
          rows.push(value);
        } catch { /* Ignore a damaged record. */ }
      }
    } catch { durable = false; }
    if (!durable && !rows.some(r => r.started === record.started)) rows.push(structuredClone(record));
    return rows;
  };
  const reset = () => {
    try {
      resetStamp = id(); storage.setItem("vp:usage:reset", resetStamp);
      const keys = Array.from({ length: storage.length }, (_, i) => storage.key(i));
      keys.filter(k => k?.startsWith(USAGE_PREFIX)).forEach(k => storage.removeItem(k));
    } catch { durable = false; }
    fresh(); lastTick = lastActivity = now();
  };
  return { visit, action, outcome, activity, tick, flush, records, reset,
    visibility(value) { tick(); visible = value; lastTick = now(); if (value) activity(); else flush(); },
    setEnabled(value) { tick(); flush(); enabled = value; lastTick = lastActivity = now(); try { storage.setItem("vp:usage:enabled", String(value)); } catch { durable = false; } },
    get enabled() { return enabled; }, get durable() { return durable; },
  };
}
