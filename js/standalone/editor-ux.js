import { createUsageTracker, LAYOUT } from "./usage.js";

const names = { scene: "Scene", audio: "Audio", inputs: "Inputs", clip: "Clip", prompt: "Prompt", findings: "Validation", cast: "Project library", usage: "Usage", generation: "Generation", review: "Review" };
const areaForTab = value => ({ media: "inputs", source: "clip" }[value] || value);

export function setupEditorUX({ refit, uuid, renderLibrary }) {
  const $ = id => document.getElementById(id);
  let storage;
  try { storage = localStorage; } catch { /* Tracker falls back to memory. */ }
  const tracker = createUsageTracker({ storage, id: uuid });
  const editor = $("editor");
  let active = "scene", utility = null, returnFocus = null, contextKey = null, hadErrors = null;
  let contextEnabled = false;
  tracker.visibility(false);
  const scope = node => (node.closest("#castdetail") ? "cast" : null)
    || (node.closest("#review-pane") ? "review" : null)
    || node.closest("[data-utility]")?.dataset.utility
    || areaForTab(node.closest("[data-panel]")?.dataset.panel)
    || node.closest("[data-usage-area]")?.dataset.usageArea || active;

  function showUsage() {
    const rows = tracker.records().filter(row => row.layout === LAYOUT);
    const host = $("usage-summary");
    host.replaceChildren();
    const note = document.createElement("p");
    note.textContent = `${rows.length} recorded sessions · ${LAYOUT}${tracker.durable ? "" : " · Storage unavailable: counters are temporary"}`;
    host.append(note);
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Area</th><th>Manual visits</th><th>Sessions</th><th>Actions</th><th>Active min</th></tr></thead>";
    const body = document.createElement("tbody");
    for (const [area, name] of Object.entries(names)) {
      const entries = rows.map(row => row.panels[area]).filter(Boolean);
      const visits = entries.reduce((n, entry) => n + (entry.visits?.manual || 0), 0);
      const actions = entries.reduce((n, entry) => n + Object.values(entry.actions || {}).reduce((a, b) => a + b, 0), 0);
      const minutes = entries.reduce((n, entry) => n + (entry.activeMs || 0), 0) / 60000;
      const tr = document.createElement("tr");
      for (const value of [name, visits, `${entries.length}/${rows.length}`, actions, minutes.toFixed(1)]) {
        const td = document.createElement("td"); td.textContent = value; tr.append(td);
      }
      body.append(tr);
    }
    table.append(body); host.append(table);
    const totals = {};
    for (const row of rows) for (const [name, count] of Object.entries(row.outcomes)) totals[name] = (totals[name] || 0) + count;
    const outcomes = document.createElement("p");
    outcomes.textContent = Object.entries(totals).map(([name, count]) => `${name.replaceAll("_", " ")}: ${count}`).join(" · ") || "No task outcomes recorded yet.";
    host.append(outcomes);
    const hint = document.createElement("p");
    hint.className = "usage-description";
    hint.textContent = "Export includes initial/contextual visits and transition counts. Actions are committed field changes, text-edit bursts and commands, not confirmed saves. Records start with this layout; there is no earlier baseline. External API edits are excluded.";
    host.append(hint);
    $("usage-enabled").checked = tracker.enabled;
  }

  function closeUtility(restore = true, trackReturn = true) {
    if (!utility) return;
    $("utility-" + utility).hidden = true;
    document.querySelectorAll("[data-open-utility]").forEach(button => button.setAttribute("aria-expanded", "false"));
    utility = null;
    if (trackReturn) tracker.visit(active, "contextual");
    if (restore) returnFocus?.focus();
    refit();
  }

  function openUtility(name, trigger, reason = "manual") {
    if (utility === name) { closeUtility(); return; }
    closeUtility(false, false);
    utility = name; returnFocus = trigger || document.activeElement;
    const panel = $("utility-" + name);
    if (name === "cast") renderLibrary?.();
    panel.hidden = false;
    document.querySelectorAll("[data-open-utility]").forEach(button => button.setAttribute("aria-expanded", String(button.dataset.openUtility === name)));
    tracker.visit(name, reason);
    if (name === "usage") showUsage();
    panel.querySelector("[data-close-utility]").focus();
    refit();
  }

  document.querySelectorAll("[data-open-utility]").forEach(button => {
    button.setAttribute("aria-controls", "utility-" + button.dataset.openUtility);
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", event => openUtility(button.dataset.openUtility, button, event.isTrusted ? "manual" : "contextual"));
  });
  document.querySelectorAll("[data-close-utility]").forEach(button => {
    button.onclick = () => closeUtility();
    if (button.closest(".project-library")) return;
    const size = document.createElement("button"); size.className = "small ghost";
    size.textContent = "Expand panel"; size.setAttribute("aria-pressed", "false");
    size.onclick = () => {
      const expanded = size.getAttribute("aria-pressed") !== "true";
      button.closest(".editor-utility").style.height = expanded ? "50vh" : "280px";
      size.textContent = expanded ? "Shrink panel" : "Expand panel";
      size.setAttribute("aria-pressed", String(expanded)); refit();
    };
    button.before(size);
  });
  editor.addEventListener("keydown", event => {
    if (event.key === "Escape" && utility && !event.defaultPrevented) {
      // Other modal sheets handle their own Escape; this panel is within the editor.
      event.preventDefault(); event.stopImmediatePropagation(); closeUtility();
    }
  });
  // The cast detail sheet and docked take review live outside #editor.
  // Observe their controls too, using one listener so expanded review is not double-counted.
  const listen = (type, handler, options) => document.addEventListener(type, event => {
    if (event.target.closest?.("#editor, #utility-cast, #castdetail, #review-pane")) handler(event);
  }, options);
  const edited = new WeakSet();
  listen("input", event => {
    if (!event.isTrusted || !contextEnabled) return;
    tracker.activity();
    if (!edited.has(event.target) && event.target.matches('textarea, [contenteditable="plaintext-only"]')) {
      edited.add(event.target); tracker.action(scope(event.target), "text_edit");
    }
  });
  listen("focusout", event => edited.delete(event.target));
  listen("focusin", event => {
    if (contextEnabled && event.target.matches("input, textarea, select, [contenteditable]")) tracker.visit(scope(event.target), "contextual");
  });
  listen("change", event => {
    if (event.isTrusted && contextEnabled && !event.target.matches("textarea")) tracker.action(scope(event.target), "field_change");
  }, true);
  listen("click", event => {
    if (!event.isTrusted || !contextEnabled) return;
    tracker.activity();
    const button = event.target.closest("button");
    if (button && !button.matches("[data-tab], [data-open-utility], [data-close-utility]")) tracker.action(scope(button), "command");
  });
  ["pointerdown", "keydown", "wheel"].forEach(type => listen(type, event => {
    if (event.isTrusted && contextEnabled) tracker.activity();
  }, { passive: true }));
  document.addEventListener("visibilitychange", () => tracker.visibility(contextEnabled && !document.hidden && document.hasFocus()));
  window.addEventListener("blur", () => tracker.visibility(false));
  window.addEventListener("focus", () => tracker.visibility(contextEnabled && !document.hidden));
  window.addEventListener("pagehide", () => { tracker.tick(); tracker.flush(); });
  setInterval(() => { if (contextEnabled) { tracker.tick(); tracker.flush(); } }, 15000);

  $("usage-enabled").onchange = event => { tracker.setEnabled(event.target.checked); showUsage(); };
  $("usage-reset").onclick = () => {
    tracker.reset(); $("usage-export-panel").hidden = true; $("usage-export-data").value = "";
    if (exportUrl) { URL.revokeObjectURL(exportUrl); exportUrl = null; }
    $("usage-download").removeAttribute("href"); showUsage();
  };
  let exportUrl = null;
  $("usage-export").onclick = () => {
    const text = JSON.stringify({ exported: new Date().toISOString(), layout: LAYOUT, sessions: tracker.records() }, null, 2);
    $("usage-export-data").value = text;
    $("usage-export-panel").hidden = false;
    if (exportUrl) URL.revokeObjectURL(exportUrl);
    exportUrl = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    $("usage-download").href = exportUrl;
    $("usage-export-data").focus(); $("usage-export-data").select();
  };
  new MutationObserver(() => { $("prompt-override-status").hidden = $("preview-dirty-hint").hidden; })
    .observe($("preview-dirty-hint"), { attributes: true, attributeFilter: ["hidden"] });

  return {
    tracker, openUtility,
    navigate(tab, reason) { closeUtility(false, false); active = areaForTab(tab); tracker.visit(active, reason); },
    context(source, depth) {
      const next = source ? `${source}:${depth}` : null;
      contextEnabled = !!source;
      if (next === contextKey) return;
      contextKey = next; hadErrors = null;
      closeUtility(false, false);
      tracker.visibility(contextEnabled && !document.hidden && document.hasFocus());
      if (source) tracker.visit(active, "initial");
    },
    findings(all) {
      const errors = all.filter(f => f.severity === "error").length;
      const warnings = all.filter(f => f.severity === "warning").length;
      const info = all.length - errors - warnings;
      const countLabel = (count, singular) => count && `${count} ${singular}${count === 1 ? "" : "s"}`;
      $("findings-count").textContent = all.length ? [countLabel(errors, "error"), countLabel(warnings, "warning"), countLabel(info, "note")].filter(Boolean).join(" · ") : "Checks clear";
      $("validation-open").classList.toggle("has-errors", errors > 0);
      if (contextEnabled && hadErrors && !errors) tracker.outcome("validation_cleared");
      hadErrors = errors > 0;
    },
  };
}
