import { FEELINGS, ACTION_SNIPPETS, QUALITIES, buildIdeaPatch, directionText, referenceURL, previewMatches } from "./shot-ideas-data.js";

export function createShotIdeas({ snapshot, commit }) {
  const dialog = document.createElement("dialog");
  dialog.id = "shot-ideas";
  dialog.setAttribute("aria-labelledby", "shot-ideas-title");
  dialog.innerHTML = `
    <header class="ideas-head">
      <div><span class="eyebrow">A CINEMATIC CHEAT SHEET</span><h2 id="shot-ideas-title">Shot Ideas</h2></div>
      <button type="button" class="small ghost" id="ideas-close" aria-label="Close Shot Ideas">Close ×</button>
    </header>
    <div class="ideas-scroll">
      <p class="ideas-intro">Choose a feeling. Copy an action, borrow a visual detail, or combine a direction.</p>
      <div class="ideas-controls">
        <label for="ideas-feeling">Feeling<select id="ideas-feeling"></select></label>
        <label for="ideas-beat">For<select id="ideas-beat"></select></label>
      </div>
      <details class="ideas-context"><summary>Scene context</summary><p id="ideas-context-text"></p></details>
      <div id="ideas-browse">
        <label class="ideas-search-label" for="ideas-search">Reference search context <span>optional</span>
          <input id="ideas-search" maxlength="160" placeholder="e.g. office, dinner table, rainy street">
        </label>
        <p class="ideas-note">Added to reference searches. The approaches below are curated starting points.</p>
        <fieldset class="ideas-qualities"><legend>Visual qualities to combine or apply</legend><div id="ideas-qualities"></div></fieldset>
        <div id="ideas-cards"></div>
        <p class="ideas-note">Use the approach that fits your story. Color and camera movement can support more than one feeling.</p>
        <p class="ideas-note">More inspiration: <a href="https://film-grab.com/" target="_blank" rel="noopener noreferrer">FilmGrab ↗</a> · <a href="https://eyecannndy.com/" target="_blank" rel="noopener noreferrer">Eyecandy techniques ↗</a></p>
      </div>
      <section id="ideas-preview" hidden aria-labelledby="ideas-preview-title">
        <button type="button" id="ideas-back" class="small ghost">← Approaches</button>
        <h3 id="ideas-preview-title">Preview direction</h3>
        <p class="ideas-note">Composition and camera replace the selected fields. Color and lighting append to the action &amp; environment. Dialogue and timing stay as written.</p>
        <div id="ideas-diff"></div>
        <button type="button" id="ideas-apply" class="primary">Apply to this beat</button>
      </section>
      <div id="ideas-copy-fallback" hidden><label for="ideas-copy-text">Select and copy this snippet</label><textarea id="ideas-copy-text" readonly rows="5"></textarea></div>
    </div>
    <footer class="ideas-foot"><p id="ideas-status" role="status" aria-live="polite"></p><button type="button" id="ideas-undo" class="small ghost" hidden>Undo last apply</button></footer>`;
  document.body.append(dialog);
  const el = id => dialog.querySelector(`#ideas-${id}`);
  let scope = null;
  let preview = null;
  let undo = null;
  let returnFocus = null;
  const selected = () => [...el("qualities").querySelectorAll("input:checked")].map(input => input.value);
  const feeling = () => FEELINGS.find(item => item.id === el("feeling").value);
  const status = text => { el("status").textContent = text; };
  const addText = (parent, tag, text, className = "") => {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    parent.append(node);
    return node;
  };
  for (const item of FEELINGS) el("feeling").append(new Option(item.name, item.id));
  for (const [key, name] of QUALITIES) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = key;
    input.checked = true;
    label.append(input, document.createTextNode(name));
    el("qualities").append(label);
  }

  function updateContext() {
    const data = snapshot();
    const beat = data?.beats[Number(el("beat").value)];
    el("context-text").textContent = [data?.name, data?.overview, beat?.action]
      .filter(Boolean).join("\n\n") || "Explore freely, or add a storyboard shot to apply a direction.";
  }

  function fillBeats(data, preferred = el("beat").value) {
    el("beat").replaceChildren(new Option("Explore scene mood", "-1"));
    data.beats.forEach((beat, index) => {
      const snippet = (beat.action || beat.angle || "Untitled").replace(/\s+/g, " ").slice(0, 52);
      el("beat").append(new Option(`${data.labels[index]} · ${snippet}`, String(index)));
    });
    el("beat").value = data.beats[Number(preferred)] ? String(preferred) : "-1";
    updateContext();
  }

  function showBrowse() {
    preview = null;
    el("preview").hidden = true;
    el("browse").hidden = false;
  }

  function searchTerms(approach) {
    return [approach.search, el("search").value.trim()].filter(Boolean).join(" ");
  }

  function updateLinks() {
    for (const card of el("cards").children) {
      const approach = feeling().approaches.find(item => item.id === card.dataset.idea);
      const link = card.querySelector("a");
      link.href = referenceURL(searchTerms(approach));
      link.title = `Search Frame Set: ${searchTerms(approach)}`;
    }
  }

  async function copyText(text, description) {
    try {
      await navigator.clipboard.writeText(text);
      status(`${description} copied.`);
      el("copy-fallback").hidden = true;
    } catch {
      // Clipboard APIs are unavailable on plain HTTP/LAN installations.
      el("copy-fallback").hidden = false;
      el("copy-text").value = text;
      el("copy-text").focus();
      el("copy-text").select();
      status("Press Ctrl+C or ⌘C to copy the selected text.");
    }
  }

  function copyDirection(approach) {
    const text = directionText(approach, selected());
    if (!text) return status("Choose at least one quality to copy.");
    return copyText(text, "Visual direction");
  }

  function copyButton(parent, text, description) {
    const button = addText(parent, "button", "Copy", "small ghost ideas-copy-one");
    button.type = "button";
    button.setAttribute("aria-label", `Copy ${description}`);
    button.onclick = () => copyText(text, description);
  }

  function renderCards() {
    showBrowse();
    el("copy-fallback").hidden = true;
    el("cards").replaceChildren();
    const canApply = Number(el("beat").value) >= 0;
    feeling().approaches.forEach((approach, index) => {
      const card = document.createElement("article");
      card.className = "ideas-card";
      card.dataset.idea = approach.id;
      addText(card, "span", `APPROACH ${String(index + 1).padStart(2, "0")}`, "eyebrow");
      addText(card, "h3", approach.title);
      addText(card, "p", approach.effect, "ideas-effect");
      const snippets = document.createElement("details");
      snippets.className = "ideas-snippets";
      snippets.open = true;
      addText(snippets, "summary", "Action snippets");
      addText(snippets, "p", "Choose an intensity. Copy and adapt the action to your characters.", "ideas-note");
      for (const snippet of ACTION_SNIPPETS[feeling().id][approach.id]) {
        const item = document.createElement("div");
        item.className = "ideas-snippet";
        const head = document.createElement("div");
        head.className = "ideas-snippet-head";
        addText(head, "span", snippet.intensity, "ideas-intensity");
        copyButton(head, snippet.text, `action: ${snippet.title}`);
        item.append(head);
        addText(item, "h4", snippet.title);
        addText(item, "p", snippet.text, "ideas-snippet-text");
        snippets.append(item);
      }
      card.append(snippets);
      const list = document.createElement("dl");
      for (const [key, name] of QUALITIES) {
        const term = addText(list, "dt", name);
        copyButton(term, approach[key], `${name.toLowerCase()}: ${approach.title}`);
        addText(list, "dd", approach[key]);
      }
      card.append(list);
      const actions = document.createElement("div");
      actions.className = "ideas-actions";
      const previewButton = addText(actions, "button", "Preview direction", "small primary");
      previewButton.type = "button";
      previewButton.disabled = !canApply || !selected().length;
      previewButton.title = canApply ? "Review changes before applying" : "Select a storyboard beat to apply direction";
      previewButton.onclick = () => showPreview(approach);
      const copy = addText(actions, "button", "Copy direction", "small ghost");
      copy.type = "button";
      copy.disabled = !selected().length;
      copy.onclick = () => copyDirection(approach);
      const link = addText(actions, "a", "References ↗");
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.append(actions);
      el("cards").append(card);
    });
    updateLinks();
  }

  function showPreview(approach) {
    const data = snapshot();
    if (!data || data.key !== scope) return sync();
    const index = Number(el("beat").value);
    const beat = data.beats[index];
    if (!beat) return status("Select a storyboard beat first.");
    const patch = buildIdeaPatch(beat, approach, selected());
    if (!Object.keys(patch).length) return status("This beat already has the selected direction.");
    preview = { key: data.key, beats: JSON.stringify(data.beats), index, patch,
      before: Object.fromEntries(Object.keys(patch).map(key => [key, beat[key] ?? ""])) };
    el("diff").replaceChildren();
    el("preview-title").textContent = `${data.labels[index]} · ${approach.title}`;
    const names = { angle: "Composition", camera: "Camera", action: "Action & environment" };
    for (const [key, value] of Object.entries(patch)) {
      const row = document.createElement("div");
      row.className = "ideas-diff-field";
      addText(row, "h4", names[key]);
      addText(row, "span", "CURRENT", "eyebrow");
      addText(row, "p", beat[key] || "Empty", "ideas-before");
      addText(row, "span", "PROPOSED", "eyebrow");
      addText(row, "p", value, "ideas-after");
      el("diff").append(row);
    }
    el("browse").hidden = true;
    el("preview").hidden = false;
    el("apply").focus();
    el("preview").scrollIntoView({ block: "start" });
    status("Review the proposed direction, then apply when ready.");
  }

  function sync() {
    if (!dialog.open) return;
    const data = snapshot();
    if (!data || data.key !== scope) { dialog.close(); return; }
    fillBeats(data);
    if (preview && !previewMatches(data, preview)) {
      renderCards();
      status("The storyboard changed. Preview again before applying.");
    }
    if (undo && !previewMatches(data, undo)) {
      undo = null;
      el("undo").hidden = true;
    }
  }

  el("apply").onclick = () => {
    const pending = preview;
    if (!pending || !previewMatches(snapshot(), pending) || !commit(pending)) {
      sync();
      return status("The storyboard changed. Preview again before applying.");
    }
    const data = snapshot();
    undo = { key: data.key, beats: JSON.stringify(data.beats), index: pending.index, patch: pending.before };
    el("undo").hidden = false;
    renderCards();
    status("Applied to the selected beat. Use Save to keep your changes.");
    el("undo").focus();
  };
  el("undo").onclick = () => {
    const pending = undo;
    if (!pending || !previewMatches(snapshot(), pending) || !commit(pending)) {
      sync();
      return status("The storyboard changed; undo is no longer available.");
    }
    undo = null;
    el("undo").hidden = true;
    renderCards();
    status("Restored the previous direction. Use Save to keep your changes.");
    el("feeling").focus();
  };
  el("close").onclick = () => dialog.close();
  el("back").onclick = () => { renderCards(); el("cards").querySelector("button")?.focus(); };
  el("feeling").onchange = () => { renderCards(); status(""); };
  el("beat").onchange = () => { updateContext(); renderCards(); status(""); };
  el("qualities").onchange = renderCards;
  el("search").oninput = updateLinks;
  // Keep editor transport, delete and generation shortcuts behind the modal.
  // Native dialog handles focus trapping and Escape; Ctrl+S still saves normally.
  dialog.addEventListener("keydown", event => {
    if (!((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s")) event.stopPropagation();
  });
  dialog.addEventListener("close", () => {
    preview = null;
    returnFocus?.focus({ preventScroll: true });
  });
  return {
    sync,
    open() {
      const data = snapshot();
      if (!data) return;
      returnFocus = document.activeElement;
      if (scope !== data.key) {
        undo = null;
        el("undo").hidden = true;
        el("search").value = "";
      }
      scope = data.key;
      fillBeats(data, "-1");
      renderCards();
      status(data.beats.length ? "Select a beat to preview and apply direction." : "Add a storyboard shot to apply direction. You can still browse and copy.");
      if (!dialog.open) dialog.showModal();
      sync();
      el("feeling").focus();
    },
  };
}
