// ComfyUI workflow graph & node progress tracking for MiniMax H3 renders.
// Extracted from main.js to keep pipeline progress calculation modular and testable.

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function getStageDescription(workflow, nodeId) {
  return getNodeStage(workflow, nodeId)?.label || null;
}

// ComfyUI reports the node that is actually executing. Keep these labels tied
// to node classes rather than elapsed-time guesses so a slow model load, VAE
// decode, or video encode is never presented as a diffusion sampling step.
export function getNodeStage(workflow, nodeId) {
  if (!nodeId || !workflow || typeof workflow !== "object") return null;
  const id = String(nodeId);
  const node = workflow[id];
  if (!node) return { key: "node", label: `Executing node ${id}`, detail: `Node ${id}` };

  const cls = String(node.class_type || "");
  const title = String(node._meta?.title || cls || `Node ${id}`);
  const kind = cls.toLowerCase();
  const named = title.toLowerCase();

  if (kind.includes("loadimage") || kind.includes("loadvideo") || kind.includes("loadlatent")) {
    return { key: "source", label: "Loading source media", detail: title };
  }
  if (kind.includes("unetloader") || kind.includes("cliploader") || kind.includes("vaeloader") ||
      kind.includes("modelload") || (kind.includes("loader") && named.includes("model"))) {
    return { key: "models", label: "Loading models", detail: title };
  }
  if (kind.includes("imagescale") || kind.includes("resolutionselector") || kind.includes("getimagesize")) {
    return { key: "source", label: "Preparing source image", detail: title };
  }
  if (kind.includes("imagetovideo") || kind.includes("referencetovideo") || kind.includes("texttovideo") ||
      kind.includes("motioncontext") || kind.includes("textencode")) {
    return { key: "conditioning", label: "Encoding prompt and conditioning", detail: title };
  }
  if (kind.includes("randomnoise") || kind.includes("guider") || kind.includes("samplerselect") ||
      kind.includes("scheduler")) {
    return { key: "prepare", label: "Preparing diffusion", detail: title };
  }
  if (kind.includes("sampler")) {
    return { key: "sampling", label: "Generating motion and audio", detail: title };
  }
  // Audio must be checked first because VAEDecodeAudio also contains VAEDecode.
  if (kind.includes("vaedecodeaudio") || kind.includes("audiodecode")) {
    return { key: "audio-decode", label: "Decoding audio", detail: title };
  }
  if (kind.includes("vaeencode")) {
    return { key: "encode", label: "Encoding continuation context", detail: title };
  }
  if (kind.includes("vaedecode")) {
    return { key: "video-decode", label: "Decoding video frames", detail: title };
  }
  if (kind.includes("trimaudio") || kind.includes("imagefrombatch")) {
    return { key: "trim", label: "Trimming continuation context", detail: title };
  }
  if (kind.includes("createvideo") || kind.includes("combinevideo")) {
    return { key: "mux", label: "Assembling audio and video", detail: title };
  }
  if (kind.includes("savevideo")) {
    return { key: "save", label: "Encoding and saving video", detail: title };
  }
  if (kind.includes("savelatent")) {
    return { key: "save-context", label: "Saving continuation context", detail: title };
  }
  return { key: "node", label: title || `Executing node ${id}`, detail: cls || `Node ${id}` };
}

export function resolveWorkflowNodeId(workflow, ...candidates) {
  for (const candidate of candidates) {
    if (candidate == null) continue;
    const id = String(candidate);
    if (workflow && Object.prototype.hasOwnProperty.call(workflow, id)) return id;
  }
  const fallback = candidates.find((candidate) => candidate != null);
  return fallback == null ? null : String(fallback);
}

export const PIPELINE_GROUP_WEIGHTS = {
  setup: 10,
  sampling: 70,
  decoding: 15,
  finishing: 5,
};

export function linkedWorkflowNodes(workflow, value, found = new Set()) {
  if (Array.isArray(value)) {
    if (value.length === 2 && typeof value[0] === "string" &&
        Object.prototype.hasOwnProperty.call(workflow, value[0]) && typeof value[1] === "number") {
      found.add(String(value[0]));
    } else {
      value.forEach((item) => linkedWorkflowNodes(workflow, item, found));
    }
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => linkedWorkflowNodes(workflow, item, found));
  }
  return found;
}

export function workflowAncestors(workflow, nodeIds) {
  const ancestors = new Set();
  const visit = (nodeId) => {
    const node = workflow?.[nodeId];
    if (!node) return;
    for (const dependency of linkedWorkflowNodes(workflow, node.inputs || {})) {
      if (ancestors.has(dependency)) continue;
      ancestors.add(dependency);
      visit(dependency);
    }
  };
  for (const nodeId of nodeIds || []) visit(String(nodeId));
  return ancestors;
}

export function executableWorkflowNodes(job) {
  const workflow = job?.workflow || {};
  const workflowIds = Object.keys(workflow);
  if (!workflowIds.length) return [];
  let outputs = (job?.outputNodes || []).map(String).filter((id) => workflow[id]);
  if (!outputs.length) {
    outputs = workflowIds.filter((id) => {
      const key = getNodeStage(workflow, id)?.key;
      return key === "save" || key === "save-context";
    });
  }
  if (!outputs.length) return workflowIds;
  const executable = workflowAncestors(workflow, outputs);
  outputs.forEach((id) => executable.add(id));
  return workflowIds.filter((id) => executable.has(id));
}

export function pipelineGroupForNode(workflow, nodeId, samplingAncestors) {
  const key = getNodeStage(workflow, nodeId)?.key || "node";
  if (key === "sampling") return "sampling";
  if (key === "video-decode" || key === "audio-decode") return "decoding";
  if (key === "mux" || key === "save" || key === "save-context") return "finishing";
  // Continuation graphs trim both before and after sampling. A trim feeding a
  // sampler is setup; a trim downstream from decoding is finishing.
  if (key === "trim" && !samplingAncestors.has(nodeId)) return "finishing";
  return "setup";
}

export function calculateOverallProgress(job, activeNodeId, activeProgress) {
  const workflow = job?.workflow || {};
  const nodeIds = executableWorkflowNodes(job);
  if (!nodeIds.length) return Math.max(0, Number(job?.overallProgress) || 0);

  const samplingIds = nodeIds.filter((id) => getNodeStage(workflow, id)?.key === "sampling");
  const samplingAncestors = workflowAncestors(workflow, samplingIds);
  const grouped = { setup: [], sampling: [], decoding: [], finishing: [] };
  nodeIds.forEach((id) => grouped[pipelineGroupForNode(workflow, id, samplingAncestors)].push(id));

  const presentWeight = Object.entries(grouped).reduce(
    (total, [group, ids]) => total + (ids.length ? PIPELINE_GROUP_WEIGHTS[group] : 0),
    0,
  ) || 100;
  const completed = new Set((job?.cachedNodes || []).map(String));
  for (const progress of Object.values(job?.nodeProgress || {})) {
    const id = resolveWorkflowNodeId(
      workflow,
      progress?.display_node_id,
      progress?.real_node_id,
      progress?.node_id,
    );
    if (id && progress?.state === "finished") completed.add(id);
  }
  // If a node is executing, every dependency feeding it has already finished
  // or came from cache, even for a browser that joined midway through the job.
  if (activeNodeId && workflow[activeNodeId]) {
    workflowAncestors(workflow, [activeNodeId]).forEach((id) => completed.add(id));
  }

  const activeValue = Number(activeProgress?.value);
  const activeMax = Number(activeProgress?.max);
  const activeFraction = Number.isFinite(activeValue) && Number.isFinite(activeMax) && activeMax > 0
    ? Math.max(0, Math.min(1, activeValue / activeMax))
    : 0;
  let raw = 0;
  for (const [group, ids] of Object.entries(grouped)) {
    if (!ids.length) continue;
    const groupWeight = (PIPELINE_GROUP_WEIGHTS[group] / presentWeight) * 100;
    const nodeWeight = groupWeight / ids.length;
    for (const id of ids) {
      if (completed.has(id)) raw += nodeWeight;
      else if (id === activeNodeId) raw += nodeWeight * activeFraction;
    }
  }

  // Execution events can arrive twice or briefly out of order because this
  // editor also receives the broadcast copy. The user-facing bar never moves
  // backward, and stays below 100 until ComfyUI reports success.
  const previous = Math.max(0, Number(job?.overallProgress) || 0);
  const overall = Math.max(previous, Math.min(99, raw));
  job.overallProgress = overall;
  return overall;
}

export function getJobStageInfo(job) {
  const workflow = job?.workflow || {};
  let nodeId = resolveWorkflowNodeId(workflow, job?.currentNode);
  let activeProgress = null;

  for (const progress of Object.values(job?.nodeProgress || {})) {
    if (progress?.state !== "running") continue;
    const progressNodeId = resolveWorkflowNodeId(
      workflow,
      progress.display_node_id,
      progress.real_node_id,
      progress.node_id,
    );
    if (!nodeId || progressNodeId === nodeId) {
      nodeId = progressNodeId || nodeId;
      activeProgress = progress;
      break;
    }
  }

  const legacy = job?.progress;
  if (!activeProgress && legacy && legacy.value != null && legacy.max != null &&
      (nodeId || legacy.node != null || legacy.display_node != null)) {
    const progressNode = resolveWorkflowNodeId(workflow, legacy.display_node, legacy.node);
    // A legacy progress event belongs to its own node. Never carry a completed
    // sampler's step count forward into the VAE or SaveVideo stages.
    if (!nodeId || !progressNode || progressNode === nodeId) activeProgress = legacy;
  }

  const stage = getNodeStage(workflow, nodeId) || {
    key: "starting",
    label: "Starting render pipeline",
    detail: "MiniMax H3",
  };
  const value = Number(activeProgress?.value);
  const max = Number(activeProgress?.max);
  const hasProgress = Number.isFinite(value) && Number.isFinite(max) && max > 0;
  const pct = hasProgress ? Math.max(0, Math.min(100, Math.round((value / max) * 100))) : null;
  let metric = "";
  if (hasProgress) {
    metric = stage.key === "sampling"
      ? `Step ${value}/${max} (${pct}%)`
      : `Stage progress ${value}/${max} (${pct}%)`;
  }
  const overallPct = Math.round(calculateOverallProgress(job, nodeId, activeProgress));
  return { ...stage, nodeId, value, max, pct, metric, overallPct };
}

export function applyProgressFill(fill, stage) {
  if (!fill) return;
  fill.classList.remove("indeterminate");
  fill.style.width = `${stage.overallPct}%`;
  fill.setAttribute("aria-valuenow", String(stage.overallPct));
  fill.setAttribute("aria-valuemin", "0");
  fill.setAttribute("aria-valuemax", "100");
  fill.title = `Overall render progress: ${stage.overallPct}%`;
}

export function stageMarkup(stage) {
  const metric = stage.metric ? ` · ${escapeHtml(stage.metric)}` : "";
  return `<b>${escapeHtml(stage.label)}</b>${metric}`;
}
