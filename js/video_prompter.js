// Canvas extension: a launcher for the editor and a capture button for render
// profiles. Deliberately thin — no editor code lives here.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const API = "/video_prompter";
const NODE = "VideoPromptNode";

async function captureProfile(node) {
  const name = window.prompt("Name this render profile", "H3 local");
  if (!name) return;

  // Ask the frontend for the graph in both shapes: API format is what gets
  // submitted, the canvas workflow is what makes the profile re-capturable.
  const { workflow, output } = await app.graphToPrompt();

  try {
    const response = await api.fetchApi(`${API}/profile/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, api_format: output, canvas_workflow: workflow }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);

    // `describes` says which roles the profile drives and which the workflow
    // keeps for itself — the useful thing to see right after capturing.
    alert(body.describes);
  } catch (error) {
    alert(`Could not capture profile: ${error.message}`);
  }
}

app.registerExtension({
  name: "videoPrompter.launcher",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);

      this.addWidget("button", "Open Picky", null, () => {
        window.open(`${API}/app`, "_blank", "noopener");
      });

      this.addWidget("button", "Capture render profile", null, () => {
        captureProfile(this);
      });

      return this;
    };
  },
});
