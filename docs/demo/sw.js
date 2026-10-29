// Picky demo: map the editor's absolute URLs onto this static folder.
//
// The editor asks for media at ComfyUI's `/view?filename=…` and for its own
// files at `/video_prompter/app/…`, both from the origin root. On GitHub Pages
// the demo lives under a sub-path, so this worker sends those requests to the
// re-encoded copies beside it. API calls never get here: the page's own fetch
// shim answers them.

const BASE = new URL("./", self.location).href;
let media = null;

async function mediaMap() {
  if (!media) {
    const response = await fetch(BASE + "data/media.json");
    const raw = await response.json();
    media = {};
    for (const [key, file] of Object.entries(raw)) {
      media[key] = file;
      media[key.split("|").pop()] = file; // filename alone, as a fallback
    }
  }
  return media;
}

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || event.request.method !== "GET") return;

  if (url.pathname.endsWith("/view")) {
    event.respondWith((async () => {
      const map = await mediaMap();
      const name = url.searchParams.get("filename") || "";
      const key = `${url.searchParams.get("type") || "output"}|${url.searchParams.get("subfolder") || ""}|${name}`;
      const file = map[key] || map[name];
      if (!file) return new Response("Not in the demo snapshot.", { status: 404 });
      const headers = {};
      const range = event.request.headers.get("range");
      if (range) headers.range = range;
      return fetch(BASE + "media/" + file, { headers });
    })());
    return;
  }

  const app = url.pathname.indexOf("/video_prompter/app/");
  if (app >= 0 && !url.pathname.startsWith(new URL(BASE).pathname + "app/")) {
    const rest = url.pathname.slice(app + "/video_prompter/app/".length);
    event.respondWith(fetch(BASE + "app/" + rest + url.search));
  }
});
