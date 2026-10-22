"""Manually unload ComfyUI's cached models and wait for the VRAM to come back.

**Do not call this before a render.** It was written to fix an apparent
memory-spill slowdown, and a 16-run interleaved A/B at 576x352 / 8 steps /
refs=match showed the opposite — freeing first is reliably *slower*:

    seed change     47.8s median without   ->  53.9s with
    prompt change   60.3s median without   ->  73.9s with

Renders starting with only 0.8-1.3GiB free VRAM showed no penalty, so ComfyUI's
own swapping is already doing the right thing and pre-emptying the card only
forces a reload. (Windows' "GPU Adapter Memory / Shared Usage" counter reads
~25GB with the GPU idle at 0% and torch holding 0.1GB — it tracks reserved
address space, not thrashing, and should not be used to diagnose this.)

What it is still good for: clearing the card by hand before running a different
large model, or recovering after a genuine OOM. Two ComfyUI details it gets
right, which are easy to trip over:

- It sends ``unload_models`` alone, never ``free_memory``. ``free_memory`` also
  calls ``PromptExecutor.reset()``, wiping the node output cache and forcing the
  32B text encoder to re-run on the next render.
- ``/free`` only sets a queue flag. The worker reads a queue item *before* it
  reads flags, and the ``soft_empty_cache()`` that actually returns the VRAM is
  deferred by ``main.py``'s ``gc_collect_interval`` (10s). So this polls past
  that interval instead of trusting an early plateau, and refuses to act while a
  render is in flight.
"""

from __future__ import annotations

import json
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8188"

# The H3 UNET alone is ~19.5GiB; leave a little room for activations.
NEED_GB = 21.0

# main.py's gc_collect_interval: how long ComfyUI may defer the cache flush.
GC_INTERVAL = 10.0


def _get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 30.0) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout):
        pass


def vram_free_gb(base_url: str = DEFAULT_URL) -> float:
    """Free VRAM in GB, as ComfyUI reports it."""
    stats = _get_json(f"{base_url}/system_stats")
    return stats["devices"][0]["vram_free"] / 1e9


def worker_busy(base_url: str = DEFAULT_URL) -> bool:
    """True when ComfyUI is mid-render, so a free flag cannot be consumed yet."""
    queue = _get_json(f"{base_url}/queue")
    return bool(queue.get("queue_running"))


def free_vram(
    base_url: str = DEFAULT_URL,
    need_gb: float = NEED_GB,
    max_wait: float = 60.0,
    verbose: bool = True,
) -> float:
    """Unload cached models and block until the VRAM is actually back.

    Returns the free VRAM in GB. A no-op (bar one status call) when the card
    already has ``need_gb`` free, so this is safe to call before every render.
    """
    before = vram_free_gb(base_url)
    if before >= need_gb:
        if verbose:
            print(f"[free] {before:.1f}GB free already, nothing to unload")
        return before

    # The worker only reads flags between queue items, so freeing now is
    # impossible and the low reading is simply the running render's own weights.
    # Say that, rather than warning about a spill that is not happening.
    if worker_busy(base_url):
        if verbose:
            print(
                f"[free] a render is in flight ({before:.1f}GB free); "
                "skipping — free before queueing, not during"
            )
        return before

    _post_json(f"{base_url}/free", {"unload_models": True})

    # The flag is consumed by the worker loop, not by the request, so poll.
    #
    # Do not trust an early plateau. unload_all_models() drops the models but
    # only sets need_gc; the soft_empty_cache() that actually hands the VRAM back
    # is deferred by main.py's gc_collect_interval (10s). Poll flat for less than
    # that and you see the *pre*-flush number and wrongly conclude it failed.
    deadline = time.monotonic() + max_wait
    plateau_after = time.monotonic() + GC_INTERVAL + 2.0
    last = before
    stable = 0
    while time.monotonic() < deadline:
        time.sleep(0.5)
        now = vram_free_gb(base_url)
        if now >= need_gb:
            last = now
            break
        stable = stable + 1 if now <= last + 0.05 else 0
        last = max(last, now)
        if stable >= 4 and time.monotonic() > plateau_after:
            break

    if verbose:
        print(f"[free] VRAM {before:.1f}GB -> {last:.1f}GB free")
    if last < need_gb:
        print(
            f"[free] WARNING: only {last:.1f}GB free, need ~{need_gb:.0f}GB. "
            "This render will likely spill into shared memory and run slow."
        )
    return last


if __name__ == "__main__":
    free_vram()
