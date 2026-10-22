"""Optional model-only LoRA stacks for prepared video workflows."""
from __future__ import annotations

import copy
import json
import math
import re
import struct
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any


def normalize_shot(value: Any) -> dict[str, Any] | None:
    """None inherits; preserve custom rows when switching off or to defaults."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("mode") not in ("inherit", "custom", "off"):
        raise ValueError("Shot LoRAs need mode inherit, custom, or off.")
    return {"mode": value["mode"], "loras": normalize(value.get("loras", []))}


def installed_names() -> list[str] | None:
    """None outside ComfyUI; an empty list means ComfyUI has no LoRAs."""
    try:
        import folder_paths
    except ImportError:
        return None
    return sorted(folder_paths.get_filename_list("loras"), key=str.casefold)


@lru_cache(maxsize=1024)
def _h3_metadata(path: str, modified_ns: int, size: int) -> bool:
    """Inspect bounded JSON headers only; never deserialize tensor payloads."""
    try:
        with open(path, "rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 2 <= header_size <= min(16 * 1024 * 1024, size - 8):
                return False
            header = json.loads(handle.read(header_size))
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            return False
        for field in ("base_model", "ss_base_model_version", "modelspec.architecture"):
            value = metadata.get(field)
            if isinstance(value, str) and re.search(r"\bminimax[-_ /]*h3(?=$|[-_ /])", value, re.I):
                return True
    except (OSError, ValueError, struct.error, AttributeError):
        pass
    return False


def h3_catalog() -> dict[str, Any]:
    """Only identified H3 adapters belong in this H3 editor's video picker.

    Unknown files stay hidden; a filename alone cannot establish compatibility.
    Existing saved stacks and ComfyUI's shared LoRA folders are not modified.
    """
    names = installed_names()
    result = {"loras": [], "available": names is not None,
              "model_family": "minimax-h3", "hidden_count": 0}
    if names is None:
        return result
    import folder_paths
    for name in names:
        if name.lower().endswith(".safetensors") and "turbo" not in name.lower():
            path = folder_paths.get_full_path("loras", name)
            if path:
                try:
                    stat = Path(path).stat()
                    if _h3_metadata(str(path), stat.st_mtime_ns, stat.st_size):
                        result["loras"].append(name)
                        continue
                except OSError:
                    pass
        result["hidden_count"] += 1
    return result


def normalize(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Video LoRAs must be a list with at most 8 entries.")
    result = []
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Each video LoRA must be an object.")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Choose a file for each video LoRA.")
        name = name.strip()
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError("LoRA files must be names from ComfyUI's LoRA folder.")
        if name in seen:
            raise ValueError(f"LoRA {name!r} is listed twice; use one weight instead.")
        seen.add(name)
        enabled = row.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("LoRA enabled must be true or false.")
        weight = row.get("strength_model", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("LoRA weight must be a number between -2 and 2.")
        weight = float(weight)
        if not math.isfinite(weight) or not -2 <= weight <= 2:
            raise ValueError("LoRA weight must be a finite number between -2 and 2.")
        result.append({"name": name, "strength_model": weight, "enabled": enabled})
    return result


def apply(workflow: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict, list]:
    """Patch the shared model root, so every pass and existing adapter sees it."""
    active = [r for r in normalize(rows) if r["enabled"] and r["strength_model"] != 0]
    if not active:
        return workflow, []
    for row in active:
        if "turbo" in row["name"].lower():
            raise ValueError("Keep Turbo in the render profile and control it with the quality tier, not the additional LoRA stack.")
        if any(
            n.get("inputs", {}).get("lora_name") == row["name"]
            and n.get("inputs", {}).get("strength_model", 1.0) != 0
            for n in workflow.values()
        ):
            raise ValueError(f"LoRA {row['name']!r} is already loaded by the render profile.")
    available = installed_names()
    if available is not None:
        missing = [r["name"] for r in active if r["name"] not in available]
        if missing:
            raise ValueError("LoRA file not installed: " + ", ".join(missing))
    roots = [k for k, n in workflow.items() if n.get("class_type") in
             {"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"}]
    if len(roots) != 1:
        raise ValueError("Video LoRAs require a profile with one model loader; this profile has an unsupported model layout.")
    graph = copy.deepcopy(workflow)
    root = roots[0]
    existing = list(graph.values())
    link = [root, 0]
    for index, row in enumerate(active):
        key = f"vp_video_lora_{index}"
        while key in graph:
            key += "_"
        graph[key] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": link, "lora_name": row["name"], "strength_model": row["strength_model"],
        }, "_meta": {"title": f"Video LoRA: {row['name']}"}}
        link = [key, 0]
    for node in existing:
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] == root and value[1] == 0:
                node["inputs"][key] = link
    return graph, active
