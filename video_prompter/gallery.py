"""What came back: finished renders, recorded per project.

Takes reference ComfyUI's own output files rather than copying them. A 15s 2K
clip is megabytes, ComfyUI already owns the file and serves it from `/view`,
and duplicating it into the project would double the disk cost of every render.

Kept in `takes.json` beside `project.json` rather than inside it, so output
history never trips the project store's shrink heuristic and a project's
authored content stays separable from its results.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

from . import paths
from .projects import ProjectStore, utc_now_iso, write_json_atomic

TAKES_JSON = "takes.json"

# SaveVideo reports through `ui.PreviewVideo`, whose `as_dict` is
# `{"images": [...], "animated": True}` — so video outputs arrive under the
# same key as stills, and are told apart by extension.
OUTPUT_KEYS = ("images", "gifs", "videos")

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif")

# Normal structured H3 prompts carry one of these official document fields.
# A short take name such as "Herd A (Revised Narration)" carries none of them
# and is a sign that a caller put a label into queue metadata instead of the
# actual prompt. SaveVideo also embeds the complete Comfy workflow in the MP4's
# `mdta:prompt` tag, which gives us a lossless recovery path.
H3_PROMPT_MARKERS = (
    "integrated_multimodal_description:",
    "subject_definitions:",
    "detailed_description:",
)

# Where this tool's metadata rides in the ComfyUI queue payload.
EXTRA_DATA_KEY = "video_prompter"

# H3 renders at a fixed 24fps, so a measured duration implies its frame count.
FPS = 24.0


@dataclass
class Take:
    """One completed render, as a pointer into ComfyUI's output directory."""

    prompt_id: str
    filename: str
    shot: str = ""
    subfolder: str = ""
    type: str = "output"
    created: str = field(default_factory=utc_now_iso)
    frames: int = 0
    seconds: float = 0.0
    seed: int | None = None
    prompt_text: str = ""
    # The canvas this take rendered at, frozen so a fine re-shoot can scale its
    # ratio up to the full-quality budget.
    width: int = 0
    height: int = 0
    rating: int = 0
    discarded: bool = False
    # Whether anyone has looked at this take yet. A fresh render lands unseen
    # and the editor badges it, so a batch queued across several sources says
    # which of them came back with something new. Takes recorded before this
    # field existed read back as seen (`from_dict` below), because backfilling
    # a whole project's history as "new" would badge everything and mean
    # nothing.
    seen: bool = False
    # The quality tier this take was ASKED for, by name. Recorded because the
    # numbers cannot always be read backwards: a continuation renders at the
    # canvas of the clip it continues, so a `preview` extend of a 720p parent
    # is 1.0 MP at 8 steps — which is `turbo_hd`'s signature exactly, and the
    # gallery duly badged it Turbo HD. Empty on takes rendered before this was
    # recorded; the editor falls back to reading the numbers for those.
    quality: str = ""
    # Number of sampling steps used during generation (e.g. 20, 30, 4). For a
    # two-pass render this is the WHOLE schedule, both passes together.
    steps: int | None = None
    # Where a two-pass render cut that schedule; None when it rendered in one
    # pass. This is the only thing that tells a two-pass take from a
    # single-pass one of the same cost — `turbo_2pass` and `turbo_hd` both
    # finish at 1.0 MP after 8 steps — so the tier readback needs it.
    pass1_steps: int | None = None
    # Elapsed generation time in seconds (how long the render took to generate)
    generation_time: float | None = None
    # Where this take's AV latent was saved, when the render saved one. Lets an
    # Extend of this take ride the latent path (cleaner audio) instead of
    # re-encoding the sound from the video. Empty for takes rendered before the
    # latent was saved, or by a workflow with no audio.
    latent_path: str = ""
    # Upstream take id ("prompt_id:filename") when this take was generated as
    # an extend of an existing take via Motion Context.
    parent_take: str = ""
    # Frames of the parent clip that were pinned as motion context, on extends
    # only. 0 on ordinary renders, and on extends made before this was recorded
    # (those all used the default).
    context_length: int = 0
    refinement: list[str] = field(default_factory=list)
    refined_from: str = ""
    reference_face: str = ""
    face_refine_report: str = ""
    # Where prompt_text came from. Historical bad queue metadata can be
    # superseded in memory by the exact prompt embedded in the video container.
    prompt_source: str = "recorded"
    # Where `seconds` came from. "recorded" is the length the render was asked
    # for; "measured" is the length the file turned out to be, which is the only
    # one an edit can be cut against.
    duration_source: str = "recorded"
    # None means an older take did not record a stack; [] is a known bypass.
    loras: list[dict[str, Any]] | None = None

    @property
    def id(self) -> str:
        """Stable handle for a take.

        Built from the pair recording already dedupes on, so it needs no
        storage of its own and cannot disagree with the dedupe rule.
        """
        return f"{self.prompt_id}:{self.filename}"

    @property
    def is_video(self) -> bool:
        return self.filename.lower().endswith(VIDEO_EXTENSIONS)

    def view_url(self) -> str:
        """ComfyUI's own file endpoint. The gallery never serves bytes itself."""
        return "/view?" + urlencode(
            {"filename": self.filename, "subfolder": self.subfolder, "type": self.type}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt_id": self.prompt_id,
            "filename": self.filename,
            "shot": self.shot,
            "subfolder": self.subfolder,
            "type": self.type,
            "created": self.created,
            "frames": self.frames,
            "seconds": self.seconds,
            "seed": self.seed,
            "prompt_text": self.prompt_text,
            "width": self.width,
            "height": self.height,
            "rating": self.rating,
            "discarded": self.discarded,
            "seen": self.seen,
            "quality": self.quality,
            "steps": self.steps,
            "pass1_steps": self.pass1_steps,
            "generation_time": self.generation_time,
            "latent_path": self.latent_path,
            "parent_take": self.parent_take,
            "context_length": self.context_length,
            "refinement": self.refinement,
            "refined_from": self.refined_from,
            "reference_face": self.reference_face,
            "face_refine_report": self.face_refine_report,
            "prompt_source": self.prompt_source,
            "duration_source": self.duration_source,
            "loras": self.loras,
            "url": self.view_url(),
            "is_video": self.is_video,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Take":
        steps_val = data.get("steps")
        steps = int(steps_val) if steps_val is not None and not isinstance(steps_val, bool) else None
        split_val = data.get("pass1_steps")
        split = int(split_val) if split_val is not None and not isinstance(split_val, bool) else None
        gen_time_val = data.get("generation_time")
        gen_time = float(gen_time_val) if gen_time_val is not None else None

        return cls(
            prompt_id=str(data.get("prompt_id", "")),
            filename=str(data.get("filename", "")),
            # `clip` is what this field was called when a project was an
            # unordered bag and takes were keyed by name.
            shot=data.get("shot", data.get("clip", "")),
            subfolder=data.get("subfolder", ""),
            type=data.get("type", "output"),
            created=data.get("created") or utc_now_iso(),
            frames=int(data.get("frames") or 0),
            seconds=float(data.get("seconds") or 0.0),
            seed=data.get("seed"),
            prompt_text=data.get("prompt_text", ""),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            rating=int(data.get("rating") or 0),
            discarded=bool(data.get("discarded")),
            # A take described without the flag is one nobody has looked at:
            # that is what a renderer posting a finished job to `record` sends.
            # Reading *stored* history back is the opposite case, and `_read`
            # handles it there rather than here.
            seen=bool(data.get("seen", False)),
            quality=str(data.get("quality") or ""),
            steps=steps,
            pass1_steps=split,
            generation_time=gen_time,
            latent_path=str(data.get("latent_path") or ""),
            parent_take=str(data.get("parent_take") or ""),
            context_length=int(data.get("context_length") or 0),
            refinement=[str(p) for p in data.get("refinement") or [] if isinstance(p, (str, int))],
            refined_from=str(data.get("refined_from") or ""),
            reference_face=str(data.get("reference_face") or ""),
            face_refine_report=str(data.get("face_refine_report") or ""),
            prompt_source=str(data.get("prompt_source") or "recorded"),
            duration_source=str(data.get("duration_source") or "recorded"),
            loras=data.get("loras") if isinstance(data.get("loras"), list) else None,
        )


def embedded_workflow(path: Path) -> dict[str, Any]:
    """The API-format workflow SaveVideo wrote into the container, or `{}`.

    ComfyUI stores the graph that produced a clip as JSON in the MP4
    `mdta:prompt` tag. That makes every output self-describing: what it was
    prompted with, and — for a continuation — which clip it continued. Reading
    it needs no ComfyUI history, so a take stays legible long after the run that
    made it has been cleared.
    """
    if not path.is_file():
        return {}

    try:
        workflow = json.loads(_mp4_prompt_tag(path) or _ffprobe_prompt_tag(path))
    except (AttributeError, json.JSONDecodeError, TypeError):
        return {}
    return workflow if isinstance(workflow, dict) else {}


def embedded_workflow_source(path: Path) -> str:
    """The file a continuation was extended from, by name, or an empty string.

    An extend loads its parent through `LoadVideo`, whose `file` input reads
    `"video/NAME.mp4 [output]"`. That reference is the lineage, recorded by the
    graph that actually ran — which is why it can repair a take whose
    `parent_take` was never written down.
    """
    for node in embedded_workflow(path).values():
        if not isinstance(node, dict) or node.get("class_type") != "LoadVideo":
            continue
        reference = (node.get("inputs") or {}).get("file")
        if isinstance(reference, str) and reference.strip():
            name = reference.split("[")[0].strip()
            return name.split("/")[-1].split("\\")[-1]
    return ""


def embedded_workflow_prompt(path: Path) -> str:
    """Return the H3 prompt embedded by SaveVideo, or an empty string.

    The workflow can contain several prompt-like inputs, so prefer a MiniMax H3
    conditioning node and fall back to the first non-empty string prompt.
    """
    workflow = embedded_workflow(path)
    if not workflow:
        return ""

    fallback = ""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        prompt = (node.get("inputs") or {}).get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt = prompt.strip()
        if not fallback:
            fallback = prompt
        class_type = str(node.get("class_type") or "").casefold()
        if "minimaxh3" in class_type:
            return prompt
    return fallback


def measured_seconds(path: Path) -> float:
    """How long a clip really is, or 0.0 when it cannot be read.

    Worth the trouble because the recorded length is the one that was *asked
    for*. A Motion Context continuation pins 22 frames of its parent at the head
    and trims them back off before saving, so a 124-frame request delivers a
    102-frame file — 0.917s shorter than the gallery was told. Every trim
    default, clip width and running total downstream of an extend inherits that
    error until the file itself is consulted.
    """
    if not path.is_file():
        return 0.0
    return _mp4_duration(path) or _ffprobe_duration(path)


def _mp4_duration(path: Path) -> float:
    """Duration from the `moov/mvhd` header, without decoding or subprocesses."""
    try:
        with path.open("rb") as handle:
            end = path.stat().st_size
            moov = _child_atom(handle, 0, end, b"moov")
            if moov is None:
                return 0.0
            mvhd = _child_atom(handle, *moov, b"mvhd")
            if mvhd is None:
                return 0.0
            handle.seek(mvhd[0])
            version = handle.read(4)[:1]
            # Version 1 widens the creation/modification times and the duration.
            if version == b"\x01":
                raw = handle.read(28)
                if len(raw) != 28:
                    return 0.0
                timescale, duration = struct.unpack(">IQ", raw[16:28])
            else:
                raw = handle.read(16)
                if len(raw) != 16:
                    return 0.0
                timescale, duration = struct.unpack(">II", raw[8:16])
            if not timescale:
                return 0.0
            return round(duration / timescale, 3)
    except (OSError, ValueError, struct.error):
        return 0.0


def _ffprobe_duration(path: Path) -> float:
    """Fallback for containers whose header is not ISO BMFF."""
    executable = shutil.which("ffprobe")
    if not executable:
        return 0.0
    try:
        finished = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    try:
        return round(float(finished.stdout.strip()), 3)
    except (TypeError, ValueError):
        return 0.0


def _mp4_prompt_tag(path: Path) -> str:
    """Read the `mdta:prompt` value without decoding media or spawning ffprobe."""
    try:
        with path.open("rb") as handle:
            end = path.stat().st_size
            moov = _child_atom(handle, 0, end, b"moov")
            if moov is None:
                return ""
            udta = _child_atom(handle, *moov, b"udta")
            if udta is None:
                return ""
            meta = _child_atom(handle, *udta, b"meta")
            if meta is None or meta[0] + 4 > meta[1]:
                return ""
            # `meta` is a full box: version/flags precede its child atoms.
            children = (meta[0] + 4, meta[1])
            keys = _child_atom(handle, *children, b"keys")
            ilst = _child_atom(handle, *children, b"ilst")
            if keys is None or ilst is None:
                return ""
            prompt_index = _metadata_key_index(handle, *keys, "prompt")
            if prompt_index is None:
                return ""
            item = _child_atom(handle, *ilst, prompt_index.to_bytes(4, "big"))
            if item is None:
                return ""
            data = _child_atom(handle, *item, b"data")
            if data is None or data[0] + 8 > data[1]:
                return ""
            handle.seek(data[0] + 8)  # data type/flags + locale
            return handle.read(data[1] - data[0] - 8).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _atoms(handle: Any, start: int, end: int):
    """Yield (type, payload_start, atom_end) for ISO BMFF children."""
    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            return
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) != 8:
                return
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        atom_end = position + size
        if size < header_size or atom_end > end:
            return
        yield kind, position + header_size, atom_end
        position = atom_end


def _child_atom(handle: Any, start: int, end: int, wanted: bytes) -> tuple[int, int] | None:
    return next(
        ((payload, atom_end) for kind, payload, atom_end in _atoms(handle, start, end)
         if kind == wanted),
        None,
    )


def _metadata_key_index(handle: Any, start: int, end: int, wanted: str) -> int | None:
    if start + 8 > end:
        return None
    handle.seek(start + 4)  # full-box version/flags
    raw_count = handle.read(4)
    if len(raw_count) != 4:
        return None
    count = struct.unpack(">I", raw_count)[0]
    position = start + 8
    for index in range(1, count + 1):
        if position + 8 > end:
            return None
        handle.seek(position)
        header = handle.read(8)
        size, _namespace = struct.unpack(">I4s", header)
        if size < 8 or position + size > end:
            return None
        value = handle.read(size - 8).decode("utf-8", errors="replace")
        if value == wanted:
            return index
        position += size
    return None


def _ffprobe_prompt_tag(path: Path) -> str:
    """Fallback for containers whose metadata layout is not ISO BMFF mdta."""
    executable = shutil.which("ffprobe")
    if not executable:
        return ""
    try:
        executable = str(Path(executable).resolve(strict=True))
    except OSError:
        pass
    try:
        finished = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format_tags=prompt",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if finished.returncode != 0:
        return ""
    try:
        tags = (json.loads(finished.stdout).get("format") or {}).get("tags") or {}
        return next(
            (value for key, value in tags.items() if str(key).casefold() == "prompt"),
            "",
        )
    except (AttributeError, json.JSONDecodeError, TypeError):
        return ""


def _int_or_none(value: Any) -> int | None:
    """A whole number, or None for anything that is not one (booleans included)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def takes_from_history(prompt_id: str, entry: dict[str, Any]) -> list[Take]:
    """Read a ComfyUI history entry into takes.

    A history entry's `prompt` field is the queue tuple
    `(number, prompt_id, prompt, extra_data, outputs_to_execute)`, so our own
    metadata comes back out of index 3. Missing or oddly shaped entries yield
    nothing rather than raising — this runs over whatever the server happens to
    have, including runs from other tools.
    """
    meta = _extra_data(entry).get(EXTRA_DATA_KEY) or {}
    if not isinstance(meta, dict):
        meta = {}

    refined_from = str(meta.get("refined_from") or "")
    reference_face = str(meta.get("reference_face") or "")
    mode = str(meta.get("mode") or "").lower()
    is_refine = mode == "refine" or bool(meta.get("refinement")) or bool(refined_from)

    parent_take = str(meta.get("parent_take") or "")
    if is_refine and parent_take and (parent_take == refined_from or parent_take.endswith(f":{refined_from}") or parent_take == f"file:{refined_from}"):
        parent_take = ""

    if not parent_take and not is_refine:
        # Fallback for historical renders: inspect prompt nodes for LoadVideo source
        prompt_nodes = _prompt_nodes(entry)
        for node in prompt_nodes.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadVideo":
                f = (node.get("inputs") or {}).get("file")
                if isinstance(f, str) and f.strip():
                    clean_f = f.split("[")[0].strip()
                    filename_only = clean_f.split("/")[-1].split("\\")[-1]
                    if filename_only:
                        parent_take = f"file:{filename_only}"
                        break

    # Extract steps (from extra_data, or falling back to inspecting prompt nodes)
    steps = meta.get("steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (ValueError, TypeError):
            steps = None
    if steps is None:
        for node in _prompt_nodes(entry).values():
            if isinstance(node, dict):
                node_inputs = node.get("inputs") or {}
                s = node_inputs.get("steps")
                if isinstance(s, (int, float)) and not isinstance(s, bool):
                    steps = int(s)
                    break

    # Where a two-pass render cut its schedule. The queue metadata carries it
    # for anything rendered since the field existed; older two-pass takes are
    # still readable, because the transform leaves a `SplitSigmas` in the graph
    # and its `step` input IS the split. Nothing else in these workflows uses
    # that node, so finding one is proof the render was two-pass.
    pass1_steps = _int_or_none(meta.get("pass1_steps"))
    if pass1_steps is None:
        for node in _prompt_nodes(entry).values():
            if isinstance(node, dict) and node.get("class_type") == "SplitSigmas":
                pass1_steps = _int_or_none((node.get("inputs") or {}).get("step"))
                if pass1_steps is not None:
                    break

    # Extract generation time if recorded
    gen_time = meta.get("generation_time") or meta.get("gen_time") or entry.get("generation_time")
    if gen_time is not None:
        try:
            gen_time = float(gen_time)
        except (ValueError, TypeError):
            gen_time = None

    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []

    found: list[Take] = []
    face_report = "\n".join(
        report for output in outputs.values() if isinstance(output, dict)
        for report in output.get("face_refine_report", []) if isinstance(report, str)
    )
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in OUTPUT_KEYS:
            for item in node_output.get(key) or []:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                # LoadVideo also publishes a UI preview in history.outputs.
                # It is the parent's input, not a second take from this seed.
                if item.get("type") == "input":
                    continue
                found.append(
                    Take(
                        prompt_id=prompt_id,
                        filename=str(item["filename"]),
                        subfolder=str(item.get("subfolder") or ""),
                        type=str(item.get("type") or "output"),
                        shot=str(meta.get("shot") or meta.get("clip") or ""),
                        frames=int(meta.get("frames") or 0),
                        seconds=float(meta.get("seconds") or 0.0),
                        seed=meta.get("seed"),
                        prompt_text=str(meta.get("prompt") or ""),
                        width=int(meta.get("width") or 0),
                        height=int(meta.get("height") or 0),
                        quality=str(meta.get("quality") or ""),
                        steps=steps,
                        loras=meta.get("loras") if isinstance(meta.get("loras"), list) else None,
                        pass1_steps=pass1_steps,
                        generation_time=gen_time,
                        latent_path=str(meta.get("latent_path") or ""),
                        parent_take=parent_take,
                        context_length=int(meta.get("context_length") or 0),
                        refinement=[str(p) for p in meta.get("refinement") or [] if isinstance(p, (str, int))],
                        refined_from=refined_from,
                        reference_face=reference_face,
                        face_refine_report=face_report,
                    )
                )

    # A workflow that saves a still alongside the clip should not bury it.
    videos = [take for take in found if take.is_video]
    return videos or found


def project_of(entry: dict[str, Any]) -> str:
    """The project a history entry belongs to, or empty for someone else's run."""
    return str(queue_metadata(entry).get("project") or "")


def queue_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """This tool's own metadata on a history entry, or an empty dict.

    Public because stills read the same block for a different reason: what the
    gallery uses to tell whose render this was, the still library uses to tell
    whether it was a render at all.
    """
    meta = _extra_data(entry).get(EXTRA_DATA_KEY)
    return meta if isinstance(meta, dict) else {}


def _extra_data(entry: dict[str, Any]) -> dict[str, Any]:
    prompt_tuple = entry.get("prompt")
    if isinstance(prompt_tuple, (list, tuple)) and len(prompt_tuple) > 3:
        extra = prompt_tuple[3]
        if isinstance(extra, dict):
            return extra
    # Some callers hand us the extra_data directly.
    extra = entry.get("extra_data")
    return extra if isinstance(extra, dict) else {}


def _prompt_nodes(entry: dict[str, Any]) -> dict[str, Any]:
    prompt_tuple = entry.get("prompt")
    if isinstance(prompt_tuple, (list, tuple)) and len(prompt_tuple) > 2:
        nodes = prompt_tuple[2]
        if isinstance(nodes, dict):
            return nodes
    return {}


class Gallery:
    """Takes per project, in a file beside the project itself."""

    def __init__(self, projects: ProjectStore, output_dir: Path | None = None) -> None:
        self.projects = projects
        self.output_dir = Path(output_dir) if output_dir is not None else paths.comfy_output_dir()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._embedded_prompt_cache: dict[Path, tuple[int, str]] = {}
        self._duration_cache: dict[Path, tuple[int, float]] = {}
        self._source_cache: dict[Path, tuple[int, str]] = {}

    def path_for(self, project: str) -> Path:
        return self.projects.path_for(project) / TAKES_JSON

    def list(self, project: str, shot: str | None = None) -> list[Take]:
        """Takes for a project, newest first, with the record repaired in memory.

        Three things are recovered from the files themselves, in an order that
        matters: lineage first, because it writes `file:` references that
        `_resolve_parent_takes` then turns into take ids; then the true
        durations; then the prompts. Every repair is in-memory, so `takes.json`
        stays exactly as it was written and the inspector can show where a
        corrected value came from.
        """
        # Hide input previews recorded by older builds without deleting files
        # or rewriting the original take records during a read.
        takes = [take for take in self._read(project) if take.type != "input"]
        self._recover_lineage(takes)
        self._resolve_parent_takes(takes)
        self._measure_durations(takes)
        self._recover_embedded_prompts(takes)
        if shot:
            takes = [take for take in takes if take.shot == shot]
        return sorted(takes, key=lambda take: take.created, reverse=True)

    def _output_path(self, take: Take) -> Path | None:
        """A take's real file, or None when it is outside the output directory.

        The containment check is what keeps a crafted `subfolder` in a takes
        file from turning a metadata read into an arbitrary file read.
        """
        if self.output_dir is None or not take.is_video:
            return None
        root = self.output_dir.resolve()
        candidate = (root / take.subfolder / take.filename).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _cached(
        self, cache: dict, path: Path, read: Any
    ) -> Any:
        """Memoise a per-file read, invalidated when the file changes."""
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            return None
        hit = cache.get(path)
        if hit is None or hit[0] != modified:
            hit = (modified, read(path))
            cache[path] = hit
        return hit[1]

    def _recover_lineage(self, takes: list[Take]) -> None:
        """Fill in a missing `parent_take` from the clip's own workflow.

        Queue metadata only carried `parent_take` for part of this tool's life,
        and the ComfyUI history that could once fill the gap is cleared long
        before a project is finished. The MP4 keeps the answer permanently: an
        extend loads its parent through `LoadVideo`. Without this, whole chains
        read as unrelated takes.
        """
        for take in takes:
            if take.parent_take or take.refinement or take.refined_from or take.quality == "refined":
                continue
            path = self._output_path(take)
            if path is None:
                continue
            source = self._cached(self._source_cache, path, embedded_workflow_source)
            if source and source != take.filename:
                take.parent_take = f"file:{source}"

    def _measure_durations(self, takes: list[Take]) -> None:
        """Replace the length that was asked for with the length delivered.

        See `measured_seconds`: a continuation is 22 frames shorter than its
        request, so trusting the recorded value overstates every extend and
        drags the running total of everything after it out of true.
        """
        for take in takes:
            path = self._output_path(take)
            if path is None:
                continue
            measured = self._cached(self._duration_cache, path, measured_seconds)
            if not measured or measured <= 0:
                continue
            take.duration_source = "measured"
            if abs(measured - float(take.seconds or 0.0)) < 0.01:
                continue
            take.seconds = measured
            # H3 runs at a fixed 24fps, so the frame count follows the duration
            # rather than needing a second, more expensive read.
            take.frames = int(round(measured * FPS))

    def _recover_embedded_prompts(self, takes: list[Take]) -> None:
        """Replace bad historical labels with the exact prompt in the MP4.

        Recovery is intentionally in-memory: the original gallery record stays
        untouched and the inspector can say where the corrected value came from.

        Takes whose recorded text already looks like a real prompt (structured
        markers or >= 256 chars) are left alone.  Extends are *not* exempt:
        natively-rendered or ingested extends can carry a short label even
        though the full prompt is embedded in the video metadata.
        """
        for take in takes:
            text = take.prompt_text.strip()
            looks_structured = any(marker in text for marker in H3_PROMPT_MARKERS)
            if looks_structured or len(text) >= 256:
                continue

            candidate = self._output_path(take)
            if candidate is None:
                continue

            recovered = (
                self._cached(self._embedded_prompt_cache, candidate, embedded_workflow_prompt)
                or ""
            ).strip()
            if recovered and recovered != text:
                take.prompt_text = recovered
                take.prompt_source = "video_metadata"

    def _resolve_parent_takes(self, takes: list[Take]) -> list[Take]:
        """Turn a parent reference that names a *file* into the take id it means.

        Two shapes arrive here. `file:NAME` is what `_recover_lineage` writes
        when it reads the parent out of the MP4's own workflow. `:NAME` — a take
        id with an empty prompt_id — is what a render wrote when it knew which
        file it was continuing but not the job that had produced it; without
        this, those continuations read as unrelated roots and their chain is
        invisible.

        Both resolve by filename, and only when that filename belongs to exactly
        one take. Output names are not guaranteed unique across a project, and a
        parent picked out of two candidates is a fabricated chain — which is
        worse than a link that stays unresolved and says so.
        """
        seen: dict[str, int] = {}
        for take in takes:
            seen[take.filename] = seen.get(take.filename, 0) + 1
        by_filename = {t.filename: t.id for t in takes if seen[t.filename] == 1}

        for take in takes:
            reference = take.parent_take
            if reference.startswith("file:"):
                wanted = reference[5:]
            elif reference.startswith(":"):
                wanted = reference[1:]
            else:
                continue
            landing = by_filename.get(wanted)
            if landing:
                take.parent_take = landing
        return takes

    def find(self, project: str, take_id: str) -> Take | None:
        return next((take for take in self._read(project) if take.id == take_id), None)

    def annotate(
        self,
        project: str,
        take_id: str,
        *,
        rating: int | None = None,
        discarded: bool | None = None,
        seen: bool | None = None,
    ) -> Take | None:
        """Set a take's rating, discard flag or seen flag. Triage, not deletion."""
        with self._lock_for(project):
            takes = self._read(project)
            found = next((take for take in takes if take.id == take_id), None)
            if found is None:
                return None
            if rating is not None:
                found.rating = max(0, min(5, int(rating)))
            if discarded is not None:
                found.discarded = bool(discarded)
            if seen is not None:
                found.seen = bool(seen)
            self._write(project, takes)
            return found

    def mark_seen(self, project: str, take_ids: Iterable[str], seen: bool = True) -> list[str]:
        """Flip the seen flag on a batch, in one write.

        Clearing a source's new badge means touching every take under it at
        once, and one annotate call each would be one file write each — with
        the project lock taken and released between them, so a render landing
        mid-sweep could be marked seen before anyone had seen it.
        """
        wanted = set(take_ids)
        if not wanted:
            return []
        with self._lock_for(project):
            takes = self._read(project)
            changed = [
                take.id for take in takes if take.id in wanted and take.seen != seen
            ]
            for take in takes:
                if take.id in wanted:
                    take.seen = seen
            if changed:
                self._write(project, takes)
            return changed

    def reassign(self, project: str, moves: dict[str, str]) -> int:
        """Re-stamp `shot` where one source has absorbed another.

        Folding a chain leaves the absorbed sources' takes naming a bin row that
        no longer exists, and `shot` is what the take grid and the take count
        filter on — so they would vanish from the source that now owns them.
        Only that one field moves: `parent_take` is the renderer's record of
        what continues what and is never rewritten here.
        """
        if not moves:
            return 0
        with self._lock_for(project):
            takes = self._read(project)
            moved = 0
            for take in takes:
                landing = moves.get(take.shot)
                if landing and landing != take.shot:
                    take.shot = landing
                    moved += 1
            if moved:
                self._write(project, takes)
            return moved

    def record(self, project: str, takes: Iterable[Take]) -> list[Take]:
        """Add takes, ignoring any whose prompt_id and filename are already in.

        Idempotent on purpose: reconciliation replays history on every page
        load, and a take must not appear twice because a tab was reopened.
        Backfills newly discovered parent_take references onto existing takes.
        """
        # Directly posted take records obey the same rule as parsed history.
        incoming = [take for take in takes if take.type != "input"]
        if not incoming:
            return []

        with self._lock_for(project):
            existing = self._read(project)
            existing_by_key = {(take.prompt_id, take.filename): take for take in existing}
            added = []
            updated = False

            for take in incoming:
                key = (take.prompt_id, take.filename)
                if key not in existing_by_key:
                    existing.append(take)
                    existing_by_key[key] = take
                    added.append(take)
                else:
                    ex = existing_by_key[key]
                    if not ex.parent_take and take.parent_take:
                        ex.parent_take = take.parent_take
                        updated = True

            if added or updated:
                self._write(project, existing)
            return added

    def forget(self, project: str, prompt_id: str) -> int:
        """Drop a take from the record. The file on disk is left alone."""
        with self._lock_for(project):
            existing = self._read(project)
            kept = [take for take in existing if take.prompt_id != prompt_id]
            if len(kept) != len(existing):
                self._write(project, kept)
            return len(existing) - len(kept)

    def _lock_for(self, project: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(project)
            if lock is None:
                lock = threading.Lock()
                self._locks[project] = lock
            return lock

    def _read(self, project: str) -> list[Take]:
        path = self.path_for(project)
        if not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError:
            return []
        except json.JSONDecodeError:
            # A corrupt takes file must not make the project unopenable, nor be
            # silently overwritten by the next record() call. Move it aside first.
            stamp = utc_now_iso().replace(":", "-")
            aside = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
            try:
                path.replace(aside)
            except OSError:
                pass
            return []
        if not isinstance(data, dict):
            return []
        takes = []
        for entry in data.get("takes") or []:
            take = Take.from_dict(entry)
            # Stored before the seen flag existed. Reading those back as new
            # would badge a whole project's history the first time it opened,
            # which is a badge that means nothing.
            if isinstance(entry, dict) and "seen" not in entry:
                take.seen = True
            takes.append(take)
        return takes

    def _write(self, project: str, takes: list[Take]) -> None:
        path = self.path_for(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {"takes": [take.to_dict() for take in takes]})
