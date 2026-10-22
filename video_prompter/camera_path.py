"""Per-beat camera trajectories compiled locally into H3 camera direction.

Coordinates are relative to the beat's opening camera. They guide the model
through prose, not geometric conditioning or reconstructed world coordinates.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math


def validate_path(path) -> str:
    if path is None:
        return ""
    if not isinstance(path, dict):
        return "Camera path must be an object."
    if path.get("interpolation", "smooth") not in ("smooth", "linear"):
        return "Choose smooth or linear camera interpolation."
    if not isinstance(path.get("target", ""), str):
        return "Camera target must be text."
    if not isinstance(path.get("freeze", False), bool):
        return "Freeze scene must be true or false."
    if path.get("frame_mode", "scene") not in ("scene", "motion"):
        return "Choose scene or motion camera mode."
    if path.get("loop_closure", "off") not in ("auto", "off"):
        return "Choose auto or off for loop closure."
    if path.get("experiment", "off") not in ("off", "compact", "numbers"):
        return "Choose a supported camera prompt recipe."
    if path.get("frame_mode") == "motion":
        if not isinstance(path.get("reference_video"), str) or not path["reference_video"].strip():
            return "Motion Frame needs a reference video."
        if path.get("freeze"):
            return "Motion Frame preserves action; turn off Freeze scene."
    box = path.get("subject_box")
    if box is not None:
        if not isinstance(box, dict) or any(isinstance(box.get(k), bool) or not isinstance(box.get(k), (int, float))
            or not math.isfinite(box[k]) or not 0 <= box[k] <= 1 for k in ("left", "top", "width", "height")):
            return "Subject box needs finite normalized left, top, width and height."
        if box["width"] <= 0 or box["height"] <= 0 or box["left"] + box["width"] > 1.000001 or box["top"] + box["height"] > 1.000001:
            return "Subject box must fit inside the reference image."
    points = path.get("keyframes")
    if not isinstance(points, list) or not 2 <= len(points) <= 12:
        return "Camera paths need 2–12 keyframes."
    previous = -1.0
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            return f"Camera keyframe {index + 1} must be an object."
        for name, low, high in (("time", 0, 1), ("azimuth", -720, 720),
                                ("elevation", -89, 89), ("distance", .05, 4)):
            value = point.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return f"Keyframe {index + 1}: {name} must be a finite number."
            if not low <= value <= high:
                return f"Keyframe {index + 1}: {name} must be between {low} and {high}."
        if point["time"] <= previous:
            return "Camera keyframe times must increase without duplicates."
        previous = point["time"]
    if any(points[0][key] != value for key, value in dict(time=0, azimuth=0, elevation=0, distance=1).items()):
        return "The first camera keyframe must stay at time 0, orbit 0°, height 0°, distance 1×."
    if points[-1]["time"] != 1:
        return "The final camera keyframe must be at the end of the beat."
    return ""


def motion_path(prompt):
    return next((b.camera_path for b in prompt.storyboard.beats
                 if isinstance(b.camera_path, dict) and b.camera_path.get("frame_mode") == "motion"
                 and not validate_path(b.camera_path)), None)


def loop_status(prompt):
    """Eligibility for a real native last-frame anchor, never a path guarantee."""
    beats = prompt.storyboard.beats
    requested = any(isinstance(b.camera_path, dict) and b.camera_path.get("loop_closure") == "auto" for b in beats)
    if not requested:
        return False, "Loop closure off."
    if len(beats) != 1 or beats[0].start != 0 or beats[0].end != prompt.duration:
        return False, "Loop closure needs one camera path covering the whole generation."
    path = beats[0].camera_path
    if validate_path(path):
        return False, "Fix the camera path before enabling loop closure."
    if path.get("frame_mode") == "motion" or not path.get("freeze"):
        return False, "Loop closure needs a frozen scene."
    if not prompt.first_frame.strip() or prompt.references.items:
        return False, "Loop closure needs a first-frame image and no omni-references."
    if prompt.last_frame.strip() and prompt.last_frame != prompt.first_frame:
        return False, "Your explicit last frame takes priority over loop closure."
    a, b = path["keyframes"][0], path["keyframes"][-1]
    if not (math.isclose(abs(b["azimuth"]-a["azimuth"]), 360, abs_tol=1e-6, rel_tol=0)
            and all(math.isclose(a[k], b[k], abs_tol=1e-6, rel_tol=0) for k in ("elevation", "distance"))):
        return False, "Loop closure needs exactly ±360° and the opening elevation and distance."
    return True, "Final frame anchored to the first image; the intervening orbit remains prompt-guided."


def camera_media(prompt):
    """Resolve camera-pass media without changing the saved authoring document."""
    path = motion_path(prompt)
    closure, _ = loop_status(prompt)
    if not path and not closure:
        return prompt
    made = deepcopy(prompt)
    if path:
        from .document import Reference
        made.first_frame = made.last_frame = ""
        if not any(r.kind == "video" and r.path == path["reference_video"] for r in made.references.items):
            made.references.items.append(Reference(kind="video", path=path["reference_video"],
                role="the source action sequence; preserve its subjects, scene and action progression while changing the camera viewpoint"))
    elif closure:
        made.last_frame = made.first_frame
    return made


def camera_motion(beat, *, end=None, reference_tag="the reference video") -> str:
    path = beat.camera_path
    if path is None or validate_path(path):
        return beat.camera
    from .storyboard import format_timecode
    end = beat.end if end is None else end
    target = path.get("target", "").strip().rstrip(".;") or "the main subject"
    parts = [f"follows one continuous physical path around {target}, keeping the lens aimed at the same target, "
             "with fixed focal length and zero roll; all offsets are relative to this beat's opening camera"]
    points = path["keyframes"]
    if all(p["azimuth"] == 0 and p["elevation"] == 0 for p in points):
        distances = [p["distance"] for p in points]
        inward = all(b <= a for a, b in zip(distances, distances[1:])) and distances[-1] < distances[0]
        outward = all(b >= a for a, b in zip(distances, distances[1:])) and distances[-1] > distances[0]
        if inward or outward:
            direction = f"straight toward {target}" if inward else f"straight away from {target}"
            change, size = ("decreases", "larger") if inward else ("increases", "smaller")
            parts[0] = (f"dollies {direction} along the opening viewing axis, keeping the same viewing angle, "
                        f"fixed focal length and zero roll; starts at the existing framing and moves only "
                        f"{'closer' if inward else 'farther away'} during each moving interval; "
                        f"the target distance {change} continuously from {distances[0]:g}× to {distances[-1]:g}× "
                        f"the opening distance, making the target progressively {size} in the frame; explicit camera holds stay still")
        elif all(d == distances[0] for d in distances):
            parts[0] = f"holds the opening viewpoint and framing on {target}, with fixed focal length and zero roll"
        else:
            parts[0] = f"dollies along the opening viewing axis toward and away from {target} at the specified times, keeping the same viewing angle and fixed focal length"
    if path.get("frame_mode") == "motion":
        parts.append(f"begins at {reference_tag}'s opening camera viewpoint and subject pose, then follows the requested path through the same scene")
    if path.get("subject_box"):
        box = path["subject_box"]
        parts.append("identifies the target in the reference's opening frame at "
                     f"[L={box['left']:.4f},T={box['top']:.4f},W={box['width']:.4f},H={box['height']:.4f}]; "
                     "this box identifies the subject, not a crop or a fixed screen position")
    for a, b in zip(path["keyframes"], path["keyframes"][1:]):
        start_time = beat.start + (end - beat.start) * a["time"]
        end_time = beat.start + (end - beat.start) * b["time"]
        moves = []
        delta = b["azimuth"] - a["azimuth"]
        if delta:
            direction = "right" if delta > 0 else "left"
            moves.append(f"physically orbits {abs(delta):g} degrees toward the camera's {direction} "
                         f"from orbit offset {a['azimuth']:g}° to {b['azimuth']:g}°, revealing perspective and occlusion changes")
        if b["elevation"] != a["elevation"]:
            verb = "rises" if b["elevation"] > a["elevation"] else "lowers"
            moves.append(f"{verb} along a vertical arc from elevation offset {a['elevation']:g}° to {b['elevation']:g}°")
        if b["distance"] != a["distance"]:
            verb = "dollies inward" if b["distance"] < a["distance"] else "dollies outward"
            moves.append(f"{verb} from {a['distance']:g}× to {b['distance']:g}× the opening distance")
        if not moves:
            moves.append("holds the same camera pose")
        parts.append(f"from {format_timecode(start_time)} to {format_timecode(end_time)}, the camera " + ", and simultaneously ".join(moves))
    if path.get("interpolation", "smooth") == "smooth":
        parts.append("interpolates each axis with a monotone cubic curve without overshoot, starting and finishing at zero speed; "
                     "keeps velocity continuous through intermediate keyframes, slowing an axis to zero at its holds and reversals")
    else:
        parts.append("travels at a constant rate within each keyframe interval")
    if path.get("freeze", False):
        parts.append("only the camera moves while the subject's pose, scene objects, lighting and captured instant remain fixed")
    if path.get("frame_mode") == "motion":
        parts.append(f"preserves {reference_tag}'s action order and timing, subject identities and scene structure; "
                     "tracks the same moving subject; camera holds keep the action running")
    experiment = path.get("experiment", "off")
    if experiment != "off":
        if experiment == "numbers":
            parts = [parts[0], *[p for p in parts[1:] if not p.startswith("from ")]]
        parts.append("Camera keyframes: " + json.dumps(dict(time_unit="fraction of this beat", azimuth_unit="degrees camera-right",
                     elevation_unit="degrees relative to opening view", distance_unit="opening radius multiplier",
                     interpolation=path.get("interpolation", "smooth"), keyframes=path["keyframes"]), separators=(",", ":")))
    return "; ".join(parts)


def materialize_paths(prompt):
    """A temporary prose copy; saved geometry and handwritten cameras stay intact."""
    if not any(beat.camera_path is not None for beat in prompt.storyboard.beats):
        return prompt
    made = deepcopy(camera_media(prompt))
    for beat in made.storyboard.beats:
        end = (made.frames - 1) / 24 if beat.end == made.duration else beat.end
        path = beat.camera_path
        ref = next((r for r in made.references.items if isinstance(path, dict)
                    and r.kind == "video" and r.path == path.get("reference_video")), None)
        beat.camera = camera_motion(beat, end=end, reference_tag=made.references.tag(ref) if ref else "the reference video")
        beat.camera_path = None
    return made
