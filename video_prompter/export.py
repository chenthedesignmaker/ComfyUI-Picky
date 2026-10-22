"""Turning the edit into a file.

Without this the timeline is a mood board with a ruler. Three outputs, in
descending order of how much they assume:

- **ffmpeg** — a concat filter graph that trims and re-times each piece and
  writes one file. Assumes ffmpeg is on PATH; the plan is emitted either way,
  so a missing binary costs you the render, not the work.
- **EDL** — CMX3600, which every NLE reads. Source timecodes are the take's
  own, so a round trip into Resolve lands on the same frames.
- **clip list** — plain text, for when the answer is "which takes went in".

A clip is a range of *lineage* time, so one clip can span several takes: a trim
that runs across the join between generation two and three resolves to two
pieces, back to back. Everything below therefore works in pieces rather than in
clips, and the join is `lineage.resolve`'s to make — the same function the
monitor plays through, so the file cannot disagree with the preview.

**Sound is resolved separately from picture** and mixed rather than
concatenated. A clip may be muted, or may borrow its audio from a take whose
picture was rejected, so the audio pieces do not line up one-to-one with the
video ones and cannot ride the same concat. Each piece is trimmed, re-timed,
delayed to its place on the timeline and mixed; that same graph is what a music
lane or an imported asset would drop into, which is why it is built this way for
a case that never overlaps.

Nothing here copies video. Takes already point at ComfyUI's output directory
and that is where the bytes stay.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import edit, h3_spec
from .gallery import Take
from .projects import Project
from .sequence import DEFAULT_FADE, AudioClip, Clip, Imported, Lifted
from .sources import Source

FORMATS = ("ffmpeg", "edl", "shotlist")

# Re-encoding is unavoidable: the pieces are trimmed at arbitrary points and
# some are re-timed, so there is no stream copy that would be correct.
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"

# One rate and layout for the whole mix. The takes need not agree — an imported
# asset certainly will not — and `amix` refuses inputs that differ.
MIX_RATE = 48000
MIX_LAYOUT = "stereo"
AUDIO_BITRATE = "192k"

# `atempo` only accepts 0.5–2.0, so anything outside is a chain of them.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0


class ExportError(ValueError):
    pass


@dataclass
class ResolvedClip:
    """One contiguous run of one take's file, placed in the edit.

    Named for what it is rather than just `Clip`, which means a timeline
    instance pointing into a source. This one is the export's own join of a
    clip, the take covering part of it, and the file on disk — so a clip that
    crosses a generation join produces two of these, back to back.
    """

    clip: Clip
    source: Source
    take: Take
    path: Path
    start: float
    length: float
    # Seconds into this take's own file, not into lineage time.
    source_in: float
    source_out: float
    speed: float

    @property
    def name(self) -> str:
        """What to call this in an EDL or a listing."""
        return self.clip.label or self.source.name


@dataclass
class ResolvedSound:
    """One contiguous run of one file's *audio*, placed in the edit.

    Deliberately not a `ResolvedClip` with a flag. A clip that borrows its sound
    produces video pieces and audio pieces that cut in different places, and one
    type carrying both would have to pretend they align.
    """

    # None for a lane item. Sound that belongs to a picture names it; sound on
    # the lane belongs to the edit rather than to any one shot, and inventing a
    # clip for it would put it back in the relationship it was lifted out of.
    clip: Clip | None
    name: str
    path: Path
    start: float
    length: float
    source_in: float
    source_out: float
    speed: float
    gain_db: float
    # True when the sound comes from somewhere other than the picture — worth
    # saying in a listing, because it is the thing a reader would not guess.
    borrowed: bool = False
    # Ramps at the ends, in seconds. Zero is the honest default for a piece that
    # ends where its picture ends: sound is supposed to cut with the picture
    # there, and a ramp would soften a cut the edit meant to be hard. They are
    # set where an edge was *made* rather than found — either side of a mute, and
    # both ends of a lane item — because a square edge mid-waveform clicks.
    fade_in: float = 0.0
    fade_out: float = 0.0
    # A lane item plays no picture, so nothing else in a listing would say where
    # it came from.
    lane: bool = False


def _placement(project: Project, takes: list[Take]) -> tuple[edit.Bin, dict[str, float], dict[str, float]]:
    """The bin, each clip's available lineage, and where each clip starts.

    One helper so picture and sound are placed by the same arithmetic; two
    copies of this is exactly how a mix ends up a frame out from the cut.
    """
    bin = edit.read(project.sources, takes)
    lengths = edit.lengths(project.sources, project.clips, bin)
    return bin, lengths, project.timeline.starts_at(lengths)


def resolve(project: Project, takes: list[Take], output_dir: Path) -> list[ResolvedClip]:
    """Every enabled clip resolved into the files that cover it, in edit order.

    Raises rather than skipping: an export that quietly drops the clips you
    forgot to choose a take for produces a video that looks finished and is not.
    """
    bin, lengths, starts = _placement(project, takes)

    unresolved: list[str] = []
    resolved: list[ResolvedClip] = []

    for clip in project.timeline.active():
        source = project.source_of(clip)
        if source is None:
            unresolved.append(clip.label or clip.source_id)
            continue

        pieces = edit.resolve_clip(clip, source, bin)
        if not pieces:
            unresolved.append(clip.label or source.name)
            continue

        speed = clip.speed if clip.speed > 0 else 1.0
        cursor = starts[clip.id]
        for piece in pieces:
            take = bin.index[piece.take_id].take
            length = round(piece.seconds / speed, 3)
            resolved.append(
                ResolvedClip(
                    clip=clip,
                    source=source,
                    take=take,
                    path=Path(output_dir) / take.subfolder / take.filename,
                    start=round(cursor, 3),
                    length=length,
                    source_in=piece.source_from,
                    source_out=piece.source_to,
                    speed=speed,
                )
            )
            cursor += length

    if unresolved:
        raise ExportError(
            "Nothing plays for: " + ", ".join(unresolved) + ". "
            "Choose a take for every enabled clip, or disable the ones not ready."
        )
    if not resolved:
        raise ExportError("Nothing to export — no enabled clips.")

    return resolved


def resolve_sound(
    project: Project, takes: list[Take], output_dir: Path,
    audio_dir: Path | None = None,
) -> list[ResolvedSound]:
    """Everything the export will hear: the clips' own sound, and the lane.

    Never raises. Picture is what the edit is made of, so :func:`resolve` refuses
    an unresolvable one; a clip whose *sound* cannot be found is a silence, and a
    silence is a legible result — you can hear it. Refusing the whole export
    because a borrowed lineage has not rendered yet would block work that is
    otherwise finished.

    Mutes are subtracted per clip, before the lane is added. Order matters and
    only in that direction: a mute is a window over one clip, so it must not
    reach the sound somebody deliberately moved on top of that clip. Cutting a
    hole and then filling it is the whole shape of a J-cut, and doing these two
    steps the other way round would silence the fill along with the hole.
    """
    bin, lengths, starts = _placement(project, takes)

    found: list[ResolvedSound] = []
    for clip in project.timeline.active():
        source = project.source_of(clip)
        if source is None or clip.audio.muted:
            continue

        holder = project.source(clip.audio.source_id) if clip.audio.source_id else source
        pieces = edit.resolve_audio(clip, source, bin, holder)
        if not pieces:
            continue

        speed = clip.speed if clip.speed > 0 else 1.0
        cursor = starts[clip.id]
        placed: list[ResolvedSound] = []
        for piece in pieces:
            take = bin.index[piece.take_id].take
            length = round(piece.seconds / speed, 3)
            placed.append(
                ResolvedSound(
                    clip=clip,
                    name=clip.label or source.name,
                    path=Path(output_dir) / take.subfolder / take.filename,
                    start=round(cursor, 3),
                    length=length,
                    source_in=piece.source_from,
                    source_out=piece.source_to,
                    speed=speed,
                    gain_db=clip.audio.gain_db,
                    borrowed=clip.audio.borrowed,
                )
            )
            cursor += length

        found += silenced(
            placed,
            edit.mute_windows(
                clip, project.timeline.mutes, starts[clip.id], lengths.get(clip.id),
            ),
        )

    return found + resolve_lane(project, takes, output_dir, audio_dir)


def _resliced(
    sound: ResolvedSound, start: float, end: float, *, fade_in: float, fade_out: float
) -> ResolvedSound:
    """The part of one piece between two timeline positions.

    Both clocks move together: a piece's file positions advance `speed` seconds
    for every timeline second, which is the same arithmetic the placement used on
    the way in, run backwards.
    """
    into = (start - sound.start) * sound.speed
    out_of = (end - sound.start) * sound.speed
    return replace(
        sound,
        start=round(start, 3),
        length=round(end - start, 3),
        source_in=round(sound.source_in + into, 3),
        source_out=round(sound.source_in + out_of, 3),
        fade_in=fade_in,
        fade_out=fade_out,
    )


def _minus_window(sound: ResolvedSound, low: float, high: float) -> list[ResolvedSound]:
    """One piece with one timeline window taken out of it: 0, 1 or 2 pieces.

    A window covering the middle leaves a head and a tail, and the two edges it
    just created get a fade. The outer edges keep whatever they had, so a mute in
    the middle of a shot does not quietly soften the cut at either end of it.
    """
    end = round(sound.start + sound.length, 3)
    if high <= sound.start + 1e-6 or low >= end - 1e-6:
        return [sound]

    parts: list[ResolvedSound] = []
    if low > sound.start + 1e-6:
        parts.append(_resliced(
            sound, sound.start, low,
            fade_in=sound.fade_in, fade_out=_edge_fade(low - sound.start),
        ))
    if high < end - 1e-6:
        parts.append(_resliced(
            sound, high, end,
            fade_in=_edge_fade(end - high), fade_out=sound.fade_out,
        ))
    return parts


def _edge_fade(length: float) -> float:
    """A fade that fits. A sliver shorter than the ramp fades across itself."""
    return round(min(DEFAULT_FADE, max(0.0, length / 2)), 3)


def silenced(
    sounds: list[ResolvedSound], windows: list[tuple[float, float]]
) -> list[ResolvedSound]:
    """Sound with the mute windows cut out of it.

    Applied piece by piece and window by window, so a mute lying across a
    generation join shortens two pieces rather than confusing one. Windows come
    in merged (see :func:`edit.mute_windows`), which is what makes a single pass
    per window enough.
    """
    if not windows:
        return sounds
    kept = sounds
    for low, high in windows:
        cut: list[ResolvedSound] = []
        for sound in kept:
            cut += _minus_window(sound, low, high)
        kept = cut
    return kept


def resolve_lane(
    project: Project,
    takes: list[Take],
    output_dir: Path,
    audio_dir: Path | None = None,
) -> list[ResolvedSound]:
    """The audio lane as placed pieces, ready to go into the same mix.

    Lane items come out as `ResolvedSound` like everything else, which is the
    whole reason the mix needs no new machinery: `ffmpeg_command` already places
    each piece by its own start rather than joining them, so sound that sits
    nowhere near its picture — or on top of three other pieces — is just more of
    what it was already doing.

    An imported item needs `audio_dir` to become a real path. Without one it is
    skipped rather than guessed at, on the same principle as everything else
    here: a missing sound is a silence, not a failed export.
    """
    bin, lengths, _ = _placement(project, takes)
    duration = project.timeline.duration(lengths)
    found: list[ResolvedSound] = []

    for item in project.timeline.active_audio():
        pieces: list[ResolvedSound] = []

        if isinstance(item.origin, Imported):
            if audio_dir is None or not item.origin.file:
                continue
            length = item.timeline_seconds()
            if length <= 0:
                continue
            pieces.append(ResolvedSound(
                clip=None,
                name=item.label or item.origin.file,
                path=Path(audio_dir) / item.origin.file,
                start=round(item.start, 3),
                length=length,
                source_in=round(max(0.0, item.origin.file_in), 3),
                source_out=round(item.origin.out_point(), 3),
                speed=1.0,
                gain_db=item.gain_db,
                borrowed=True,
                lane=True,
            ))
        else:
            source = project.source(item.origin.source_id)
            if source is None:
                continue
            cursor = item.start
            for piece in edit.resolve_lifted(item, source, bin):
                take = bin.index[piece.take_id].take
                pieces.append(ResolvedSound(
                    clip=None,
                    name=item.label or source.name,
                    path=Path(output_dir) / take.subfolder / take.filename,
                    start=round(cursor, 3),
                    length=piece.seconds,
                    source_in=piece.source_from,
                    source_out=piece.source_to,
                    speed=1.0,
                    gain_db=item.gain_db,
                    borrowed=True,
                    lane=True,
                ))
                cursor += piece.seconds

        if not pieces:
            continue

        # The item's own ramps go on its outer ends only — the joins between its
        # pieces are a continuous lineage and must stay seamless.
        pieces[0] = replace(pieces[0], fade_in=item.fade_in)
        pieces[-1] = replace(pieces[-1], fade_out=item.fade_out)

        # The export is as long as the picture. Sound dragged past the end is
        # trimmed rather than allowed to stretch the file, so what plays after
        # the last frame is nothing at all.
        found += silenced(pieces, [(duration, duration + 1e9)])

    return found


def ffmpeg_command(
    clips: list[ResolvedClip],
    destination: Path,
    sounds: list[ResolvedSound] | None = None,
) -> list[str]:
    """Picture concatenated, sound mixed, into one file.

    Every piece is scaled and padded to the first one's canvas. Mixed sizes
    would otherwise fail the concat filter outright, and letterboxing a stray
    piece is friendlier than refusing the whole export.

    Sound is placed rather than joined: each piece is delayed to its own start
    on the timeline and mixed. That cost nothing when the pieces all sat end to
    end, and it is the only shape that still works now that the audio lane can
    put one on top of another. `normalize=0` on the mix matters — the default
    divides by the input count, which would quietly drop a ten-clip edit by 20 dB.

    Pure on purpose: no probing, no disk. A file with no audio stream would fail
    the whole render, so callers filter those out with :func:`silent_sources`
    first.
    """
    width, height = _canvas(clips)
    sounds = sounds or []

    args: list[str] = ["ffmpeg", "-y"]
    for clip in clips:
        args += ["-i", str(clip.path)]
    for sound in sounds:
        args += ["-i", str(sound.path)]

    steps: list[str] = []
    labels: list[str] = []
    for index, clip in enumerate(clips):
        label = f"v{index}"
        chain = (
            f"[{index}:v]"
            f"trim=start={clip.source_in}:end={clip.source_out},"
            f"setpts=PTS-STARTPTS"
        )
        if clip.speed != 1.0:
            chain += f",setpts=PTS/{clip.speed}"
        chain += (
            f",scale={width}:{height}:force_original_aspect_ratio=decrease"
            f",pad={width}:{height}:-1:-1:color=black"
            f",setsar=1[{label}]"
        )
        steps.append(chain)
        labels.append(f"[{label}]")

    steps.append(f"{''.join(labels)}concat=n={len(clips)}:v=1:a=0[out]")

    mixed: list[str] = []
    for offset, sound in enumerate(sounds):
        label = f"a{offset}"
        index = len(clips) + offset
        chain = [
            f"[{index}:a]atrim=start={sound.source_in}:end={sound.source_out}",
            "asetpts=PTS-STARTPTS",
        ]
        chain += _atempo_chain(sound.speed)
        if sound.gain_db:
            chain.append(f"volume={sound.gain_db}dB")
        # Before the delay, so the ramps are timed against the piece's own zero
        # rather than against where it lands on the ruler.
        if sound.fade_in > 0:
            chain.append(f"afade=t=in:st=0:d={sound.fade_in}")
        if sound.fade_out > 0:
            ramp = min(sound.fade_out, sound.length)
            chain.append(
                f"afade=t=out:st={round(max(0.0, sound.length - ramp), 3)}:d={ramp}"
            )
        chain.append(
            f"aformat=sample_fmts=fltp:sample_rates={MIX_RATE}"
            f":channel_layouts={MIX_LAYOUT}"
        )
        # Milliseconds, and `all=1` so it delays every channel rather than just
        # the first — the classic way to end up with a lopsided stereo mix.
        chain.append(f"adelay={round(sound.start * 1000)}:all=1")
        steps.append(",".join(chain) + f"[{label}]")
        mixed.append(f"[{label}]")

    if mixed:
        steps.append(
            f"{''.join(mixed)}amix=inputs={len(mixed)}"
            f":normalize=0:dropout_transition=0[aout]"
        )

    args += [
        "-filter_complex", ";".join(steps),
        "-map", "[out]",
    ]
    if mixed:
        args += ["-map", "[aout]"]
    args += [
        # H3 renders at 24; pinning it keeps the concat from inheriting a
        # variable frame rate off the first input.
        "-r", str(h3_spec.FPS),
        "-c:v", "libx264",
        "-crf", str(DEFAULT_CRF),
        "-preset", DEFAULT_PRESET,
        "-pix_fmt", "yuv420p",
    ]
    if mixed:
        args += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2"]
    args.append(str(destination))
    return args


def _atempo_chain(speed: float) -> list[str]:
    """`atempo` filters whose product is `speed`.

    One filter only stretches between 0.5× and 2×, and the timeline allows 0.1×
    to 10×, so the rest is made by chaining. Returns nothing at 1×, where a
    filter would only cost a resample.
    """
    if speed <= 0 or abs(speed - 1.0) < 1e-9:
        return []
    parts: list[str] = []
    remaining = speed
    while remaining > ATEMPO_MAX + 1e-9:
        parts.append(f"atempo={ATEMPO_MAX}")
        remaining /= ATEMPO_MAX
    while remaining < ATEMPO_MIN - 1e-9:
        parts.append(f"atempo={ATEMPO_MIN}")
        remaining /= ATEMPO_MIN
    parts.append(f"atempo={round(remaining, 6):g}")
    return parts


def edl(clips: list[ResolvedClip], title: str,
        sounds: list[ResolvedSound] | None = None) -> str:
    """CMX3600, every event a straight cut.

    The track code carries what a clip is *heard* as, which is the one thing a
    round trip into another NLE would otherwise lose: `B` is picture with its
    own sound, `V` is picture alone — muted, or sounding from somewhere else —
    and a borrowed track gets its own `A` event naming the file it came from.
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    borrowed: dict[str, list[ResolvedSound]] = {}
    for sound in sounds or []:
        # A lane piece hangs off no clip, so there is no picture event to tie an
        # `A` event to. It gets its own run of events after the cuts.
        if sound.borrowed and sound.clip is not None:
            borrowed.setdefault(sound.clip.id, []).append(sound)

    def event(number: int, code: str, src_in: float, src_out: float,
              rec_in: float, rec_out: float) -> str:
        return (
            f"{number:03d}  AX       {code:<6}C        "
            f"{_timecode(src_in)} {_timecode(src_out)} "
            f"{_timecode(rec_in)} {_timecode(rec_out)}"
        )

    number = 0
    record = 0.0
    written: set[str] = set()
    for clip in clips:
        number += 1
        source_out = clip.source_in + clip.length * clip.speed
        heard = clip.clip.audio
        lines.append(event(
            number,
            "V" if (heard.muted or heard.borrowed) else "B",
            clip.source_in, source_out, record, record + clip.length,
        ))
        lines.append(f"* FROM CLIP NAME: {clip.take.filename}")
        lines.append(f"* SOURCE: {clip.name}")
        if clip.speed != 1.0:
            # M2 is how CMX3600 carries a speed change; the rate is fps-scaled.
            lines.append(f"M2   AX             {clip.speed * h3_spec.FPS:.1f}"
                         f"                {_timecode(clip.source_in)}")
        lines.append("")
        record += clip.length

        # Once per clip, not once per piece: a clip crossing a generation join
        # has two picture events but only one borrowed track.
        if clip.clip.id in written:
            continue
        written.add(clip.clip.id)
        for sound in borrowed.get(clip.clip.id, []):
            number += 1
            lines.append(event(
                number, "A", sound.source_in,
                sound.source_in + sound.length * sound.speed,
                sound.start, sound.start + sound.length,
            ))
            lines.append(f"* FROM CLIP NAME: {sound.path.name}")
            lines.append(f"* SOURCE: {sound.name} (sound only)")
            lines.append("")

    lane = sorted((sound for sound in sounds or [] if sound.clip is None),
                  key=lambda sound: sound.start)
    if lane:
        lines.append("* AUDIO LANE: separate sound events follow the picture edit.")
        lines.append("* WARNING: overlapping audio events require separate tracks in the target NLE.")
    for sound in lane:
        number += 1
        lines.append(event(
            number, "A", sound.source_in, sound.source_out,
            sound.start, sound.start + sound.length,
        ))
        lines.append(f"* FROM CLIP NAME: {sound.path.name}")
        lines.append(f"* SOURCE: {sound.name} (audio lane)")
        if sound.speed != 1.0:
            lines.append(f"M2   AX             {sound.speed * h3_spec.FPS:.1f}"
                         f"                {_timecode(sound.source_in)}")
        if sound.gain_db or sound.fade_in or sound.fade_out:
            lines.append(
                f"* WARNING: apply audio gain {sound.gain_db:g} dB, "
                f"fade in {sound.fade_in:g}s, fade out {sound.fade_out:g}s manually."
            )
        lines.append("")

    return "\n".join(lines)


def clip_list(clips: list[ResolvedClip], title: str,
              sounds: list[ResolvedSound] | None = None) -> str:
    """Plain text: which takes went in, and what each one is heard as."""
    heard: dict[str, list[ResolvedSound]] = {}
    lane: list[ResolvedSound] = []
    for sound in sounds or []:
        if sound.clip is None:
            lane.append(sound)
        else:
            heard.setdefault(sound.clip.id, []).append(sound)

    lines = [f"{title}", "=" * len(title), ""]
    written: set[str] = set()
    for index, clip in enumerate(clips, start=1):
        speed = "" if clip.speed == 1.0 else f"  @{clip.speed}x"
        lines.append(
            f"{index:2d}. {_timecode(clip.start)}  {clip.length:6.3f}s{speed}  "
            f"{clip.name}"
        )
        lines.append(f"    {clip.take.filename}  "
                     f"[{clip.source_in}s-{clip.source_out}s]")
        if clip.take.seed is not None:
            lines.append(f"    seed {clip.take.seed}")
        # Only worth a line when the sound is not simply this file's own, and
        # only once per clip however many pieces the picture is in.
        if clip.clip.id not in written:
            written.add(clip.clip.id)
            for note in _sound_notes(clip, heard.get(clip.clip.id, [])):
                lines.append(f"    {note}")
        lines.append("")
    if lane:
        # Listed apart from the numbered cuts, because its whole point is that it
        # does not line up with them.
        lines += ["audio lane", "-" * 10]
        for sound in sorted(lane, key=lambda s: s.start):
            lines.append(
                f"    {_timecode(sound.start)}  {sound.length:6.3f}s  "
                f"{sound.name}{_gain_note(sound.gain_db)}"
            )
            lines.append(f"    {sound.path.name}  "
                         f"[{sound.source_in}s-{sound.source_out}s]")
        lines.append("")

    total = sum(clip.length for clip in clips)
    lines.append(f"{len(clips)} pieces, {round(total, 3)}s")
    return "\n".join(lines)


# The old name, kept because the format id and the route both still say
# "shotlist", and renaming a saved export format is not this refactor's job.
shot_list = clip_list


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_duration(path: Path) -> float | None:
    """How long a file really is, or None when it cannot be asked.

    A take records the length that was *requested*. If the file disagrees —
    an interrupted render, a workflow that trimmed on the way out — then
    every trim past the real end is silently dropped and the export comes
    out short. Cheap to check, and the alternative is finding out by
    watching the result.
    """
    if shutil.which("ffprobe") is None:
        return None
    try:
        finished = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return round(float(finished.stdout.strip()), 3)
    except ValueError:
        return None


def has_audio(path: Path) -> bool | None:
    """Whether a file carries an audio stream, or None when it cannot be asked.

    None rather than False when ffprobe is missing: "I don't know" and "there is
    no sound" lead to opposite decisions, and guessing False would silence an
    export on a machine that simply lacks a tool.
    """
    if shutil.which("ffprobe") is None:
        return None
    try:
        finished = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(finished.stdout.strip())


def playable_sounds(
    sounds: list[ResolvedSound],
) -> tuple[list[ResolvedSound], list[str]]:
    """Split the mix into what ffmpeg can read and what it cannot.

    A silent input is not a warning to print after the fact — mapping `[n:a]` on
    a file with no audio stream fails the entire render, taking the picture with
    it. Dropping those pieces costs a silence; keeping them costs the export.
    """
    known: dict[Path, bool | None] = {}
    keep: list[ResolvedSound] = []
    dropped: list[str] = []
    for sound in sounds:
        if sound.path not in known:
            # A file that is not there is as unreadable as one with no sound in
            # it, and the picture resolver has already raised on missing takes,
            # so this can only be a borrowed lineage that has gone astray.
            known[sound.path] = has_audio(sound.path) if sound.path.is_file() else False
        if known[sound.path] is False:
            if sound.name not in dropped:
                dropped.append(sound.name)
            continue
        keep.append(sound)
    return keep, dropped


def short_sources(clips: list[ResolvedClip], tolerance: float = 0.1) -> list[str]:
    """Clips whose file is shorter than the edit expects. Empty when unknown."""
    found: list[str] = []
    for clip in clips:
        if not clip.path.is_file():
            continue
        real = probe_duration(clip.path)
        if real is not None and clip.source_out > real + tolerance:
            found.append(
                f"{clip.name}: needs {clip.source_out}s, file is {real}s"
            )
    return found


def run_ffmpeg(command: list[str], timeout: int = 3600) -> tuple[bool, str]:
    """Run the export. Returns success plus whatever ffmpeg said about it.

    Never raises: a failed export is a message to show, not a crashed server.
    """
    if not have_ffmpeg():
        return False, "ffmpeg is not on PATH."
    try:
        finished = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)

    if finished.returncode == 0:
        return True, "Exported."
    # ffmpeg's diagnosis is in the last few lines; the rest is banner.
    tail = (finished.stderr or "").strip().splitlines()[-6:]
    return False, "\n".join(tail) or f"ffmpeg exited {finished.returncode}."


def _sound_notes(clip: ResolvedClip, sounds: list[ResolvedSound]) -> list[str]:
    """What to say about a clip's sound, or nothing when it is unremarkable."""
    heard = clip.clip.audio
    if heard.muted:
        return ["sound: muted"]
    if not heard.borrowed:
        return [f"sound: own{_gain_note(heard.gain_db)}"] if heard.gain_db else []
    if not sounds:
        return ["sound: borrowed, but nothing resolved — silent"]
    files = ", ".join(dict.fromkeys(sound.path.name for sound in sounds))
    offset = f", offset {heard.offset:+g}s" if heard.offset else ""
    return [f"sound: from {files}{offset}{_gain_note(heard.gain_db)}"]


def _gain_note(gain_db: float) -> str:
    return f", {gain_db:+g} dB" if gain_db else ""


def _canvas(clips: list[ResolvedClip]) -> tuple[int, int]:
    """The canvas everything is fitted to: the first piece's own segment."""
    first = clips[0].source.segments[0].prompt
    return first.width, first.height


def _timecode(seconds: float, fps: int = h3_spec.FPS) -> str:
    """HH:MM:SS:FF, which is what an EDL wants."""
    total = max(0, round(seconds * fps))
    frames = total % fps
    whole = total // fps
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}:{frames:02d}"


def summary(clips: list[ResolvedClip],
            sounds: list[ResolvedSound] | None = None) -> dict[str, Any]:
    sounds = sounds or []
    audible = {sound.clip.id for sound in sounds if sound.clip is not None}
    return {
        "pieces": len(clips),
        "duration": round(sum(clip.length for clip in clips), 3),
        "audio": [
            {
                "clip": sound.clip.id if sound.clip else None,
                "lane": sound.lane,
                "name": sound.name,
                "file": sound.path.name,
                "starts_at": sound.start,
                "length": sound.length,
                "source_in": sound.source_in,
                "source_out": sound.source_out,
                "speed": sound.speed,
                "gain_db": sound.gain_db,
                "borrowed": sound.borrowed,
                "exists": sound.path.is_file(),
            }
            for sound in sounds
        ],
        # Named rather than counted: "three clips are silent" sends you hunting.
        "silent": list(dict.fromkeys(
            clip.name for clip in clips if clip.clip.id not in audible
        )),
        "clips": [
            {
                "name": clip.name,
                "clip": clip.clip.id,
                "source": clip.source.id,
                "file": clip.take.filename,
                "starts_at": clip.start,
                "length": clip.length,
                "source_in": clip.source_in,
                "source_out": clip.source_out,
                "speed": clip.speed,
                "exists": clip.path.is_file(),
            }
            for clip in clips
        ],
    }
