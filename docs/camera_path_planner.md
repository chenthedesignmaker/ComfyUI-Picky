# Camera path planner

In **Scene**, choose **Camera path** beside **Settings** in any shot header.
This button stays visible when the shot or its Camera field is collapsed.
You can also use **Plan camera path** inside an expanded **Camera** field.

Choose an orbit, dolly, rise, or full-circle preset. Describe the object or person
to aim at, then select a keyframe to change orbit, height angle, distance, and
time. Add intermediate points for a more detailed path. Drag the preview to
adjust the selected camera; the number fields provide the same controls with
a keyboard. Play or scrub to preview the planned motion.

The v19.1-inspired update adds lowering and static presets, monotone cubic
interpolation without overshoot, and these editing aids:

- **Even travel timing** distributes the available time by a combined angular
  and relative-radius distance. Explicit hold durations are preserved. This is
  a planning metric, not a promise of constant physical speed in an unknown scene.
- **Unwrap angles** takes the short arc across angle boundaries, preserving
  explicit complete turns. Inspect the path after using it.
- **Add 0.5s end holds** adds a stationary camera interval at each end.
- **Close orbit** sets the endpoint to ±360°, opening elevation and radius.

Draw a subject box on the reference preview or enter normalized left/top/width/
height values. It identifies the initial target without cropping or locking its
screen position. The Motion Frame preview shows the video's opening frame.

The opening keyframe is fixed at 0° orbit, 0° height, and 1× distance. Orbit
angles are relative to that opening camera, with positive angles moving toward
camera-right. A height angle changes camera elevation around the target;
distance is a multiplier of the opening radius, ranging from 0.05× to 4×.
Values closer to 1× give gentler moves; values approaching 0.05× request a
stronger push-in. This diagram is a schematic,
not a reconstruction of the scene. An attached first frame appears beside it
as a visual reference.

**Apply camera path** puts the path on that beat. Save and render normally.
The path follows the beat when you change its duration, move it, or split it
into its own source. Scene, action, dialogue, audio, references, and written
camera text remain saved. **Use written camera** removes the path and restores
the original camera direction. Cancel and Escape discard the planner draft.

**Freeze scene** adds a direction to hold objects and people in their captured
pose while the camera moves. Leave it off for a speaking or moving performance.

Paths compile into the usual H3 prompt on the server. Scene paths work with the
editor's normal generation profiles and quality presets. Preview shows the actual compiled instructions.
Saved keyframes are available to agents as `beats[].camera_path` in `/schema`.

**Motion Frame** is a separate camera-pass mode. Use one beat covering the whole
generation. Pick an existing video reference or a project take, or upload a
2–15-second clip, then apply the path and render with a Ref2VA-capable profile
(locally, **H3 Hybrid local t2v**). Saved first/last images remain in the document
but are inactive while this mode is selected. Switching back restores their use.
Other attached omni-references remain active. Select a duration matching the
action to preserve; reference frames beyond the generation length are omitted.

The bundled `VideoPrompterCameraFrames` node uses the file's actual FPS to sample
at 24 fps and trims to the 17k+5 grid (up to 16 trailing frames omitted). It caps
reference resolution at about 0.26 MP to bound video-VAE cost. The frames feed
native `MiniMaxH3ReferenceToVideo`: Qwen sees video frames and timestamps, and
the video VAE supplies reference latents. Source audio is not connected by this
mode; the normal prompt soundscape still generates audio. Camera holds preserve
the ongoing action. The requested replacement camera path remains text.

**Loop closure → Auto** wires the opening image as the native last-frame input
only for one full-generation frozen path, exactly ±360°, matching elevation and
radius, and no omni-references or conflicting explicit last frame. It defaults
off. The planner and prompt preview report eligibility. The anchor constrains
the endpoint; it cannot prove that a complete orbit happened between endpoints.

**Advanced → Prompt experiments** contains direction plus numeric keyframes and
a numbers-led recipe. These are optional local text recipes inspired by the
upstream experiments, not a native H3 geometric protocol or a byte-for-byte port.
Compare identical seeds, models and quality before drawing accuracy conclusions.

This integration uses the same prompt-based approach explored in the
[camera-control workflow](https://github.com/NyckM/3d-Camera-control-H3-Minimax).
It is an editor-native planner, not an embedded ComfyUI node. Exact angles,
timing, and scene geometry remain model-dependent. No upstream camera plugin
installation or update is required. Rewriting a shot's entire storyboard through the language-model
writer may replace its camera path along with the rest of that shot.

## Local validation — 2026-09-13

The full 1,722-test suite passed; focused checks and desktop/mobile browser
checks also passed after the final prompt adjustment. Browser coverage included
drawn boxes, timing helpers, loop eligibility, save/reload, upload, restoring
saved still anchors, experimental controls, and horizontal overflow.

Two actual Motion Frame renders completed on the RTX 4090 with the H3 Hybrid
profile: 124 frames at 24 fps, 576×352, eight steps, seed 20260913, requesting a
30° orbit. The source was a five-second excerpt of a woman handling a red shirt.
The runs took approximately 86s and 47s. The second added an explicit opening-
viewpoint/pose instruction while holding the seed and generation recipe fixed.

Both demonstrate working video-reference conditioning and visible camera
movement. Both also reinterpret the woman's pose and room layout and start from
a different viewpoint. The second instruction did not resolve this drift.
These are integration tests, not evidence of accurate 30° travel or exact action
preservation; Motion Frame is therefore labeled preview. Both takes are in the
editor project **Camera Pass Tests**. No comparison establishes an improvement
for the optional numeric prompt recipes.
