// Curated creative options, not rules about what an emotion must look like.
// Keep search terms short: providers match visual ingredients better than a script.
const idea = (id, title, effect, composition, color, lighting, camera, search) =>
  ({ id, title, effect, composition, color, lighting, camera, search });

export const FEELINGS = [
  { id: "tension", name: "Tension", approaches: [
    idea("pressure", "Pressure closing in", "Let the audience notice danger before anyone reacts.",
      "A tight medium shot, with a foreground edge partially enclosing the subject.",
      "A restrained palette with one contrasting detail drawing the eye.",
      "Directional side light leaves part of the background in shadow.",
      "The camera makes a small, very slow push toward the subject at a steady speed.", "tension tight framing shadow"),
    idea("waiting", "Something in the distance", "Make the viewer search an apparently quiet frame.",
      "An eye-level wide shot with the subject off-center and a clear view into the background.",
      "Neutral colors with subdued saturation.",
      "Uneven pools of practical light separate foreground and background.",
      "The camera remains locked off, with no movement or zoom.", "suspense wide shot doorway"),
  ] },
  { id: "isolation", name: "Isolation", approaches: [
    idea("small", "Small in the world", "Let the setting outweigh the person.",
      "A wide shot with a small subject near one edge and generous negative space.",
      "Muted, cool surroundings with restrained skin tones.",
      "Broad overcast light gives the space an even, subdued appearance.",
      "The camera slowly pulls back a short distance, revealing more empty space.", "loneliness wide shot negative space"),
    idea("apart", "Together, but apart", "Create distance even when other people are present.",
      "A medium shot through a foreground obstruction, separating the subject from the surrounding group.",
      "A subdued subject palette against slightly richer background colors.",
      "Soft side light singles out the subject while the background stays gently dimmer.",
      "The camera remains locked off, with no movement or zoom.", "isolated person crowd foreground framing"),
  ] },
  { id: "intimacy", name: "Intimacy", approaches: [
    idea("shared", "A shared space", "Keep both people in the same visual world.",
      "An eye-level medium two-shot, with little empty space between the subjects.",
      "Warm neutrals and gentle, natural skin tones.",
      "Soft window light wraps across both faces with gentle shadow falloff.",
      "The camera remains locked off at eye level, without zooming.", "intimate two shot soft window light"),
    idea("close", "A private moment", "Bring attention to a small expression or gesture.",
      "An eye-level close-up with a softly blurred background and relaxed headroom.",
      "Low-contrast earth tones with subtle warm highlights.",
      "A nearby practical lamp gives a soft, localized glow.",
      "The camera makes a subtle, slow push in over a short distance.", "intimate close up practical lamp"),
  ] },
  { id: "joy", name: "Joy", approaches: [
    idea("energy", "Joy in motion", "Give physical energy room to unfold.",
      "A roomy medium-wide shot with space ahead of the subject's movement.",
      "Clear, lively colors with natural skin tones.",
      "Bright, soft daylight keeps faces open and readable.",
      "The camera tracks alongside the subject smoothly at a moderate, even speed.", "joy tracking shot daylight"),
    idea("quiet", "Quiet delight", "Let a small change in expression carry the moment.",
      "An eye-level medium close-up with a simple, uncluttered background.",
      "Soft warm neutrals with one gentle color accent.",
      "Diffuse window light and soft fill create low-contrast shadows.",
      "The camera remains locked off, with no movement or zoom.", "happy quiet close up soft light"),
  ] },
  { id: "grief", name: "Grief", approaches: [
    idea("absence", "Make absence visible", "Let unused space carry some of the loss.",
      "A static medium-wide shot with the subject offset beside a conspicuously empty area.",
      "Restrained earth tones with low saturation.",
      "Soft, fading window light leaves gentle shadows across the room.",
      "The camera remains locked off, with no movement or zoom.", "grief empty space window light"),
    idea("witness", "Stay with the person", "Allow emotion without making the frame perform it.",
      "An eye-level close-up with a simple background and natural facial proportions.",
      "Natural, understated colors with soft contrast.",
      "Diffuse side light preserves subtle changes in the face.",
      "The camera holds completely still at eye level.", "grief close up natural light"),
  ] },
  { id: "anger", name: "Anger", approaches: [
    idea("suppressed", "Suppressed anger", "Build pressure beneath a controlled exterior.",
      "A tight eye-level close-up with cramped headroom and little surrounding space.",
      "A restrained, muted palette with cool neutral shadows.",
      "Hard side light creates a clear division between light and shadow across the face.",
      "The camera makes a small, very slow push toward the subject at a steady speed.", "suppressed anger close up side light"),
    idea("explosive", "Anger breaking out", "Make the frame feel reactive and physically close.",
      "A close medium shot in which the subject nearly fills the frame.",
      "Strong tonal contrast with restrained warm accents.",
      "Hard overhead light creates pronounced shadows under the brow and jaw.",
      "The handheld camera follows with small, quick reactive corrections, staying close to the subject.", "angry confrontation handheld harsh light"),
  ] },
  { id: "awe", name: "Awe", approaches: [
    idea("scale", "Reveal the scale", "Let the audience discover how large the world is.",
      "An extreme wide shot with a small human figure providing a clear scale reference.",
      "A broad, cohesive palette with luminous highlights.",
      "Directional backlight separates the distant layers and outlines the subject.",
      "The camera slowly tilts upward through a broad arc to reveal the height of the setting.", "awe extreme wide scale backlight"),
    idea("discovery", "Share the discovery", "Let the reveal arrive alongside a person's perspective.",
      "An over-the-shoulder wide shot with the subject at the edge and the view beyond unobstructed.",
      "Subdued foreground tones give way to richer colors in the distance.",
      "A softly shaded foreground frames a brighter, luminous background.",
      "The camera tracks forward a short distance, very slowly and smoothly.", "wonder over shoulder landscape reveal"),
  ] },
  { id: "unease", name: "Unease", approaches: [
    idea("wrong", "Almost normal", "Introduce a subtle imbalance in an ordinary scene.",
      "An eye-level wide shot with slightly excessive headroom and an off-center subject.",
      "Muted neutrals with a faint green cast in the background.",
      "Flat overhead practical light leaves the eye sockets slightly dark.",
      "The camera remains locked off, with no movement or zoom.", "uneasy wide shot overhead light"),
    idea("watched", "A watched space", "Make the viewpoint feel distant and concealed.",
      "A distant medium-long shot seen past an out-of-focus foreground edge.",
      "Low-saturation colors with dense, cool shadows.",
      "Small practical lights illuminate isolated areas of the background.",
      "The camera makes a barely perceptible, slow lateral slide over a short distance.", "voyeur foreground obstruction night"),
  ] },
  { id: "confidence", name: "Confidence", approaches: [
    idea("grounded", "Quiet authority", "Let stability and clarity establish confidence.",
      "A balanced eye-level medium shot with clean lines and comfortable headroom.",
      "A controlled palette with rich, natural midtones.",
      "A broad directional key light shapes the face while keeping both eyes readable.",
      "The camera remains locked off, with no movement or zoom.", "confident balanced medium shot"),
    idea("presence", "Command the space", "Give the subject a strong relationship to the setting.",
      "A low-angle medium-wide shot with the subject clearly separated from the background.",
      "Deep, clean colors with distinct tonal separation.",
      "Defined side light and a subtle rim separate the subject from the setting.",
      "The camera makes a short, deliberate forward track at a slow, constant speed.", "authority low angle rim light"),
  ] },
  { id: "confusion", name: "Confusion", approaches: [
    idea("searching", "Searching for an answer", "Keep the viewer close to a person's uncertainty.",
      "A medium close-up with an indistinct background and slightly off-center framing.",
      "Muted background colors keep attention on the face.",
      "Soft side light keeps small expressions visible.",
      "The camera makes a shallow, slow arc around the subject over a short distance.", "confused close up shallow focus"),
    idea("disoriented", "An unstable viewpoint", "Suggest a loss of bearings without changing the action.",
      "A medium-wide shot with a subtle tilted horizon and strong foreground depth.",
      "Mixed warm and cool areas divide the frame.",
      "Contrasting practical light sources create uneven illumination.",
      "The handheld camera drifts slightly from side to side with slow, uneven corrections.", "disorientation dutch angle mixed lighting"),
  ] },
  { id: "nostalgia", name: "Nostalgia", approaches: [
    idea("warm", "A gentle memory", "Give familiar details room to be noticed.",
      "An eye-level medium-wide shot with layered foreground and background details.",
      "Soft amber highlights, faded earth tones and gentle contrast.",
      "Low, warm sunlight falls across the setting with soft reflected fill.",
      "The camera makes a slow, short lateral track at an even speed.", "nostalgia warm sunlight interior"),
    idea("distant", "A memory at a distance", "Allow warmth and loss to coexist.",
      "A wide shot viewed through a doorway, with the subject set back in the frame.",
      "Faded, cool neutrals with a small warm area in the distance.",
      "Soft overcast light fills the foreground while a practical lamp warms the background.",
      "The camera remains locked off, with no movement or zoom.", "nostalgia doorway distant figure"),
  ] },
  { id: "awkwardness", name: "Awkwardness", approaches: [
    idea("deadpan", "Let the silence show", "Give the audience room to notice an uncomfortable gap.",
      "A static eye-level wide shot with conspicuous space between the subjects.",
      "Ordinary, slightly muted workplace neutrals.",
      "Even overhead practical lighting keeps the entire situation plainly visible.",
      "The camera remains locked off, with no movement or zoom.", "awkward comedy office wide shot"),
    idea("reaction", "Stay on the reaction", "Let a small expression reveal the social mismatch.",
      "An eye-level medium close-up with a little extra looking room.",
      "Natural colors with restrained contrast.",
      "Soft frontal daylight makes tiny changes in expression readable.",
      "The camera holds completely still at eye level.", "awkward reaction medium close up"),
  ] },
];

// Standalone performance snippets. They are copied explicitly, never included
// in an automatic visual-direction patch. Intensity describes the performance.
const action = (intensity, title, text) => ({ intensity, title, text });
export const ACTION_SNIPPETS = {
  tension: {
    pressure: [
      action("Subtle", "A held breath", "The character's hand stops just short of the door handle. They hold their breath, listening, then ease their fingers away."),
      action("Building", "The grip tightens", "The character grips the edge of the table until their knuckles pale. Their eyes stay on the doorway as their shoulders slowly rise."),
    ],
    waiting: [
      action("Subtle", "A second glance", "The character checks the empty hallway, turns back to their task, then looks over their shoulder once more."),
      action("Strong", "Waiting for the turn", "The character backs away from the closed door, one careful step at a time. They stop when the handle begins to turn."),
    ],
  },
  isolation: {
    small: [
      action("Subtle", "An unoccupied seat", "The character rests a hand on the back of the empty chair beside them, then lets it fall into their lap."),
      action("Strong", "Left behind", "The last people leave the room. The character starts to rise after them, hesitates, and sits back down alone."),
    ],
    apart: [
      action("Subtle", "A smile that fades", "The character offers a small smile toward the group. Nobody notices; the smile fades as they lower their eyes."),
      action("Building", "Outside the circle", "The character leans forward to join the conversation, then draws back as everyone turns toward someone else. They fold their hands and stay silent."),
    ],
  },
  intimacy: {
    shared: [
      action("Subtle", "Make a little room", "One character shifts closer on the bench, leaving their hand beside the other's. Their little fingers touch and neither pulls away."),
      action("Strong", "A shared lean", "One character rests their head on the other's shoulder. The other tilts gently toward them, and both settle into the same slow breathing rhythm."),
    ],
    close: [
      action("Subtle", "An unguarded look", "The character glances up, holds the other person's gaze a little longer than expected, and lets a small smile appear."),
      action("Building", "A careful touch", "The character reaches up to brush a stray hair from the other person's face. Their hand pauses at the cheek before slowly lowering."),
    ],
  },
  joy: {
    energy: [
      action("Building", "A spring in the step", "The character's walk breaks into a few quick skipping steps. They turn toward their companion with a grin they cannot contain."),
      action("Peak", "An uncontained celebration", "The character throws both arms up, jumps once, and lands laughing. They turn and pull their companion into a bouncing embrace."),
    ],
    quiet: [
      action("Subtle", "The smile arrives", "The character reads the note again. The corners of their mouth lift, and their shoulders relax as they hold it a little closer."),
      action("Building", "A laugh slips out", "The character presses their lips together to hold back a smile. A small laugh escapes; they look down, then back up with a broad grin."),
    ],
  },
  grief: {
    absence: [
      action("Subtle", "Set out of habit", "The character sets two cups on the table, pauses, then slowly returns one to the cupboard."),
      action("Strong", "An unfinished gesture", "The character reaches toward the empty side of the bed. Their hand stops on the cold pillow, and they remain bent over it without moving."),
    ],
    witness: [
      action("Subtle", "Keep it together", "The character swallows, blinks slowly, and steadies their breathing. Their eyes remain fixed on a point just below the other person's face."),
      action("Peak", "Composure gives way", "The character tries to draw a steady breath, but it catches. Their face crumples, their shoulders fold inward, and they cover their mouth as they begin to sob."),
    ],
  },
  anger: {
    suppressed: [
      action("Subtle", "Irritation under control", "The character's jaw tightens. They set the cup down with deliberate care, flatten their palm on the table, and hold the other person's gaze."),
      action("Building", "Barely contained", "The character presses their lips together and draws a sharp breath through the nose. Their fingers curl against the tabletop, then slowly straighten as they force their shoulders down."),
    ],
    explosive: [
      action("Strong", "A sudden confrontation", "The character pushes their chair back with a scrape and stands abruptly. They plant both hands on the table and lean toward the other person."),
      action("Peak", "The outburst", "The character strikes the tabletop with an open palm. Their whole body pitches forward; they pull back, turn sharply, and pace away with clenched hands."),
    ],
  },
  awe: {
    scale: [
      action("Subtle", "Look higher", "The character slows to a stop and gradually tilts their head back. Their mouth parts slightly as their gaze travels upward."),
      action("Strong", "Stopped in their tracks", "The character steps into the open and stops mid-stride. Their arms fall loosely to their sides as they turn slowly, taking in the scale around them."),
    ],
    discovery: [
      action("Subtle", "A glimpse", "The character leans around the corner, then slowly steps out from behind it. Their eyes widen as the view opens in front of them."),
      action("Building", "Share the wonder", "The character reaches back for their companion's hand without looking away. They draw the companion forward and point toward the distant view."),
    ],
  },
  unease: {
    wrong: [
      action("Subtle", "Something does not fit", "The character notices an object out of place. Their hand hovers above it before they withdraw and scan the room."),
      action("Building", "Check behind", "The character stops what they are doing and turns slowly toward the hallway. After a pause, they shift their chair so their back is against the wall."),
    ],
    watched: [
      action("Subtle", "A glance at the glass", "The character catches a movement in the window's reflection. They glance over their shoulder, then return to their task more slowly."),
      action("Strong", "Leave the light", "The character steps away from the lit window and draws the curtain with two fingers. They leave a narrow gap and watch through it, keeping their body back."),
    ],
  },
  confidence: {
    grounded: [
      action("Subtle", "Take the pause", "The character finishes placing their notes on the table before looking up. They meet the room's gaze and take one unhurried breath."),
      action("Building", "A steady response", "The character listens without interrupting, then gives a small nod. They straighten in their seat and rest their open hands on the table."),
    ],
    presence: [
      action("Building", "Enter with purpose", "The character walks directly into the room at an even pace. They stop at the head of the table and look from one person to the next."),
      action("Strong", "Hold the room", "The character steps forward and raises one open hand. As the surrounding movement settles, they lower it slowly and hold their position with relaxed shoulders."),
    ],
  },
  confusion: {
    searching: [
      action("Subtle", "Read it twice", "The character reads the same line again, their brow drawing together. They look up at the other person, then back down at the page."),
      action("Building", "The answer changes", "The character points to one item, hesitates, and shifts their finger to another. They open their mouth to respond, then stop and look between the two."),
    ],
    disoriented: [
      action("Building", "Retrace a step", "The character takes a few steps down the corridor, stops, and turns back. They check both directions before reaching for the wall."),
      action("Strong", "Lose their bearings", "The character turns toward one doorway, then another, breathing faster. They reach for a nearby surface to steady themselves and scan the room without settling on a direction."),
    ],
  },
  nostalgia: {
    warm: [
      action("Subtle", "A familiar detail", "The character runs a thumb along the worn edge of an old photograph. A small smile appears as they tilt it toward the light."),
      action("Building", "Remember the motion", "The character opens the old music box and turns its key from habit. Their fingers continue the familiar rhythm on the lid as they smile."),
    ],
    distant: [
      action("Subtle", "Pause at the threshold", "The character stops in the doorway and looks across the room. Their hand rests briefly on a familiar mark in the doorframe."),
      action("Strong", "Put the memory away", "The character lifts an old coat to their face and closes their eyes. After a long breath, they fold it carefully and return it to the box."),
    ],
  },
  awkwardness: {
    deadpan: [
      action("Subtle", "Fill the pause", "The character straightens a pen that is already straight, glances toward the other person, and returns their hands to their lap."),
      action("Strong", "The failed handshake", "The character offers a hand just as the other person turns away. They hold it there for a moment, then use the same hand to smooth their own sleeve."),
    ],
    reaction: [
      action("Subtle", "A late smile", "The character blinks once, then produces a small smile a beat too late. Their eyes dart briefly toward the nearest exit."),
      action("Building", "A laugh without company", "The character gives a short laugh and looks around for someone to join in. Finding only silence, they let the smile drop and clear their throat."),
    ],
  },
};

export const QUALITIES = [
  ["composition", "Composition"], ["color", "Color"],
  ["lighting", "Lighting"], ["camera", "Camera motion"],
];

export function referenceURL(query) {
  return `https://frameset.app/search?search=${encodeURIComponent(query.trim())}`;
}

export function directionText(approach, selected) {
  return QUALITIES.filter(([key]) => selected.includes(key)).map(([key]) => approach[key]).join(" ");
}

// The storyboard has no per-beat color/light fields. Add those descriptions to
// the environment prose; never rewrite action, dialogue, timing or scene defaults.
export function buildIdeaPatch(beat, approach, selected) {
  const patch = {};
  if (selected.includes("composition")) patch.angle = approach.composition;
  if (selected.includes("camera")) patch.camera = approach.camera;
  const additions = ["color", "lighting"].filter(key => selected.includes(key))
    .map(key => approach[key]).filter(text => !(beat.action || "").includes(text));
  if (additions.length) patch.action = `${beat.action || ""}${beat.action ? "\n\n" : ""}${additions.join(" ")}`;
  return Object.fromEntries(Object.entries(patch).filter(([key, value]) => value !== beat[key]));
}

// Beat indices are positional. Reject a preview after any beat is edited,
// reordered or removed, or after the user changes project/source/generation.
export function previewMatches(snapshot, preview) {
  return !!snapshot && snapshot.key === preview.key
    && JSON.stringify(snapshot.beats) === preview.beats;
}
