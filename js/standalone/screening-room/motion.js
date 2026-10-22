/*
 * Screening Room — the two motion helpers that need JavaScript.
 * Framework-free ES module; React, Svelte or plain pages can all call these.
 */

/**
 * Mark an element as just changed. Restarts the fade even if one is already
 * running, so a number that changes twice in a row flashes twice.
 * kind: 'text' (the value turns amber), 'edge' (the border does), 'bg' (a wash).
 */
export function flash(el, kind = 'text') {
  if (!el) return;
  const cls = kind === 'bg' ? 'sr-flash' : `sr-flash-${kind}`;
  el.classList.remove(cls);
  void el.offsetWidth; // a reflow, so removing and re-adding restarts it
  el.classList.add(cls);
}

const VIEW_ANIMATIONS = new Set(['sr-fade-out', 'sr-title-out', 'sr-title-in', 'sr-rise-in']);

/**
 * Resolves once every stage-change animation inside `el` has finished.
 * Ignores looping animations (shimmer, breathing), which never finish.
 */
export function settled(el) {
  return Promise.all(
    el
      .getAnimations({ subtree: true })
      .filter((a) => VIEW_ANIMATIONS.has(a.animationName))
      .map((a) => a.finished.catch(() => {})),
  );
}

/**
 * Swap one .sr-view for another with the title-card transition.
 * Both must be .sr-view elements; `from` may be null on first show.
 */
export async function changeView(from, to) {
  if (from) {
    from.classList.add('is-leaving');
    await settled(from);
    from.classList.remove('is-current', 'is-leaving');
  }
  to.classList.add('is-current', 'is-entering');
  await settled(to);
  to.classList.remove('is-entering');
}
