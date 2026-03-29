/**
 * Transition buffer system for smooth key/scale changes (§3.6).
 *
 * When a scale or key change produces too much dissonance (≥3 differing
 * pitch classes), a buffer bar is inserted using only the intersection
 * of the outgoing and incoming pitch sets.
 */

import type { ModeDefinition } from "./types";
import { scaleTonesSet } from "./modes";

/* ------------------------------------------------------------------ */
/*  Dissonance threshold                                               */
/* ------------------------------------------------------------------ */

export const DISSONANCE_THRESHOLD = 3;

/* ------------------------------------------------------------------ */
/*  Pitch class set operations                                         */
/* ------------------------------------------------------------------ */

/** Get the absolute pitch classes for a mode rooted at a given semitone. */
export function pitchClassSet(mode: ModeDefinition, rootSemitone: number): ReadonlySet<number> {
  return new Set(mode.scaleSemitones.map(s => (s + rootSemitone) % 12));
}

/** Count of pitch classes in A that are not in B, plus those in B not in A. */
export function dissonanceScore(a: ReadonlySet<number>, b: ReadonlySet<number>): number {
  let diff = 0;
  for (const pc of a) if (!b.has(pc)) diff++;
  for (const pc of b) if (!a.has(pc)) diff++;
  return diff;
}

/** Intersection of two pitch class sets. */
export function intersectPitchClasses(
  a: ReadonlySet<number>,
  b: ReadonlySet<number>,
): ReadonlySet<number> {
  const result = new Set<number>();
  for (const pc of a) {
    if (b.has(pc)) result.add(pc);
  }
  return result;
}

/* ------------------------------------------------------------------ */
/*  Buffer decision                                                    */
/* ------------------------------------------------------------------ */

/**
 * Determine whether a transition buffer bar is needed and compute its pitch set.
 *
 * @param prevMode       - Outgoing mode
 * @param prevRoot       - Outgoing root semitone (0–11)
 * @param nextMode       - Incoming mode
 * @param nextRoot       - Incoming root semitone (0–11)
 * @param currentBuffer  - Pitch set of current buffer bar (if already in a buffer)
 * @returns null if no buffer needed, otherwise the buffer's pitch class set
 */
export function computeTransitionBuffer(
  prevMode: ModeDefinition,
  prevRoot: number,
  nextMode: ModeDefinition,
  nextRoot: number,
  currentBuffer: ReadonlySet<number> | null = null,
): ReadonlySet<number> | null {
  const prevFull = pitchClassSet(prevMode, prevRoot);
  const incoming = pitchClassSet(nextMode, nextRoot);

  // Always compare full scales to decide if a buffer is needed.
  // Using the narrowed buffer as outgoing would cause infinite buffering
  // because the small intersection has high dissonance against any full scale.
  const score = dissonanceScore(prevFull, incoming);
  if (score < DISSONANCE_THRESHOLD) return null;

  // If already in a buffer, narrow further (intersection can only shrink).
  // Otherwise start from the full previous scale.
  const outgoing = currentBuffer ?? prevFull;
  return intersectPitchClasses(outgoing, incoming);
}
