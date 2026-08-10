/**
 * Bar grid timing — pure functions for BPM-based scheduling.
 *
 * All timing derives from timestamps, never frame counters (§7.2).
 * The bar grid is anchored to a single reference time (tSoundStart).
 */

/* ------------------------------------------------------------------ */
/*  Core timing                                                        */
/* ------------------------------------------------------------------ */

/** Duration of one bar in seconds (fixed 4 beats per bar). */
export function barDuration(
  bpm: number,
  timeMultiplier: number,
): number {
  return (60 / Math.max(20, bpm)) * 4 / timeMultiplier;
}

/** Which bar number we're in (0-based) given an absolute time. */
export function currentBarNumber(
  tSoundStart: number,
  now: number,
  barDur: number,
): number {
  if (barDur <= 0) return 0;
  return Math.floor((now - tSoundStart) / barDur);
}

/** Absolute time when bar N starts. */
export function barStartTime(
  tSoundStart: number,
  barNumber: number,
  barDur: number,
): number {
  return tSoundStart + barNumber * barDur;
}

/**
 * Evenly-spaced hit times within a bar.
 * Returns `count` timestamps starting from `barStart`.
 */
export function subdivisionTimes(
  barStart: number,
  barDur: number,
  count: number,
): readonly number[] {
  if (count <= 0) return [];
  const result: number[] = [];
  for (let i = 0; i < count; i++) {
    result.push(barStart + (i / count) * barDur);
  }
  return result;
}
