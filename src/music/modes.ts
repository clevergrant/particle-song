/**
 * Scale/mode definitions and fishtail note selection.
 *
 * Extracted from chord-progression.ts. Only the 9 modes used by the
 * stability-band system (§4.2) are included, plus the fishtail algorithm
 * for stable pitch assignment.
 *
 * Pure data + pure functions — no side effects.
 */

import type { ModeDefinition } from "./types";

/* ------------------------------------------------------------------ */
/*  9 stability-band modes (§4.2)                                      */
/*  Ordered from highest stability → lowest stability                  */
/* ------------------------------------------------------------------ */

const LYDIAN: ModeDefinition = {
  name: "Lydian",
  scaleSemitones: [0, 2, 4, 6, 7, 9, 11],
};

const IONIAN: ModeDefinition = {
  name: "Ionian",
  scaleSemitones: [0, 2, 4, 5, 7, 9, 11],
};

const MIXOLYDIAN: ModeDefinition = {
  name: "Mixolydian",
  scaleSemitones: [0, 2, 4, 5, 7, 9, 10],
};

const DORIAN: ModeDefinition = {
  name: "Dorian",
  scaleSemitones: [0, 2, 3, 5, 7, 9, 10],
};

const AEOLIAN: ModeDefinition = {
  name: "Aeolian",
  scaleSemitones: [0, 2, 3, 5, 7, 8, 10],
};

const PHRYGIAN: ModeDefinition = {
  name: "Phrygian",
  scaleSemitones: [0, 1, 3, 5, 7, 8, 10],
};

const LOCRIAN: ModeDefinition = {
  name: "Locrian",
  scaleSemitones: [0, 1, 3, 5, 6, 8, 10],
};

const WHOLE_TONE: ModeDefinition = {
  name: "Whole Tone",
  scaleSemitones: [0, 2, 4, 6, 8, 10],
};

const CHROMATIC: ModeDefinition = {
  name: "Chromatic",
  scaleSemitones: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

/**
 * Stability bands ordered from highest (index 0) to lowest (index 8).
 * Net stability maps linearly into this array via §4.2 boundaries.
 */
export const STABILITY_BANDS: readonly ModeDefinition[] = [
  LYDIAN,       // 0.85–1.00
  IONIAN,       // 0.74–0.85
  MIXOLYDIAN,   // 0.64–0.74
  DORIAN,       // 0.53–0.64
  AEOLIAN,      // 0.43–0.53
  PHRYGIAN,     // 0.32–0.43
  LOCRIAN,      // 0.21–0.32
  WHOLE_TONE,   // 0.11–0.21
  CHROMATIC,    // 0.00–0.11
];

/** Band boundaries (upper thresholds, descending). Band i covers [BAND_BOUNDARIES[i+1], BAND_BOUNDARIES[i]). */
export const BAND_BOUNDARIES: readonly number[] = [
  1.00, 0.85, 0.74, 0.64, 0.53, 0.43, 0.32, 0.21, 0.11, 0.00,
];

/* ------------------------------------------------------------------ */
/*  Diatonic chords                                                    */
/* ------------------------------------------------------------------ */

/**
 * Absolute pitch classes of the triad built on a scale degree (stacked
 * scale thirds: degrees d, d+2, d+4). Degree 0 = I, 4 = V, etc.
 * Works for any scale length (whole-tone, chromatic included).
 */
export function diatonicTriad(
  mode: ModeDefinition,
  rootSemitone: number,
  degreeIndex: number,
): ReadonlySet<number> {
  const scale = mode.scaleSemitones;
  const n = scale.length;
  if (n === 0) return new Set([((rootSemitone % 12) + 12) % 12]);
  const d = ((degreeIndex % n) + n) % n;
  const pcs = new Set<number>();
  for (const step of [0, 2, 4]) {
    const semitone = scale[(d + step) % n];
    pcs.add(((rootSemitone + semitone) % 12 + 12) % 12);
  }
  return pcs;
}
