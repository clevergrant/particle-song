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

export const LYDIAN: ModeDefinition = {
  name: "Lydian",
  scaleSemitones: [0, 2, 4, 6, 7, 9, 11],
};

export const IONIAN: ModeDefinition = {
  name: "Ionian",
  scaleSemitones: [0, 2, 4, 5, 7, 9, 11],
};

export const MIXOLYDIAN: ModeDefinition = {
  name: "Mixolydian",
  scaleSemitones: [0, 2, 4, 5, 7, 9, 10],
};

export const DORIAN: ModeDefinition = {
  name: "Dorian",
  scaleSemitones: [0, 2, 3, 5, 7, 9, 10],
};

export const AEOLIAN: ModeDefinition = {
  name: "Aeolian",
  scaleSemitones: [0, 2, 3, 5, 7, 8, 10],
};

export const PHRYGIAN: ModeDefinition = {
  name: "Phrygian",
  scaleSemitones: [0, 1, 3, 5, 7, 8, 10],
};

export const LOCRIAN: ModeDefinition = {
  name: "Locrian",
  scaleSemitones: [0, 1, 3, 5, 6, 8, 10],
};

export const WHOLE_TONE: ModeDefinition = {
  name: "Whole Tone",
  scaleSemitones: [0, 2, 4, 6, 8, 10],
};

export const CHROMATIC: ModeDefinition = {
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
/*  Scale helpers                                                      */
/* ------------------------------------------------------------------ */

/** Get the full scale semitones as a Set (pitch classes mod 12). */
export function scaleTonesSet(mode: ModeDefinition): ReadonlySet<number> {
  return new Set(mode.scaleSemitones.map(s => s % 12));
}

/* ------------------------------------------------------------------ */
/*  Fishtail note selection                                            */
/* ------------------------------------------------------------------ */

/**
 * Walk scale degree indices by thirds (step by 2, mod N).
 * For a 7-note scale: [0, 2, 4, 6, 1, 3, 5].
 * For a 5-note scale: [0, 2, 4, 1, 3].
 */
export function fishtailDegreeOrder(scaleLength: number): readonly number[] {
  const order: number[] = [];
  const visited = new Set<number>();
  let idx = 0;
  for (let i = 0; i < scaleLength; i++) {
    order.push(idx);
    visited.add(idx);
    idx = (idx + 2) % scaleLength;
    while (visited.has(idx) && visited.size < scaleLength) {
      idx = (idx + 1) % scaleLength;
    }
  }
  return order;
}

/**
 * Compute a single pass of fishtail MIDI notes for one octave root.
 *
 * Degree ordering walks by thirds. Octave placement alternates UP/DOWN:
 * - UP notes: nearest instance of that degree above the root pitch
 * - DOWN notes: nearest instance of that degree below the root pitch
 *
 * All notes stay within one octave of the root.
 */
function fishtailPass(
  rootMidi: number,
  scaleSemitones: readonly number[],
  keyRootPitchClass?: number,
): readonly number[] {
  const n = scaleSemitones.length;
  const order = fishtailDegreeOrder(n);
  const result: number[] = [];
  const rootOctaveBase = rootMidi - (rootMidi % 12);
  const keyPC = keyRootPitchClass ?? (rootMidi % 12);

  for (let i = 0; i < n; i++) {
    const degreeIdx = order[i];
    const semitoneInScale = scaleSemitones[degreeIdx];
    const pitchClass = (keyPC + semitoneInScale) % 12;

    if (i === 0) {
      result.push(rootMidi);
    } else if (i % 2 === 1) {
      let midi = rootOctaveBase + pitchClass;
      while (midi <= rootMidi) midi += 12;
      result.push(midi);
    } else {
      let midi = rootOctaveBase + pitchClass;
      while (midi >= rootMidi) midi -= 12;
      result.push(midi);
    }
  }

  return result;
}

/**
 * Get the MIDI note at a given index in the infinite fishtail sequence.
 *
 * Pass 0: root octave. Subsequent passes fishtail outward:
 * root+12, root-12, root+24, root-24, ...
 */
export function fishtailMidi(
  rootMidi: number,
  mode: ModeDefinition,
  index: number,
  keyRootPitchClass?: number,
): number {
  const n = mode.scaleSemitones.length;
  const passIndex = Math.floor(index / n);
  const noteIndex = index % n;

  let octaveOffset: number;
  if (passIndex === 0) {
    octaveOffset = 0;
  } else {
    const half = Math.ceil(passIndex / 2);
    octaveOffset = passIndex % 2 === 1 ? half * 12 : -half * 12;
  }

  const passRoot = rootMidi + octaveOffset;
  const passNotes = fishtailPass(passRoot, mode.scaleSemitones, keyRootPitchClass);
  return passNotes[noteIndex];
}
