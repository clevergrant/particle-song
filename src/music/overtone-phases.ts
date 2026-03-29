/**
 * Overtone series progression (§4.3).
 *
 * An organism's age determines its harmonic vocabulary by recapitulating
 * the overtone series: octaves → fifths → thirds → sevenths → chromaticism.
 */

import type { HarmonicPhase, RegisterWidth } from "./types";

/* ------------------------------------------------------------------ */
/*  Phase definitions                                                  */
/* ------------------------------------------------------------------ */

/**
 * Available intervals per phase (semitone offsets from root).
 * Each phase accumulates — phase N includes all of phases 1..N-1.
 */
const PHASE_INTERVALS: readonly (readonly number[])[] = [
  [0],                                     // Phase 1: fundamental only
  [0, 12, -12],                            // Phase 2: + octaves
  [0, 12, -12, 7, -5],                     // Phase 3: + fifth (3:2)
  [0, 12, -12, 7, -5, 4, 3],              // Phase 4: + third (5:4, major & minor)
  [0, 12, -12, 7, -5, 4, 3, 10, 11],      // Phase 5: + seventh (7:4)
  [0, 12, -12, 7, -5, 4, 3, 10, 11, 2, 5, 9, 14], // Phase 6: + upper partials (9ths, 11ths, 13ths)
];

/* ------------------------------------------------------------------ */
/*  Phase computation                                                  */
/* ------------------------------------------------------------------ */

/**
 * Compute the harmonic phase for an organism based on its age in bars.
 *
 * @param ageInBars   - How many bars old the organism is
 * @param barsPerPhase - Configurable rate (how many bars per phase transition)
 * @returns HarmonicPhase with phase number (1–6) and available intervals
 */
export function computeHarmonicPhase(
  ageInBars: number,
  barsPerPhase: number,
): HarmonicPhase {
  const phase = Math.min(Math.floor(ageInBars / Math.max(1, barsPerPhase)) + 1, 6);
  return {
    phase,
    availableIntervals: PHASE_INTERVALS[phase - 1],
  };
}

/**
 * Map an organelle's cross-type link count to a harmonic phase.
 * More structural connections = richer harmonic vocabulary.
 *
 * 1 link  → phase 1: fundamental only
 * 2 links → phase 3: + fifths
 * 3 links → phase 4: + thirds
 * 4 links → phase 5: + sevenths
 * 5+ links → phase 6: upper partials
 */
export function crossTypeLinksToPhase(links: number): HarmonicPhase {
  let phase: number;
  if (links <= 1) phase = 1;
  else if (links === 2) phase = 3;
  else if (links === 3) phase = 4;
  else if (links === 4) phase = 5;
  else phase = 6;
  return {
    phase,
    availableIntervals: PHASE_INTERVALS[phase - 1],
  };
}

/**
 * Given a scale degree's semitone offset, check if it's available in the
 * current phase. If not, collapse to the nearest available interval.
 *
 * @param degreeSemitone  - The scale degree's semitone offset from root
 * @param phase           - Current harmonic phase
 * @returns The (possibly collapsed) semitone offset
 */
export function collapseToPhase(
  degreeSemitone: number,
  phase: HarmonicPhase,
): number {
  // Normalize to pitch class for comparison
  const pc = ((degreeSemitone % 12) + 12) % 12;

  // Check if this pitch class is available in the phase
  for (const interval of phase.availableIntervals) {
    const intervalPC = ((interval % 12) + 12) % 12;
    if (intervalPC === pc) return degreeSemitone;
  }

  // Collapse to nearest available interval (by pitch class distance)
  let bestInterval = 0;
  let bestDistance = 12;
  for (const interval of phase.availableIntervals) {
    const intervalPC = ((interval % 12) + 12) % 12;
    const dist = Math.min(
      ((pc - intervalPC) % 12 + 12) % 12,
      ((intervalPC - pc) % 12 + 12) % 12,
    );
    if (dist < bestDistance) {
      bestDistance = dist;
      bestInterval = interval;
    }
  }

  // Return the collapsed pitch, preserving the original octave intent
  const octave = Math.floor(degreeSemitone / 12);
  const collapsedPC = ((bestInterval % 12) + 12) % 12;
  return octave * 12 + collapsedPC;
}

/* ------------------------------------------------------------------ */
/*  Register width (§4.3.1)                                            */
/* ------------------------------------------------------------------ */

/**
 * Determine available octave range based on total organism count.
 */
export function computeRegisterWidth(organismCount: number): RegisterWidth {
  if (organismCount <= 1) return { lowOctave: 4, highOctave: 4 };
  if (organismCount <= 3) return { lowOctave: 3, highOctave: 5 };
  if (organismCount <= 6) return { lowOctave: 2, highOctave: 6 };
  return { lowOctave: 1, highOctave: 7 };
}

/**
 * Clamp a MIDI note to the current register width.
 */
export function clampToRegister(midiNote: number, width: RegisterWidth): number {
  const lowMidi = width.lowOctave * 12;      // C of low octave
  const highMidi = (width.highOctave + 1) * 12 - 1; // B of high octave
  if (midiNote < lowMidi) {
    // Shift up by octaves until in range
    while (midiNote < lowMidi) midiNote += 12;
  } else if (midiNote > highMidi) {
    // Shift down by octaves until in range
    while (midiNote > highMidi) midiNote -= 12;
  }
  return midiNote;
}
