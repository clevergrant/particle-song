/**
 * Timbre derivation from force matrix (§8).
 *
 * Each particle type's "sociability" (average attraction to others)
 * maps to a continuous waveform spectrum and envelope curve shapes.
 */

import type { ForceMatrix } from "../particles/particle";
import type { WaveformParams } from "./types";

/* ------------------------------------------------------------------ */
/*  Sociability score                                                  */
/* ------------------------------------------------------------------ */

/**
 * Compute the sociability score for a particle type.
 * Average of its force matrix row values against all other types,
 * normalized to [0, 1].
 *
 * @param forceMatrix - Full force matrix
 * @param typeKey     - The type to score
 * @param typeKeys    - All type keys
 * @returns 0 (antisocial / repulsive) to 1 (sociable / attractive)
 */
export function sociabilityScore(
  forceMatrix: ForceMatrix,
  typeKey: string,
  typeKeys: readonly string[],
): number {
  const row = forceMatrix[typeKey];
  if (!row || typeKeys.length <= 1) return 0.5;

  let sum = 0;
  let count = 0;
  for (const otherKey of typeKeys) {
    if (otherKey === typeKey) continue;
    sum += row[otherKey] ?? 0;
    count++;
  }

  const avg = count > 0 ? sum / count : 0;
  // Force matrix values range [-1, 1]; map to [0, 1]
  return (avg + 1) / 2;
}

/**
 * Compute sociability scores for all types, then normalize to [0, 1]
 * across the full range of observed values.
 */
export function computeAllSociabilities(
  forceMatrix: ForceMatrix,
  typeKeys: readonly string[],
): ReadonlyMap<string, number> {
  if (typeKeys.length === 0) return new Map();

  const raw = new Map<string, number>();
  let min = Infinity, max = -Infinity;
  for (const key of typeKeys) {
    const score = sociabilityScore(forceMatrix, key, typeKeys);
    raw.set(key, score);
    if (score < min) min = score;
    if (score > max) max = score;
  }

  const range = max - min;
  const result = new Map<string, number>();
  for (const [key, score] of raw) {
    result.set(key, range > 0 ? (score - min) / range : 0.5);
  }
  return result;
}

/* ------------------------------------------------------------------ */
/*  Waveform blend                                                     */
/* ------------------------------------------------------------------ */

/**
 * Map sociability to waveform blend parameters.
 *
 * 0.0 (antisocial) → sawtooth (all harmonics, harsh)
 * 0.33             → square (odd harmonics, buzzy)
 * 0.66             → triangle (few harmonics, soft)
 * 1.0 (sociable)   → sine (fundamental only, warm)
 */
export function sociabilityToWaveform(sociability: number): WaveformParams {
  return { sociability, blend: sociability };
}

/**
 * Convert a WaveformParams blend value to the two nearest OscillatorType
 * values and an interpolation factor.
 *
 * This is used by the audio graph to crossfade between two oscillator types.
 */
/**
 * Volume scalar that attenuates harsh (low-sociability) waveforms so
 * sawtooth-heavy voices don't dominate the mix.
 *
 * Maps sociability [0, 1] → gain [0.35, 1.0] with a sqrt curve so the
 * penalty tapers off quickly once you leave pure sawtooth territory.
 */
export function sociabilityGain(sociability: number): number {
  const MIN_GAIN = 0.35;
  const clamped = Math.max(0, Math.min(1, sociability));
  return MIN_GAIN + (1 - MIN_GAIN) * Math.sqrt(clamped);
}

/**
 * Lowpass cutoff that scales with waveform blend so harmonically rich
 * waveforms (sawtooth) get their highs rolled off while pure sine is
 * essentially unfiltered.
 *
 * blend 0 (sawtooth) → 1 500 Hz ;  blend 1 (sine) → 20 000 Hz
 * Exponential sweep for perceptually even brightness change.
 */
export function waveformLowpassCutoff(blend: number): number {
  const MIN_CUTOFF = 1500;
  const MAX_CUTOFF = 20000;
  const clamped = Math.max(0, Math.min(1, blend));
  return MIN_CUTOFF * Math.pow(MAX_CUTOFF / MIN_CUTOFF, clamped);
}

export function waveformBlendToTypes(
  blend: number,
): { typeA: OscillatorType; typeB: OscillatorType; mix: number } {
  const clamped = Math.max(0, Math.min(1, blend));

  if (clamped <= 0.33) {
    const mix = clamped / 0.33;
    return { typeA: "sawtooth", typeB: "square", mix };
  }
  if (clamped <= 0.66) {
    const mix = (clamped - 0.33) / 0.33;
    return { typeA: "square", typeB: "triangle", mix };
  }
  const mix = (clamped - 0.66) / 0.34;
  return { typeA: "triangle", typeB: "sine", mix };
}
