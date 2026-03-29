/**
 * Stability-band mode selection with hysteresis (§4.2, §4.2.1).
 *
 * Maps net_stability ∈ [0, 1] to one of 9 modes. Hysteresis prevents
 * flickering at band boundaries.
 */

import type { ModeDefinition } from "./types";
import { STABILITY_BANDS, BAND_BOUNDARIES } from "./modes";

/* ------------------------------------------------------------------ */
/*  Default hysteresis margin                                          */
/* ------------------------------------------------------------------ */

export const DEFAULT_HYSTERESIS_MARGIN = 0.03;

/* ------------------------------------------------------------------ */
/*  Band index lookup                                                  */
/* ------------------------------------------------------------------ */

/**
 * Raw band index for a stability value (no hysteresis).
 * Returns 0 (highest = Lydian) through 8 (lowest = Chromatic).
 */
function rawBandIndex(stability: number, preferNice?: boolean): number {
  // When preferNiceModes is on, boost stability so nicer modes trigger sooner
  const s = preferNice ? Math.min(1, stability * 2) : stability;
  // BAND_BOUNDARIES = [1.00, 0.85, 0.74, 0.64, 0.53, 0.43, 0.32, 0.21, 0.11, 0.00]
  // Band i covers [BAND_BOUNDARIES[i+1], BAND_BOUNDARIES[i])
  for (let i = 0; i < STABILITY_BANDS.length; i++) {
    if (s >= BAND_BOUNDARIES[i + 1]) return i;
  }
  return STABILITY_BANDS.length - 1;
}

/* ------------------------------------------------------------------ */
/*  Select mode with hysteresis                                        */
/* ------------------------------------------------------------------ */

/**
 * Select the current mode from net_stability, applying hysteresis to
 * prevent rapid flickering between adjacent modes.
 *
 * The mode only changes if stability crosses a band boundary by at least
 * `hysteresisMargin` (default 0.03). Hovering within the margin keeps
 * the current mode locked.
 */
export function selectMode(
  stability: number,
  currentMode: ModeDefinition | null,
  hysteresisMargin: number = DEFAULT_HYSTERESIS_MARGIN,
  preferNiceModes?: boolean,
): ModeDefinition {
  if (!currentMode) return STABILITY_BANDS[rawBandIndex(stability, preferNiceModes)];

  const currentIndex = STABILITY_BANDS.indexOf(currentMode);
  if (currentIndex === -1) return STABILITY_BANDS[rawBandIndex(stability, preferNiceModes)];

  // Apply the same boost for hysteresis comparisons
  const s = preferNiceModes ? Math.min(1, stability * 2) : stability;

  // Current band boundaries
  const upperBound = BAND_BOUNDARIES[currentIndex];
  const lowerBound = BAND_BOUNDARIES[currentIndex + 1];

  // Check if stability has crossed into an adjacent band with margin
  if (currentIndex > 0 && s > upperBound + hysteresisMargin) {
    // Moving to higher stability band
    return STABILITY_BANDS[rawBandIndex(stability, preferNiceModes)];
  }
  if (currentIndex < STABILITY_BANDS.length - 1 && s < lowerBound - hysteresisMargin) {
    // Moving to lower stability band
    return STABILITY_BANDS[rawBandIndex(stability, preferNiceModes)];
  }

  // Stay in current mode
  return currentMode;
}
