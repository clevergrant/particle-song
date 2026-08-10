/**
 * Supremacy tracker: detect when one species dominates for too long
 * and trigger auto-randomization.
 */

import type { SnapshotOrganism } from "./types";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface SupremacyState {
  /** Species signature that currently leads (most organisms). */
  readonly leadSpecies: string | null;
  /** Number of consecutive completed cycles the leader has held. */
  readonly consecutiveCycles: number;
  /** Bars elapsed since last randomize (for fallback timer). */
  readonly barsSinceRandomize: number;
}

export interface SupremacyConfig {
  /** Number of full organism cycles before supremacy triggers randomize. */
  readonly cyclesBeforeRandomize: number;
  /** Fallback bar count: randomize if no species dominates for a full cycle. */
  readonly fallbackBars: number;
  /** Whether the fallback timer is enabled. */
  readonly fallbackEnabled: boolean;
}

/* ------------------------------------------------------------------ */
/*  Initial state                                                      */
/* ------------------------------------------------------------------ */

export function createInitialSupremacy(): SupremacyState {
  return {
    leadSpecies: null,
    consecutiveCycles: 0,
    barsSinceRandomize: 0,
  };
}

/* ------------------------------------------------------------------ */
/*  Update on each bar                                                 */
/* ------------------------------------------------------------------ */

/**
 * Update supremacy state. Call once per bar.
 *
 * @param cycleCompleted - true when the organism cycle just wrapped around
 */
export function updateSupremacy(
  state: SupremacyState,
  organisms: readonly SnapshotOrganism[],
  cycleCompleted: boolean,
): SupremacyState {
  const barsSinceRandomize = state.barsSinceRandomize + 1;

  if (!cycleCompleted) {
    return { ...state, barsSinceRandomize };
  }

  // Count organisms per species
  const counts = new Map<string, number>();
  for (const org of organisms) {
    counts.set(org.colorSignature, (counts.get(org.colorSignature) ?? 0) + 1);
  }

  // Find leader
  let leader: string | null = null;
  let maxCount = 0;
  for (const [sig, count] of counts) {
    if (count > maxCount) {
      maxCount = count;
      leader = sig;
    }
  }

  // Check if leader is the same as before
  if (leader !== null && leader === state.leadSpecies) {
    return {
      leadSpecies: leader,
      consecutiveCycles: state.consecutiveCycles + 1,
      barsSinceRandomize,
    };
  }

  // Leader changed — reset
  return {
    leadSpecies: leader,
    consecutiveCycles: leader !== null ? 1 : 0,
    barsSinceRandomize,
  };
}

/* ------------------------------------------------------------------ */
/*  Should randomize?                                                  */
/* ------------------------------------------------------------------ */

/**
 * Check if randomization should trigger.
 * Returns true if supremacy threshold met OR fallback timer expired.
 */
export function shouldRandomize(
  state: SupremacyState,
  config: SupremacyConfig,
): boolean {
  // Supremacy: one species has held for enough cycles
  if (state.consecutiveCycles >= config.cyclesBeforeRandomize) {
    return true;
  }

  // Fallback: no cycle completed in time
  if (config.fallbackEnabled && state.barsSinceRandomize >= config.fallbackBars) {
    return true;
  }

  return false;
}

/**
 * Reset supremacy state after a randomization.
 */
export function resetSupremacy(): SupremacyState {
  return createInitialSupremacy();
}
