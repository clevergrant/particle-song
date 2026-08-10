/**
 * Organism-driven bar cycle.
 *
 * Each living organism gets one bar in a repeating cycle.
 * Round-robins through species, visiting each once per cycle.
 *
 * Harmony model: the root is STICKY — it moves only when a full cycle
 * completes (all species visited) AND the key has been held for at
 * least MIN_ADVANCES_PER_KEY fresh bars, keeping a stable tonal center
 * without ever freezing (a single-species world completes a cycle every
 * bar, so the bar floor — not a species count — is what prevents
 * per-bar modulation). The per-bar structural difference between
 * consecutive organisms instead selects a diatonic chord degree within
 * the held key (I, V, IV, vi, …) — moment-to-moment behavior becomes a
 * chord progression rather than a modulation.
 */

import type { SnapshotOrganism } from "./types";
import {
  extractTypeAdjacency,
  jaccardDistance,
  structureDiffToInterval,
  structureDiffToCoFSteps,
} from "./structure-fingerprint";
import type { OrganelleTreeNode } from "../detection";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export type TypeAdjacencyGraph = ReadonlySet<number>;

export type RootStrategy = "table" | "circle-of-fifths";

export interface OrganismCycle {
  /** Ordered list of organism registryIds (sorted by creationTime). */
  readonly organismOrder: readonly number[];
  /** Species signatures visited so far this cycle. */
  readonly visitedSpecies: ReadonlySet<string>;
  /** Current position in the organism order. */
  readonly currentIndex: number;
  /** Accumulated root in semitones (mod 12). */
  readonly accumulatedRoot: number;
  /** Fingerprint of the previous bar's organism (for diff). */
  readonly prevFingerprint: TypeAdjacencyGraph | null;
  /** Increments each time a full cycle completes. */
  readonly cycleNumber: number;
  /** Fresh-bar advances since the root last moved (min-hold enforcement). */
  readonly advancesSinceRootChange: number;
}

/* ------------------------------------------------------------------ */
/*  Initial state                                                      */
/* ------------------------------------------------------------------ */

export function createInitialCycle(): OrganismCycle {
  return {
    organismOrder: [],
    visitedSpecies: new Set(),
    currentIndex: 0,
    accumulatedRoot: 0,
    prevFingerprint: null,
    cycleNumber: 0,
    advancesSinceRootChange: 0,
  };
}

/* ------------------------------------------------------------------ */
/*  Build / rebuild cycle when organisms change                        */
/* ------------------------------------------------------------------ */

/**
 * Build or rebuild the organism order from the current organisms.
 * Sorted by creationTime (oldest first) for deterministic ordering.
 * Preserves accumulatedRoot and prevFingerprint across rebuilds.
 */
export function buildOrganismCycle(
  organisms: readonly SnapshotOrganism[],
  prevCycle: OrganismCycle | null,
): OrganismCycle {
  const sorted = [...organisms].sort((a, b) => a.creationTime - b.creationTime);
  const order = sorted.map(o => o.registryId);

  if (!prevCycle) {
    return {
      organismOrder: order,
      visitedSpecies: new Set(),
      currentIndex: 0,
      accumulatedRoot: 0,
      prevFingerprint: null,
      cycleNumber: 0,
      advancesSinceRootChange: 0,
    };
  }

  // Check if organism set changed
  const prevSet = new Set(prevCycle.organismOrder);
  const newSet = new Set(order);
  const same = prevSet.size === newSet.size && [...prevSet].every(id => newSet.has(id));

  if (same) return prevCycle;

  // Rebuild but preserve harmonic state
  return {
    organismOrder: order,
    visitedSpecies: prevCycle.visitedSpecies,
    currentIndex: 0,
    accumulatedRoot: prevCycle.accumulatedRoot,
    prevFingerprint: prevCycle.prevFingerprint,
    cycleNumber: prevCycle.cycleNumber,
    advancesSinceRootChange: prevCycle.advancesSinceRootChange,
  };
}

/* ------------------------------------------------------------------ */
/*  Structural distance → diatonic chord degree                        */
/* ------------------------------------------------------------------ */

/** Distance bands → scale degree index (0 = I, 4 = V, 3 = IV, 5 = vi …).
 *  Small differences map to strong functional moves; large differences
 *  drift to color chords. */
const DEGREE_TABLE: ReadonlyArray<readonly [number, number]> = [
  [0.0, 0],   // I
  [0.01, 4],  // V
  [0.15, 3],  // IV
  [0.30, 5],  // vi
  [0.45, 1],  // ii
  [0.60, 2],  // iii
  [0.75, 6],  // vii
];

function structureDiffToDegree(distance: number): number {
  for (let i = DEGREE_TABLE.length - 1; i >= 0; i--) {
    if (distance >= DEGREE_TABLE[i][0]) return DEGREE_TABLE[i][1];
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  Advance cycle                                                      */
/* ------------------------------------------------------------------ */

/** Minimum fresh-bar advances a key is held before the root may move.
 *  Each advance spans `barsPerMelody` musical bars, so the audible hold
 *  is several times longer than this count. */
const MIN_ADVANCES_PER_KEY = 4;

export interface AdvanceResult {
  readonly cycle: OrganismCycle;
  readonly organism: SnapshotOrganism | null;
  /** Diatonic degree for this bar's chord (0 = I). */
  readonly chordDegreeIndex: number;
}

/**
 * Find the next unvisited organism (by species) and advance the cycle.
 * Returns the updated cycle, the selected organism (or null if none),
 * and this bar's chord degree.
 *
 * The root moves only when a cycle completes; within a cycle the per-bar
 * structural distance selects a chord degree instead.
 *
 * @param eligibleSpecies - When provided, only these species may be
 *   selected (e.g. species with at least one organism old enough to
 *   qualify). If none are eligible, the cycle is returned unchanged with
 *   a null organism so silent bars don't burn cycle progress.
 */
export function advanceCycle(
  cycle: OrganismCycle,
  organisms: readonly SnapshotOrganism[],
  organismTrees: ReadonlyMap<number, OrganelleTreeNode>,
  strategy: RootStrategy,
  eligibleSpecies?: ReadonlySet<string> | null,
): AdvanceResult {
  if (organisms.length === 0) {
    return { cycle, organism: null, chordDegreeIndex: 0 };
  }

  const isEligible = (sig: string): boolean =>
    eligibleSpecies == null || eligibleSpecies.has(sig);

  const orgById = new Map<number, SnapshotOrganism>();
  for (const org of organisms) orgById.set(org.registryId, org);

  // Collect all unique eligible species currently alive
  const allSpecies = new Set<string>();
  for (const org of organisms) {
    if (isEligible(org.colorSignature)) allSpecies.add(org.colorSignature);
  }
  if (allSpecies.size === 0) {
    return { cycle, organism: null, chordDegreeIndex: 0 };
  }

  // Check if all eligible species have been visited — if so, reset
  let visited = cycle.visitedSpecies;
  let cycleNumber = cycle.cycleNumber;
  let completedCycle = false;

  const allVisited = [...allSpecies].every(s => visited.has(s));
  if (allVisited) {
    visited = new Set();
    cycleNumber++;
    completedCycle = true;
  }

  // Scan for next unvisited eligible species
  for (let i = 0; i < cycle.organismOrder.length; i++) {
    const idx = (cycle.currentIndex + i) % cycle.organismOrder.length;
    const regId = cycle.organismOrder[idx];
    const org = orgById.get(regId);
    if (!org) continue;
    if (!isEligible(org.colorSignature)) continue;
    if (visited.has(org.colorSignature)) continue;

    const tree = organismTrees.get(regId);
    const fingerprint = tree ? extractTypeAdjacency(tree) : new Set<number>();

    const distance = cycle.prevFingerprint !== null
      ? jaccardDistance(cycle.prevFingerprint, fingerprint)
      : 0;

    // Per-bar: chord degree within the held key
    const chordDegreeIndex = structureDiffToDegree(distance);

    // Root moves only at cycle boundaries — sticky tonal center — and
    // only after the key has been held MIN_ADVANCES_PER_KEY fresh bars
    // (a single-species world completes a cycle every bar, which would
    // otherwise degenerate into per-bar modulation).
    const advances = cycle.advancesSinceRootChange + 1;
    let newRoot = cycle.accumulatedRoot;
    let rootMoved = false;
    if (
      completedCycle &&
      cycle.prevFingerprint !== null &&
      advances >= MIN_ADVANCES_PER_KEY
    ) {
      let interval = strategy === "circle-of-fifths"
        ? structureDiffToCoFSteps(distance)
        : structureDiffToInterval(distance);
      // A structurally static ecosystem maps to unison — which would
      // freeze the root forever. Walk a fifth instead so the key still
      // breathes even when the organisms don't change.
      if (interval === 0) interval = 7;
      newRoot = ((cycle.accumulatedRoot + interval) % 12 + 12) % 12;
      rootMoved = true;
    }

    const newVisited = new Set(visited);
    newVisited.add(org.colorSignature);

    return {
      cycle: {
        organismOrder: cycle.organismOrder,
        visitedSpecies: newVisited,
        currentIndex: (idx + 1) % cycle.organismOrder.length,
        accumulatedRoot: newRoot,
        prevFingerprint: fingerprint,
        cycleNumber,
        advancesSinceRootChange: rootMoved ? 0 : advances,
      },
      organism: org,
      chordDegreeIndex,
    };
  }

  // No unvisited eligible species found in the order (stale order) —
  // return unchanged rather than forcing an ineligible pick.
  return { cycle, organism: null, chordDegreeIndex: 0 };
}
