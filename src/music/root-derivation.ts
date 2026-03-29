/**
 * Root key derivation from force matrix (§4.4).
 *
 * Maps particle types to positions on the circle of fifths based on
 * attraction/repulsion relationships. The species holding the oldest
 * living organism determines the active root.
 */

import type { ForceMatrix } from "../particles/particle";

/* ------------------------------------------------------------------ */
/*  Circle of fifths (semitones from C)                                */
/* ------------------------------------------------------------------ */

/** Circle-of-fifths positions in semitone order: C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F */
const CIRCLE_OF_FIFTHS: readonly number[] = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5];

/* ------------------------------------------------------------------ */
/*  Affinity-based type→root mapping                                   */
/* ------------------------------------------------------------------ */

/**
 * Compute a circle-of-fifths root assignment for each particle type.
 *
 * Strategy: sort types by average net affinity (most attractive first),
 * assign to circle-of-fifths positions in order. Types that attract
 * most end up close on the circle; repulsive types end up far apart.
 *
 * Returns a map: typeKey → MIDI root semitone (0–11).
 */
export function computeTypeRoots(
  forceMatrix: ForceMatrix,
  typeKeys: readonly string[],
): ReadonlyMap<string, number> {
  if (typeKeys.length === 0) return new Map();

  // Compute average net affinity per type (how attracted it is to others on average)
  const affinities: { key: string; avgAffinity: number }[] = typeKeys.map(key => {
    const row = forceMatrix[key];
    if (!row) return { key, avgAffinity: 0 };
    let sum = 0;
    let count = 0;
    for (const otherKey of typeKeys) {
      if (otherKey === key) continue;
      sum += row[otherKey] ?? 0;
      count++;
    }
    return { key, avgAffinity: count > 0 ? sum / count : 0 };
  });

  // Sort by affinity descending — most attractive types get adjacent CoF positions
  affinities.sort((a, b) => b.avgAffinity - a.avgAffinity);

  // Assign circle-of-fifths positions
  const result = new Map<string, number>();
  for (let i = 0; i < affinities.length; i++) {
    const cofIndex = i % CIRCLE_OF_FIFTHS.length;
    result.set(affinities[i].key, CIRCLE_OF_FIFTHS[cofIndex]);
  }

  return result;
}

/**
 * Determine the current root from organism data and type-root mapping.
 *
 * The oldest living organism's species (colorSignature) determines root.
 * The species' root is the root of its first (lowest-index) type.
 *
 * Returns a MIDI semitone (0–11), or a default if no organisms exist.
 */
export function deriveRoot(
  typeRoots: ReadonlyMap<string, number>,
  oldestSpeciesSignature: string | null,
  typeKeys: readonly string[],
  defaultRoot: number = 0,
): number {
  if (!oldestSpeciesSignature) return defaultRoot;

  // Parse species signature "0+1+3" → type indices
  const typeIndices = oldestSpeciesSignature.split("+").map(Number);
  if (typeIndices.length === 0) return defaultRoot;

  // Use the first type in the species as the root source
  const firstTypeKey = typeKeys[typeIndices[0]];
  if (!firstTypeKey) return defaultRoot;

  return typeRoots.get(firstTypeKey) ?? defaultRoot;
}

/**
 * Find the colorSignature of the oldest living organism.
 * "Oldest" = smallest creationTime (earliest formation).
 */
export function findOldestSpecies(
  organisms: readonly { readonly colorSignature: string; readonly creationTime: number }[],
): string | null {
  if (organisms.length === 0) return null;
  let oldest = organisms[0];
  for (let i = 1; i < organisms.length; i++) {
    if (organisms[i].creationTime < oldest.creationTime) {
      oldest = organisms[i];
    }
  }
  return oldest.colorSignature;
}
