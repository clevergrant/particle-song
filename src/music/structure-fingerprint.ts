/**
 * Structure fingerprint: extract a canonical type-adjacency graph from an
 * organism's organelle tree and compute distances between organisms.
 */

import type { OrganelleTreeNode } from "../detection";

/* ------------------------------------------------------------------ */
/*  Type-adjacency graph                                               */
/* ------------------------------------------------------------------ */

/** Canonical edge key — order-independent pair encoding. */
function edgeKey(a: number, b: number): number {
  return Math.min(a, b) * 256 + Math.max(a, b);
}

/**
 * Walk the OrganelleTreeNode MST and collect every cross-type edge
 * as an encoded pair. Same-type edges are ignored (they represent
 * intra-type clustering, not structural bonds).
 *
 * Each present type also contributes a self-edge (t,t) marker, so an
 * organism's type composition is part of its fingerprint. Without this,
 * every single-type organism has an EMPTY edge set — all such organisms
 * compare as identical (distance 0), which freezes the root and pins
 * every chord to degree I.
 */
export function extractTypeAdjacency(
  tree: OrganelleTreeNode,
): ReadonlySet<number> {
  const edges = new Set<number>();

  function walk(node: OrganelleTreeNode, parentTypeId: number | null): void {
    edges.add(edgeKey(node.typeId, node.typeId));
    if (parentTypeId !== null && node.typeId !== parentTypeId) {
      edges.add(edgeKey(parentTypeId, node.typeId));
    }
    for (const child of node.children) {
      walk(child, node.typeId);
    }
  }

  walk(tree, null);
  return edges;
}

/* ------------------------------------------------------------------ */
/*  Jaccard distance                                                   */
/* ------------------------------------------------------------------ */

/**
 * Jaccard distance between two type-adjacency graphs.
 * Returns 0 when identical (or both empty), 1 when fully disjoint.
 */
export function jaccardDistance(
  a: ReadonlySet<number>,
  b: ReadonlySet<number>,
): number {
  if (a.size === 0 && b.size === 0) return 0;

  let intersection = 0;
  const smaller = a.size <= b.size ? a : b;
  const larger = a.size <= b.size ? b : a;
  for (const edge of smaller) {
    if (larger.has(edge)) intersection++;
  }

  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : 1 - intersection / union;
}

/* ------------------------------------------------------------------ */
/*  Distance → interval mapping                                        */
/* ------------------------------------------------------------------ */

/** Interval table: Jaccard distance bands → semitone intervals. */
const INTERVAL_TABLE: ReadonlyArray<readonly [number, number]> = [
  [0.0, 0],   // unison
  [0.01, 7],  // perfect fifth
  [0.15, 5],  // perfect fourth
  [0.30, 4],  // major third
  [0.45, 3],  // minor third
  [0.60, 9],  // major sixth
  [0.75, 2],  // major second
  [0.90, 6],  // tritone
];

/**
 * Strategy A: map structural distance to a fixed harmonic-series interval.
 */
export function structureDiffToInterval(distance: number): number {
  for (let i = INTERVAL_TABLE.length - 1; i >= 0; i--) {
    if (distance >= INTERVAL_TABLE[i][0]) return INTERVAL_TABLE[i][1];
  }
  return 0;
}

/**
 * Strategy B: walk the circle of fifths by a number of steps proportional
 * to the structural distance. 0 → 0 steps (unison), 1 → 6 steps.
 * Returns the semitone interval (each CoF step = 7 semitones).
 */
export function structureDiffToCoFSteps(distance: number): number {
  const steps = Math.round(distance * 6);
  return (steps * 7) % 12;
}
