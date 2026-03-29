import type { ForceMatrix } from "./particles/particle";

export interface PredictedOrganism {
  readonly typeKeys: ReadonlyArray<string>;
  readonly signature: string;
  readonly stabilityScore: number;
}

export interface OrganismPrediction {
  readonly viableTypes: ReadonlyArray<string>;
  readonly organisms: ReadonlyArray<PredictedOrganism>;
}

const MAX_ORGANISMS = 50;
const MIN_PARTICLES_FOR_VIABILITY = 3;

// No self-force filter — detection.ts doesn't require self-attraction
// for a type to participate in organisms. The scoring function accounts
// for self-force as a continuous factor rather than a hard cutoff.

// Cross-type pairs need meaningful net mutual attraction to bind.
// Barely-positive pairs won't overcome particle velocity / damping.
const MUTUAL_ATTRACTION_THRESHOLD = 0.15;

// If one direction is strongly repulsive, the pair won't bind
const ONE_SIDED_REPULSION_CUTOFF = -0.35;

// Predictions scoring below this fraction of the top score are pruned
const RELATIVE_SCORE_CUTOFF = 0.2;

// Maximum sub-clique size to enumerate from each maximal clique
const MAX_SUB_CLIQUE_SIZE = 8;

/**
 * Predict organism species from the force matrix and particle counts.
 * Pure function — no side effects.
 *
 * Sign convention: positive force values = attraction, negative = repulsion.
 * Organelles form via spatial clustering (proximity + coherence), not just
 * self-attraction. Organisms form when organelles of different types attract.
 *
 * Mirrors detection.ts logic: organisms require net positive cross-type
 * affinity weighted by particle counts.
 */
export function predictOrganisms(
  forceMatrix: ForceMatrix,
  typeKeys: ReadonlyArray<string>,
  typeCounts: Readonly<Record<string, number>>,
): OrganismPrediction {
  // Step 1: Find viable types — must have enough particles and not
  // be strongly self-repulsive (which prevents organelle formation)
  const viableTypes = typeKeys.filter((t) => {
    const count = typeCounts[t] ?? 0;
    return count >= MIN_PARTICLES_FOR_VIABILITY;
  });

  // Step 2: Build compatibility graph with weighted edges
  // Mirrors detection.ts: affinity = (ab + ba) * aCount * bCount > 0
  const adjacency = new Map<string, Set<string>>();
  const edgeWeight = new Map<string, number>();
  for (const t of viableTypes) adjacency.set(t, new Set());

  for (let i = 0; i < viableTypes.length; i++) {
    const a = viableTypes[i];
    for (let j = i + 1; j < viableTypes.length; j++) {
      const b = viableTypes[j];
      const ab = forceMatrix[a]?.[b] ?? 0;
      const ba = forceMatrix[b]?.[a] ?? 0;
      const mutual = ab + ba;

      if (mutual <= MUTUAL_ATTRACTION_THRESHOLD) continue;
      if (ab < ONE_SIDED_REPULSION_CUTOFF || ba < ONE_SIDED_REPULSION_CUTOFF) continue;

      adjacency.get(a)!.add(b);
      adjacency.get(b)!.add(a);

      // Weight mirrors detection: mutual attraction × particle abundance
      const aCount = typeCounts[a] ?? 0;
      const bCount = typeCounts[b] ?? 0;
      const abundanceWeight = Math.sqrt(aCount * bCount);
      const asymmetry = Math.abs(ab - ba) / (Math.abs(ab) + Math.abs(ba) + 0.01);
      const symmetryBonus = 1 - asymmetry * 0.3; // 0.7..1.0
      edgeWeight.set(`${a},${b}`, mutual * abundanceWeight * symmetryBonus);
    }
  }

  // Step 3: Enumerate maximal cliques (Bron-Kerbosch with pivoting)
  const maximalCliques: string[][] = [];

  function bronKerbosch(R: Set<string>, P: Set<string>, X: Set<string>) {
    if (maximalCliques.length >= 100) return; // generous limit for clique finding
    if (P.size === 0 && X.size === 0) {
      if (R.size >= 2) maximalCliques.push([...R]);
      return;
    }
    let pivot = "";
    let pivotDeg = -1;
    for (const u of P) {
      const deg = adjacency.get(u)?.size ?? 0;
      if (deg > pivotDeg) { pivot = u; pivotDeg = deg; }
    }
    for (const u of X) {
      const deg = adjacency.get(u)?.size ?? 0;
      if (deg > pivotDeg) { pivot = u; pivotDeg = deg; }
    }

    const pivotNeighbors = adjacency.get(pivot) ?? new Set();
    for (const v of [...P]) {
      if (pivotNeighbors.has(v)) continue;
      const neighbors = adjacency.get(v) ?? new Set();
      const newR = new Set(R);
      newR.add(v);
      const newP = new Set<string>();
      for (const w of P) if (neighbors.has(w)) newP.add(w);
      const newX = new Set<string>();
      for (const w of X) if (neighbors.has(w)) newX.add(w);
      bronKerbosch(newR, newP, newX);
      P.delete(v);
      X.add(v);
    }
  }

  bronKerbosch(new Set(), new Set(viableTypes), new Set());

  // Step 4: Generate all connected sub-cliques of size >= 2 from maximal cliques.
  // Real organisms form as subsets — a 5-type clique implies valid 2,3,4-type organisms too.
  const seen = new Set<string>();
  const organisms: PredictedOrganism[] = [];

  for (const clique of maximalCliques) {
    const sorted = [...clique].sort();
    const limit = Math.min(sorted.length, MAX_SUB_CLIQUE_SIZE);

    // Enumerate subsets of size 2..limit
    const subsets = enumerateSubsets(sorted, 2, limit);

    for (const subset of subsets) {
      if (organisms.length >= MAX_ORGANISMS) break;

      const indices = subset
        .map((k) => typeKeys.indexOf(k))
        .sort((a, b) => a - b);
      const sig = indices.join("+");

      if (seen.has(sig)) continue;
      seen.add(sig);

      const score = scoreOrganism(subset, forceMatrix, typeCounts);
      if (score <= 0) continue;

      organisms.push({
        typeKeys: indices.map((i) => typeKeys[i]),
        signature: sig,
        stabilityScore: score,
      });
    }
  }

  organisms.sort((a, b) => b.stabilityScore - a.stabilityScore);

  // Prune predictions that score far below the top — they're noise
  if (organisms.length > 0) {
    const topScore = organisms[0].stabilityScore;
    const cutoff = topScore * RELATIVE_SCORE_CUTOFF;
    const pruned = organisms.filter((o) => o.stabilityScore >= cutoff);
    organisms.length = 0;
    organisms.push(...pruned);
  }

  if (organisms.length > MAX_ORGANISMS) organisms.length = MAX_ORGANISMS;

  return { viableTypes, organisms };
}

/** Score an organism candidate. Higher = more likely to form. */
function scoreOrganism(
  types: ReadonlyArray<string>,
  forceMatrix: ForceMatrix,
  typeCounts: Readonly<Record<string, number>>,
): number {
  // 1. Self-cohesion: positive self-attraction helps, negative hurts (but mildly)
  let selfCohesion = 0;
  for (const t of types) {
    const selfForce = forceMatrix[t]?.[t] ?? 0;
    // Clamp contribution: strong self-attraction helps a lot,
    // mild self-repulsion only hurts a little (shader repulsion compensates)
    selfCohesion += Math.max(selfForce, selfForce * 0.3);
  }

  // 2. Cross-type binding: mirrors detection affinity calculation
  let crossBinding = 0;
  let pairCount = 0;
  for (let i = 0; i < types.length; i++) {
    for (let j = i + 1; j < types.length; j++) {
      const ab = forceMatrix[types[i]]?.[types[j]] ?? 0;
      const ba = forceMatrix[types[j]]?.[types[i]] ?? 0;
      const aCount = typeCounts[types[i]] ?? 0;
      const bCount = typeCounts[types[j]] ?? 0;
      // Match detection.ts: affinity = (ab + ba) * aCount * bCount
      crossBinding += (ab + ba) * Math.sqrt(aCount * bCount);
      pairCount++;
    }
  }

  if (crossBinding <= 0) return 0; // net repulsive — won't form

  // 3. Abundance: geometric mean of particle counts
  let countProduct = 1;
  for (const t of types) countProduct *= Math.max(1, typeCounts[t] ?? 0);
  const abundanceFactor = Math.pow(countProduct, 1 / types.length);

  // 4. Size bonus: larger organisms score slightly higher when binding is strong
  const sizeBonus = 1 + (types.length - 2) * 0.1;

  return (selfCohesion + crossBinding) * Math.log2(abundanceFactor + 1) * sizeBonus;
}

/** Enumerate all subsets of `items` with size in [minSize, maxSize]. */
function enumerateSubsets(
  items: ReadonlyArray<string>,
  minSize: number,
  maxSize: number,
): string[][] {
  const result: string[][] = [];
  const n = items.length;

  // For the full set, include it directly
  if (n >= minSize && n <= maxSize) {
    result.push([...items]);
  }

  // For subsets smaller than the full set
  if (maxSize >= n) maxSize = n - 1; // full set already added

  function enumerate(start: number, current: string[]) {
    if (current.length >= minSize && current.length <= maxSize) {
      result.push([...current]);
    }
    if (current.length >= maxSize) return;
    for (let i = start; i < n; i++) {
      current.push(items[i]);
      enumerate(i + 1, current);
      current.pop();
    }
  }

  if (maxSize >= minSize) {
    enumerate(0, []);
  }

  return result;
}
