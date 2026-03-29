/**
 * Pure functions for computing global simulation metrics from detection data.
 *
 * These metrics feed the stability calculator (§4.2) and bass layer (§9).
 * All functions are stateless — pass in data, get metrics out.
 */

import type { ReadbackData, DetectionFrame, OrganelleState, OrganismState } from "../detection";
import type { OrganismPrediction } from "../organism-prediction";
import type { GlobalMetrics, SimEvent } from "./types";

/* ------------------------------------------------------------------ */
/*  Global Metrics                                                     */
/* ------------------------------------------------------------------ */

/**
 * Compute global metrics from a detection frame and raw readback data.
 *
 * @param data       - Raw particle positions/velocities from GPU readback
 * @param frame      - Current detection frame (organelles, organisms, ledger)
 * @param typeCounts - Total particle count per typeId (from simulation config)
 * @param prevFrame  - Previous detection frame for event diffing (null on first tick)
 */
export function computeGlobalMetrics(
  data: ReadbackData,
  frame: DetectionFrame,
  typeCounts: ReadonlyMap<number, number>,
  prevFrame: DetectionFrame | null,
  prediction: OrganismPrediction | null,
  typeKeys: ReadonlyArray<string>,
): GlobalMetrics {
  // ── Free particles ──────────────────────────────────────────────────
  const boundByType = new Map<number, number>();
  for (const org of frame.organelles) {
    const prev = boundByType.get(org.typeId) ?? 0;
    boundByType.set(org.typeId, prev + org.particleIndices.length);
  }

  let totalBound = 0;
  for (const count of boundByType.values()) totalBound += count;
  const freeParticleCount = data.n - totalBound;

  const freeParticlePercentByType = new Map<number, number>();
  for (const [typeId, total] of typeCounts) {
    const bound = boundByType.get(typeId) ?? 0;
    freeParticlePercentByType.set(typeId, total > 0 ? (total - bound) / total : 1);
  }

  // ── Average velocity ────────────────────────────────────────────────
  let velSum = 0;
  for (let i = 0; i < data.n; i++) {
    const vx = data.velX[i];
    const vy = data.velY[i];
    velSum += Math.sqrt(vx * vx + vy * vy);
  }
  const avgVelocity = data.n > 0 ? velSum / data.n : 0;

  // ── Average organelle density ───────────────────────────────────────
  let densitySum = 0;
  for (const org of frame.organelles) {
    const area = organelleArea(org, data.cellSize);
    densitySum += area > 0 ? org.particleIndices.length / area : 0;
  }
  const avgOrganelleDensity = frame.organelles.length > 0
    ? densitySum / frame.organelles.length
    : 0;

  // ── Species & organism counts ───────────────────────────────────────
  const speciesSet = new Set<string>();
  for (const osm of frame.organisms) speciesSet.add(osm.colorSignature);
  const speciesCount = speciesSet.size;
  const organismCount = frame.organisms.length;

  // ── Spatial entropy (§2.6, §6.2) ────────────────────────────────────
  // Use a coarse grid (8×8) to measure how evenly particles are distributed.
  // Shannon entropy normalized to [0,1]: 1 = uniform, 0 = all in one cell.
  const ENTROPY_GRID = 8;
  const spatialEntropy = computeSpatialEntropy(data, ENTROPY_GRID);

  // ── Organism fulfillment ─────────────────────────────────────────────
  // What fraction of organism-capable particles are actually in organisms?
  const organismFulfillment = computeOrganismFulfillment(
    frame, typeCounts, prediction, typeKeys,
  );

  // ── Events ──────────────────────────────────────────────────────────
  const events = prevFrame ? diffEvents(prevFrame, frame) : [];

  return {
    freeParticleCount,
    freeParticlePercentByType,
    avgVelocity,
    avgOrganelleDensity,
    speciesCount,
    organismCount,
    spatialEntropy,
    events,
    organismFulfillment,
  };
}

/* ------------------------------------------------------------------ */
/*  Organism fulfillment                                               */
/* ------------------------------------------------------------------ */

/**
 * Fraction of organism-capable particles that are currently bound in organisms.
 * Returns 0 when no organism-capable types exist or none are in organisms.
 */
function computeOrganismFulfillment(
  frame: DetectionFrame,
  typeCounts: ReadonlyMap<number, number>,
  prediction: OrganismPrediction | null,
  typeKeys: ReadonlyArray<string>,
): number {
  if (!prediction || prediction.organisms.length === 0) return 0;

  // Collect all typeIds that participate in any predicted organism
  const capableTypeIds = new Set<number>();
  for (const org of prediction.organisms) {
    for (const key of org.typeKeys) {
      const idx = typeKeys.indexOf(key);
      if (idx >= 0) capableTypeIds.add(idx);
    }
  }

  if (capableTypeIds.size === 0) return 0;

  // Total particles of capable types
  let totalCapable = 0;
  for (const typeId of capableTypeIds) {
    totalCapable += typeCounts.get(typeId) ?? 0;
  }
  if (totalCapable === 0) return 0;

  // Build organelle lookup: organelleId → OrganelleState
  const organelleMap = new Map<number, OrganelleState>();
  for (const org of frame.organelles) {
    organelleMap.set(org.id, org);
  }

  // Count particles that are in organisms (not just organelles)
  let inOrganism = 0;
  for (const osm of frame.organisms) {
    for (const orgId of osm.organelleIds) {
      const organelle = organelleMap.get(orgId);
      if (organelle && capableTypeIds.has(organelle.typeId)) {
        inOrganism += organelle.particleIndices.length;
      }
    }
  }

  return Math.min(inOrganism / totalCapable, 1);
}

/* ------------------------------------------------------------------ */
/*  Spatial entropy                                                    */
/* ------------------------------------------------------------------ */

/** Shannon entropy of particle positions on a coarse grid, normalized to [0,1]. */
function computeSpatialEntropy(data: ReadbackData, gridSize: number): number {
  if (data.n === 0) return 1;
  const cellW = data.width / gridSize;
  const cellH = data.height / gridSize;
  const counts = new Uint32Array(gridSize * gridSize);
  for (let i = 0; i < data.n; i++) {
    const col = Math.min(Math.floor(data.posX[i] / cellW), gridSize - 1);
    const row = Math.min(Math.floor(data.posY[i] / cellH), gridSize - 1);
    counts[row * gridSize + col]++;
  }
  let entropy = 0;
  const n = data.n;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i] === 0) continue;
    const p = counts[i] / n;
    entropy -= p * Math.log(p);
  }
  const maxEntropy = Math.log(gridSize * gridSize);
  return maxEntropy > 0 ? entropy / maxEntropy : 1;
}

/* ------------------------------------------------------------------ */
/*  Organelle area helper                                              */
/* ------------------------------------------------------------------ */

function organelleArea(org: OrganelleState, cellSize: number): number {
  const w = (org.maxCol - org.minCol + 1) * cellSize;
  const h = (org.maxRow - org.minRow + 1) * cellSize;
  return w * h;
}

/* ------------------------------------------------------------------ */
/*  Event diffing                                                      */
/* ------------------------------------------------------------------ */

function diffEvents(
  prev: DetectionFrame,
  curr: DetectionFrame,
): SimEvent[] {
  const events: SimEvent[] = [];

  // Organelle formation/dissolution — compare by id sets per type
  const prevOrganelleIds = new Set(prev.organelles.map(o => o.id));
  const currOrganelleIds = new Set(curr.organelles.map(o => o.id));

  for (const org of curr.organelles) {
    if (!prevOrganelleIds.has(org.id)) {
      events.push({ kind: "organelle-formed", id: org.id, typeId: org.typeId });
    }
  }
  for (const org of prev.organelles) {
    if (!currOrganelleIds.has(org.id)) {
      events.push({ kind: "organelle-dissolved", id: org.id, typeId: org.typeId });
    }
  }

  // Organism formation/dissolution — compare by id
  const prevOrganismIds = new Set(prev.organisms.map(o => o.id));
  const currOrganismIds = new Set(curr.organisms.map(o => o.id));

  for (const osm of curr.organisms) {
    if (!prevOrganismIds.has(osm.id)) {
      events.push({ kind: "organism-formed", id: osm.id, signature: osm.colorSignature });
    }
  }
  for (const osm of prev.organisms) {
    if (!currOrganismIds.has(osm.id)) {
      events.push({ kind: "organism-dissolved", id: osm.id, signature: osm.colorSignature });
    }
  }

  // Organelle joining/leaving organisms — compare organelle membership
  const prevMembership = buildOrganelleMembership(prev.organisms);
  const currMembership = buildOrganelleMembership(curr.organisms);

  for (const [organelleId, currOsmId] of currMembership) {
    const prevOsmId = prevMembership.get(organelleId);
    if (prevOsmId === undefined) {
      events.push({ kind: "organelle-joined", id: organelleId });
    }
  }
  for (const [organelleId, prevOsmId] of prevMembership) {
    if (!currMembership.has(organelleId)) {
      events.push({ kind: "organelle-left", id: organelleId });
    }
  }

  return events;
}

/** Map organelleId → organismId for membership tracking. */
function buildOrganelleMembership(
  organisms: ReadonlyArray<OrganismState>,
): Map<number, number> {
  const map = new Map<number, number>();
  for (const osm of organisms) {
    for (const orgId of osm.organelleIds) {
      map.set(orgId, osm.id);
    }
  }
  return map;
}
