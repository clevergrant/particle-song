/**
 * Play-time parameter computation for the tuplet grid architecture.
 *
 * At play time (when the scrubber reaches a slot), these functions
 * derive volume, pan, filter, envelope, gate, waveform, and vibrato
 * from the LIVE simulation state — not a stale bar-boundary snapshot.
 */

import type {
  SlotNote,
  EnvelopeParams,
  EnvelopeRanges,
  NoteDuration,
  WaveformParams,
} from "./types";
import type {
  OrganelleState,
  OrganismState,
  RegisteredOrganism,
  OrganismRegistry,
  DetectionFrame,
  OrganelleTreeNode,
} from "../detection";
import type { ForceMatrix } from "../particles/particle";
import { computeAllSociabilities, sociabilityToWaveform } from "./timbre";

/* ------------------------------------------------------------------ */
/*  Organelle physics (derived from live OrganelleState)               */
/* ------------------------------------------------------------------ */

/** Physics values needed for ranking — derived from live OrganelleState. */
export interface OrganellePhysics {
  readonly id: number;
  readonly particleCount: number;
  readonly centroidSpeed: number;
  readonly density: number;
  readonly spatialRadius: number;
}

/** Derive physics values from a live OrganelleState.
 *  Mirrors the computation in buildBarSnapshot(). */
export function deriveOrganellePhysics(
  org: OrganelleState,
  proximityRadius: number,
): OrganellePhysics {
  const speed = Math.sqrt(org.avgVelX * org.avgVelX + org.avgVelY * org.avgVelY);
  const w = (org.maxCol - org.minCol + 1) * proximityRadius;
  const h = (org.maxRow - org.minRow + 1) * proximityRadius;
  const area = w * h;
  const density = area > 0 ? org.particleIndices.length / area : 0;
  const spatialRadius = Math.max(w, h) / 2;
  return {
    id: org.id,
    particleCount: org.particleIndices.length,
    centroidSpeed: speed,
    density,
    spatialRadius,
  };
}

/* ------------------------------------------------------------------ */
/*  Frame-level index caches (rebuilt once per frame)                   */
/* ------------------------------------------------------------------ */

export interface FrameIndex {
  readonly registryById: ReadonlyMap<number, RegisteredOrganism>;
  readonly organelleById: ReadonlyMap<number, OrganelleState>;
  readonly physicsByOrganelle: ReadonlyMap<number, OrganellePhysics>;
  /** BFS-ordered organelle IDs per registered organism. */
  readonly bfsOrderByRegistryId: ReadonlyMap<number, readonly number[]>;
  /** Organisms grouped by species (colorSignature). */
  readonly organismsBySignature: ReadonlyMap<string, readonly RegisteredOrganism[]>;
}

/** Build lookup indexes from live detection state. Call once per frame. */
export function buildFrameIndex(
  registry: OrganismRegistry,
  frame: DetectionFrame,
  proximityRadius: number,
): FrameIndex {
  const registryById = new Map<number, RegisteredOrganism>();
  for (const ro of registry.organisms) {
    registryById.set(ro.registryId, ro);
  }

  const organelleById = new Map<number, OrganelleState>();
  const physicsByOrganelle = new Map<number, OrganellePhysics>();
  for (const org of frame.organelles) {
    organelleById.set(org.id, org);
    physicsByOrganelle.set(org.id, deriveOrganellePhysics(org, proximityRadius));
  }

  const bfsOrderByRegistryId = new Map<number, readonly number[]>();
  for (const ro of registry.organisms) {
    const order: number[] = [];
    const queue: OrganelleTreeNode[] = [ro.tree];
    while (queue.length > 0) {
      const node = queue.shift()!;
      order.push(node.organelleId);
      for (const child of node.children) queue.push(child);
    }
    bfsOrderByRegistryId.set(ro.registryId, order);
  }

  const organismsBySignature = new Map<string, RegisteredOrganism[]>();
  for (const ro of registry.organisms) {
    let group = organismsBySignature.get(ro.colorSignature);
    if (!group) { group = []; organismsBySignature.set(ro.colorSignature, group); }
    group.push(ro);
  }

  return { registryById, organelleById, physicsByOrganelle, bfsOrderByRegistryId, organismsBySignature };
}

/* ------------------------------------------------------------------ */
/*  Rank-based normalization (same logic as hit-scheduler)             */
/* ------------------------------------------------------------------ */

/** Pre-computed 0–1 normalized values for envelope-driving properties. */
export interface EnvelopeNorms {
  readonly countNorm: number;
  readonly speedNorm: number;
  readonly densityNorm: number;
  readonly radiusNorm: number;
  readonly ageNorm: number;
}

function normalizeRange(value: number, min: number, max: number): number {
  return max > min ? (value - min) / (max - min) : 0;
}

/** Compute rank-based norms for a set of organelle physics.
 *  Same algorithm as hit-scheduler's computeOrganelleRanks. */
export function computeOrganelleRanks(
  organelles: readonly OrganellePhysics[],
  ranges: EnvelopeRanges,
): ReadonlyMap<number, EnvelopeNorms> {
  const n = organelles.length;
  const result = new Map<number, EnvelopeNorms>();
  if (n === 0) return result;

  if (n === 1) {
    const o = organelles[0];
    result.set(o.id, {
      countNorm: normalizeRange(o.particleCount, ranges.particleCountMin, ranges.particleCountMax),
      speedNorm: normalizeRange(o.centroidSpeed, ranges.speedMin, ranges.speedMax),
      densityNorm: normalizeRange(o.density, ranges.densityMin, ranges.densityMax),
      radiusNorm: normalizeRange(o.spatialRadius, ranges.radiusMin, ranges.radiusMax),
      ageNorm: 0,
    });
    return result;
  }

  const byCount = [...organelles].sort((a, b) => a.particleCount - b.particleCount);
  const bySpeed = [...organelles].sort((a, b) => a.centroidSpeed - b.centroidSpeed);
  const byDensity = [...organelles].sort((a, b) => a.density - b.density);
  const byRadius = [...organelles].sort((a, b) => a.spatialRadius - b.spatialRadius);

  const countRank = new Map<number, number>();
  const speedRank = new Map<number, number>();
  const densityRank = new Map<number, number>();
  const radiusRank = new Map<number, number>();

  for (let i = 0; i < n; i++) {
    const r = i / (n - 1);
    countRank.set(byCount[i].id, r);
    speedRank.set(bySpeed[i].id, r);
    densityRank.set(byDensity[i].id, r);
    radiusRank.set(byRadius[i].id, r);
  }

  for (const o of organelles) {
    result.set(o.id, {
      countNorm: countRank.get(o.id)!,
      speedNorm: speedRank.get(o.id)!,
      densityNorm: densityRank.get(o.id)!,
      radiusNorm: radiusRank.get(o.id)!,
      ageNorm: 0,
    });
  }
  return result;
}

/* ------------------------------------------------------------------ */
/*  Expression mapping (same formulas as hit-scheduler)                */
/* ------------------------------------------------------------------ */

function normalizeCount(particleCount: number, minCount: number, maxCount: number): number {
  if (maxCount <= minCount) return 0.5;
  return (particleCount - minCount) / (maxCount - minCount);
}

export function densityToVolume(density: number): number {
  return Math.min(Math.max(density / 2, 0.15), 1);
}

export function countToFilterCutoff(particleCount: number, minCount: number, maxCount: number): number {
  const normalized = normalizeCount(particleCount, minCount, maxCount);
  return 8000 * Math.pow(200 / 8000, normalized);
}

export function computeEnvelope(norms: EnvelopeNorms): EnvelopeParams {
  const { countNorm, speedNorm, densityNorm, radiusNorm, ageNorm } = norms;
  const baseAttack = 0.02 + (1 - speedNorm) * 0.3 + countNorm * 0.2;
  const attackDuration = Math.max(0.01, baseAttack * ageNorm * 1.2);
  const peakLevel = (0.4 + speedNorm * 0.35 + densityNorm * 0.25) * (0.3 + 0.7 * ageNorm);
  const decayDuration = 0.1 + (1 - densityNorm) * 0.6;
  const sustainLevel = 0.1 + countNorm * 0.7;
  const releaseDuration = 0.1 + radiusNorm * 1.0;
  const attackCurve: EnvelopeParams["attackCurve"] = speedNorm > 0.6 ? "linear" : "ease-in";
  const decayCurve: EnvelopeParams["decayCurve"] = densityNorm > 0.5 ? "exponential" : "ease-out";
  const releaseCurve: EnvelopeParams["releaseCurve"] = "ease-out";
  return { attackDuration, attackCurve, peakLevel, decayDuration, decayCurve, sustainLevel, releaseDuration, releaseCurve };
}

const NOTE_DURATION_BEATS: Record<NoteDuration, number> = {
  whole: 4, half: 2, quarter: 1, eighth: 0.5, sixteenth: 0.25,
};

export function computeNoteDuration(norms: EnvelopeNorms): NoteDuration {
  const blend = 0.5 * norms.countNorm + 0.3 * norms.densityNorm + 0.2 * norms.ageNorm;
  if (blend > 0.85) return "whole";
  if (blend > 0.65) return "half";
  if (blend > 0.40) return "quarter";
  if (blend > 0.20) return "eighth";
  return "sixteenth";
}

export function noteDurationToSeconds(nd: NoteDuration, beatDur: number): number {
  return NOTE_DURATION_BEATS[nd] * beatDur;
}

/* ------------------------------------------------------------------ */
/*  Full playback params (computed at play time per note)              */
/* ------------------------------------------------------------------ */

/** Everything the AudioGraph needs to create a note's AudioNodes. */
export interface PlaybackParams {
  readonly volume: number;
  readonly pan: number;
  readonly filterCutoff: number;
  readonly envelope: EnvelopeParams;
  readonly gateDuration: number;
  readonly waveform: WaveformParams;
  readonly vibratoDepth: number;
}

/** Cached per-frame data that doesn't change between notes. */
export interface PlayTimeContext {
  readonly frameIndex: FrameIndex;
  readonly ranks: ReadonlyMap<number, EnvelopeNorms>;
  readonly envelopeRanges: EnvelopeRanges;
  readonly sociabilities: ReadonlyMap<string, number>;
  readonly typeKeys: readonly string[];
  readonly canvasWidth: number;
  readonly barDur: number;
  readonly barStartTime: number;
}

/** Build the per-frame context. Call once per frame (or lazily). */
export function buildPlayTimeContext(
  registry: OrganismRegistry,
  frame: DetectionFrame,
  proximityRadius: number,
  envelopeRanges: EnvelopeRanges,
  forceMatrix: ForceMatrix,
  typeKeys: readonly string[],
  canvasWidth: number,
  barDur: number,
  barStartTime: number,
): PlayTimeContext {
  const frameIndex = buildFrameIndex(registry, frame, proximityRadius);

  // Collect all playable organelle physics for ranking
  const allPhysics: OrganellePhysics[] = [];
  for (const ro of registry.organisms) {
    const bfsOrder = frameIndex.bfsOrderByRegistryId.get(ro.registryId);
    if (!bfsOrder) continue;
    for (const orgId of bfsOrder) {
      const p = frameIndex.physicsByOrganelle.get(orgId);
      if (p) allPhysics.push(p);
    }
  }

  const ranks = computeOrganelleRanks(allPhysics, envelopeRanges);
  const sociabilities = computeAllSociabilities(forceMatrix, typeKeys);

  return {
    frameIndex,
    ranks,
    envelopeRanges,
    sociabilities,
    typeKeys,
    canvasWidth,
    barDur,
    barStartTime,
  };
}

/** Compute playback parameters for a species-level SlotNote from live state.
 *  Aggregates properties across all organisms of the species.
 *  Returns null if no organisms of the species are alive. */
export function computePlaybackParams(
  note: SlotNote,
  ctx: PlayTimeContext,
): PlaybackParams | null {
  const { frameIndex, ranks, sociabilities, typeKeys, canvasWidth, barDur } = ctx;

  // Find all organisms of this species
  const speciesOrganisms = frameIndex.organismsBySignature.get(note.speciesSignature);
  if (!speciesOrganisms || speciesOrganisms.length === 0) return null;

  // Collect all organelle physics of the target typeId across all organisms
  const allPhysics: OrganellePhysics[] = [];
  for (const ro of speciesOrganisms) {
    const bfsOrder = frameIndex.bfsOrderByRegistryId.get(ro.registryId);
    if (!bfsOrder) continue;
    for (const orgId of bfsOrder) {
      const orgState = frameIndex.organelleById.get(orgId);
      if (orgState && orgState.typeId === note.typeId) {
        const p = frameIndex.physicsByOrganelle.get(orgId);
        if (p) allPhysics.push(p);
      }
    }
  }
  if (allPhysics.length === 0) return null;

  // Mean density across all organelles of this type in species
  const meanDensity = allPhysics.reduce((s, p) => s + p.density, 0) / allPhysics.length;

  // Mean particle count
  const meanParticleCount = allPhysics.reduce((s, p) => s + p.particleCount, 0) / allPhysics.length;

  // Pan from mean organism centroid X
  const meanCentroidX = speciesOrganisms.reduce((s, ro) => s + ro.centroidX, 0) / speciesOrganisms.length;
  const pan = canvasWidth > 0 ? (meanCentroidX / canvasWidth) * 2 - 1 : 0;

  // Age norm from oldest organism in species
  const AGE_HALF_LIFE = 8;
  let oldestCreation = Infinity;
  for (const ro of speciesOrganisms) {
    if (ro.creationTime < oldestCreation) oldestCreation = ro.creationTime;
  }
  const ageInBars = (ctx.barStartTime - oldestCreation) / barDur;
  const ageNorm = ageInBars > 0 ? ageInBars / (ageInBars + AGE_HALF_LIFE) : 0;

  // Mean rank norms across all organelles of this type in species
  let countNormSum = 0, speedNormSum = 0, densityNormSum = 0, radiusNormSum = 0;
  let normsCount = 0;
  for (const p of allPhysics) {
    const n = ranks.get(p.id);
    if (n) {
      countNormSum += n.countNorm;
      speedNormSum += n.speedNorm;
      densityNormSum += n.densityNorm;
      radiusNormSum += n.radiusNorm;
      normsCount++;
    }
  }
  const norms: EnvelopeNorms = normsCount > 0
    ? {
      countNorm: countNormSum / normsCount,
      speedNorm: speedNormSum / normsCount,
      densityNorm: densityNormSum / normsCount,
      radiusNorm: radiusNormSum / normsCount,
      ageNorm,
    }
    : { countNorm: 0.5, speedNorm: 0.5, densityNorm: 0.5, radiusNorm: 0.5, ageNorm };

  // Volume from mean density + count norm
  const volume = densityToVolume(meanDensity) * (1 - 0.5 * norms.countNorm);

  // Filter cutoff from mean particle count
  const filterCutoff = countToFilterCutoff(
    meanParticleCount,
    ctx.envelopeRanges.particleCountMin,
    ctx.envelopeRanges.particleCountMax,
  );

  // Envelope from aggregated norms
  const envelope = computeEnvelope(norms);

  // Gate duration from note duration
  const noteDur = computeNoteDuration(norms);
  const beatDur = barDur / 4;
  const gateDuration = noteDurationToSeconds(noteDur, beatDur);

  // Waveform from sociability (per typeId, unchanged)
  const typeKey = typeKeys[note.typeId] ?? "";
  const sociability = sociabilities.get(typeKey) ?? 0.5;
  const waveform = sociabilityToWaveform(sociability);

  // Vibrato from mean speed norm
  const vibratoDepth = norms.speedNorm * 100;

  return { volume, pan, filterCutoff, envelope, gateDuration, waveform, vibratoDepth };
}
