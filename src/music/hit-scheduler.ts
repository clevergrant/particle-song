/**
 * Bar-boundary hit scheduler (§3.2, §3.3).
 *
 * Pure function: given a BarSnapshot + timing info + music state,
 * produce a ScheduledBar containing every hit for the upcoming bar.
 * The audio graph then plays these fire-and-forget.
 */

import type {
  BarSnapshot,
  ScheduledBar,
  ScheduledHit,
  SnapshotOrganelle,
  MusicState,
  BassUpdate,
  ArpNote,
  NoteDuration,
  EnvelopeParams,
  EnvelopeRanges,
  TransitionChord,
} from "./types";

import { computeNetStability } from "./stability";
import { selectMode } from "./scale-selector";
import { computeTypeRoots, deriveRoot, findOldestSpecies } from "./root-derivation";
import { computeTransitionBuffer, pitchClassSet } from "./transition-buffer";
import { collapseToPhase, computeRegisterWidth, clampToRegister, crossTypeLinksToPhase } from "./overtone-phases";
import { computeAllSociabilities, sociabilityToWaveform } from "./timbre";
import { subdivisionTimes } from "./timing";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const MAX_SUBDIVISION = 16;

/** EMA smoothing factor: 30% current bar, 70% history. */
const EMA_ALPHA = 0.3;

/** Minimum range width — prevents degenerate normalization when all
 *  organelles have near-identical properties. */
const MIN_RANGE_PC = 2;       // particle count
const MIN_RANGE_SPEED = 0.5;  // centroid speed
const MIN_RANGE_DENSITY = 0.3;
const MIN_RANGE_RADIUS = 5;

/** Ensure a min/max range has at least `minWidth`, expanding symmetrically. */
function ensureMinWidth(min: number, max: number, minWidth: number): [number, number] {
  if (max - min >= minWidth) return [min, max];
  const mid = (min + max) / 2;
  return [mid - minWidth / 2, mid + minWidth / 2];
}

/** Blend current-bar range with the EMA from the previous bar. */
function emaBlend(current: number, prev: number): number {
  return EMA_ALPHA * current + (1 - EMA_ALPHA) * prev;
}

/* ------------------------------------------------------------------ */
/*  Expression mapping helpers                                         */
/* ------------------------------------------------------------------ */

/** Normalize particle count to [0,1] within the observed range. */
function normalizeCount(particleCount: number, minCount: number, maxCount: number): number {
  if (maxCount <= minCount) return 0.5;
  return (particleCount - minCount) / (maxCount - minCount);
}

/** Map organelle density to volume (0–1). Denser clusters sound louder/more cohesive. */
function densityToVolume(density: number): number {
  // density is typically 0–2+; clamp to [0,1] with a reasonable ceiling
  return Math.min(Math.max(density / 2, 0.15), 1);
}

/** Map particle count to octave offset (larger = lower). */
function countToOctaveOffset(particleCount: number, minCount: number, maxCount: number): number {
  const normalized = normalizeCount(particleCount, minCount, maxCount);
  // Range: +1 (small, high) to -1 (large, low)
  return Math.round(1 - 2 * normalized);
}

/** Map particle count to bandpass filter cutoff Hz. */
function countToFilterCutoff(particleCount: number, minCount: number, maxCount: number): number {
  const normalized = normalizeCount(particleCount, minCount, maxCount);
  // Larger = lower cutoff (warm), smaller = higher (bright)
  // Range: 200 Hz (large) to 8000 Hz (small)
  return 8000 * Math.pow(200 / 8000, normalized);
}

/** Normalize a value into 0–1 given min/max bounds. */
function normalizeRange(value: number, min: number, max: number): number {
  return max > min ? (value - min) / (max - min) : 0;
}

/** Pre-computed 0–1 normalized values for the envelope-driving properties. */
interface EnvelopeNorms {
  readonly countNorm: number;
  readonly speedNorm: number;
  readonly densityNorm: number;
  readonly radiusNorm: number;
  readonly ageNorm: number;
}

/** Compute rank-based normalization for a set of organelles.
 *  Each organelle's rank for each property is mapped to [0,1] so that
 *  the lowest always gets 0 (staccato) and highest gets 1 (sustained),
 *  regardless of how close the absolute values are.  When only one
 *  organelle exists, the EMA-based absolute normalization is used as
 *  the fallback so the physics still drives the envelope shape. */
function computeOrganelleRanks(
  organelles: readonly SnapshotOrganelle[],
  ranges: EnvelopeRanges,
): ReadonlyMap<number, EnvelopeNorms> {
  const n = organelles.length;
  const result = new Map<number, EnvelopeNorms>();
  if (n === 0) return result;

  // Single organelle — use absolute normalization against EMA ranges
  if (n === 1) {
    const o = organelles[0];
    result.set(o.id, {
      countNorm: normalizeRange(o.particleCount, ranges.particleCountMin, ranges.particleCountMax),
      speedNorm: normalizeRange(o.centroidSpeed, ranges.speedMin, ranges.speedMax),
      densityNorm: normalizeRange(o.density, ranges.densityMin, ranges.densityMax),
      radiusNorm: normalizeRange(o.spatialRadius, ranges.radiusMin, ranges.radiusMax),
      ageNorm: 0, // overridden per-organism at hit site
    });
    return result;
  }

  // Multiple organelles — rank-based normalization
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
      ageNorm: 0, // overridden per-organism at hit site
    });
  }
  return result;
}

/** Compute ADSR envelope timing from pre-normalized 0–1 values (§5.4, §8.3).
 *  Sustain duration is no longer pre-computed — it comes from the gate length. */
function computeEnvelope(norms: EnvelopeNorms): EnvelopeParams {
  const { countNorm, speedNorm, densityNorm, radiusNorm, ageNorm } = norms;

  // Attack ← centroid speed × particle count, shortened for young organelles
  // Young (ageNorm≈0) → very short attack; older organisms stretch slightly (1.2×)
  const baseAttack = 0.02 + (1 - speedNorm) * 0.3 + countNorm * 0.2;
  const attackDuration = Math.max(0.01, baseAttack * ageNorm * 1.2);

  // Peak level ← speed × density, quieter for young organelles
  const peakLevel = (0.4 + speedNorm * 0.35 + densityNorm * 0.25) * (0.3 + 0.7 * ageNorm);

  // Decay ← density: dense = short, diffuse = long
  const decayDuration = 0.1 + (1 - densityNorm) * 0.6;

  // Sustain level ← particle count: big = high, small = low
  const sustainLevel = 0.1 + countNorm * 0.7;

  // Release ← spatial radius: large = long tail, small = quick cutoff
  const releaseDuration = 0.1 + radiusNorm * 1.0;

  // Curve shapes from speed (§8.3): fast → percussive (linear), slow → swelling (ease-in)
  const attackCurve: EnvelopeParams["attackCurve"] = speedNorm > 0.6 ? "linear" : "ease-in";
  const decayCurve: EnvelopeParams["decayCurve"] = densityNorm > 0.5 ? "exponential" : "ease-out";
  const releaseCurve: EnvelopeParams["releaseCurve"] = "ease-out";

  return { attackDuration, attackCurve, peakLevel, decayDuration, decayCurve, sustainLevel, releaseDuration, releaseCurve };
}

/* ------------------------------------------------------------------ */
/*  Note duration (gate model)                                         */
/* ------------------------------------------------------------------ */

const NOTE_DURATION_BEATS: Record<NoteDuration, number> = {
  whole: 4,
  half: 2,
  quarter: 1,
  eighth: 0.5,
  sixteenth: 0.25,
};

/** Map organelle physics to a musical note duration.
 *  Primary driver: countNorm (large = long notes).
 *  Secondary: ageNorm (older organisms hold longer). */
function computeNoteDuration(norms: EnvelopeNorms): NoteDuration {
  const blend = 0.5 * norms.countNorm + 0.3 * norms.densityNorm + 0.2 * norms.ageNorm;
  if (blend > 0.85) return "whole";
  if (blend > 0.65) return "half";
  if (blend > 0.40) return "quarter";
  if (blend > 0.20) return "eighth";
  return "sixteenth";
}

function noteDurationToSeconds(nd: NoteDuration, beatDur: number): number {
  return NOTE_DURATION_BEATS[nd] * beatDur;
}


/* ------------------------------------------------------------------ */
/*  Main scheduler                                                     */
/* ------------------------------------------------------------------ */

export interface ScheduleConfig {
  readonly barsPerPhase: number;         // overtone phase rate
  readonly qualificationFraction: number; // fraction of a bar for organism qualification
  readonly hysteresisMargin?: number;
  readonly preferNiceModes?: boolean;    // divide thresholds by 2 → nicer modes at lower stability
  readonly beatsPerBar: number;          // for gate duration computation
}

/**
 * Schedule all hits for the upcoming bar.
 *
 * This is the core of the bar-boundary architecture (§3.1, §3.2):
 * 1. Snapshot is taken at bar boundary
 * 2. All musical decisions are made here
 * 3. Returns a ScheduledBar that the audio graph plays fire-and-forget
 */
export function scheduleBar(
  snapshot: BarSnapshot,
  barNumber: number,
  barStartTime: number,
  barDur: number,
  prevState: MusicState | null,
  config: ScheduleConfig,
): ScheduledBar {
  const { organisms, globalMetrics, forceMatrix, typeKeys, canvasWidth } = snapshot;

  // ── Stability → mode selection ────────────────────────────────────
  const netStability = computeNetStability(globalMetrics);
  const prevMode = prevState?.currentMode ?? null;
  const mode = selectMode(netStability, prevMode, config.hysteresisMargin, config.preferNiceModes);

  // ── Root derivation ───────────────────────────────────────────────
  const typeRoots = computeTypeRoots(forceMatrix, typeKeys);
  const oldestSpecies = findOldestSpecies(organisms);
  const rootSemitone = deriveRoot(typeRoots, oldestSpecies, typeKeys);
  const rootMidi = 60 + rootSemitone; // C4 + semitone offset

  // ── Transition buffer check ───────────────────────────────────────
  let isBufferBar = false;
  let bufferChord: TransitionChord | null = null;

  if (prevState) {
    bufferChord = computeTransitionBuffer(
      prevState.currentMode,
      prevState.currentRootMidi % 12,
      mode,
      rootSemitone,
      prevState.bufferChord,
    );
    isBufferBar = bufferChord !== null;
  }

  // ── Active pitch classes for this bar ─────────────────────────────
  const activePitchClasses = isBufferBar && bufferChord
    ? bufferChord.pitchClasses
    : pitchClassSet(mode, rootSemitone);

  // ── Timbre (sociability → waveform) ───────────────────────────────
  const sociabilities = computeAllSociabilities(forceMatrix, typeKeys);

  // ── Register width ────────────────────────────────────────────────
  const registerWidth = computeRegisterWidth(organisms.length);

  // ── Global ranges + playable organelle collection ──────────────────
  // Organelles arrive in BFS tree order from the snapshot.  Each organism's
  // total organelle count (capped at MAX_SUBDIVISION) becomes its subdivision.
  const allPlayable: SnapshotOrganelle[] = [];
  let gPcMin = Infinity;
  let gPcMax = -Infinity;
  let gSpeedMin = Infinity;
  let gSpeedMax = -Infinity;
  let gDensityMin = Infinity;
  let gDensityMax = -Infinity;
  let gRadiusMin = Infinity;
  let gRadiusMax = -Infinity;
  for (const org of organisms) {
    const playable = org.organelles.length <= MAX_SUBDIVISION
      ? org.organelles
      : org.organelles.slice(0, MAX_SUBDIVISION);
    for (const o of playable) {
      allPlayable.push(o);
      if (o.particleCount < gPcMin) gPcMin = o.particleCount;
      if (o.particleCount > gPcMax) gPcMax = o.particleCount;
      if (o.centroidSpeed < gSpeedMin) gSpeedMin = o.centroidSpeed;
      if (o.centroidSpeed > gSpeedMax) gSpeedMax = o.centroidSpeed;
      if (o.density < gDensityMin) gDensityMin = o.density;
      if (o.density > gDensityMax) gDensityMax = o.density;
      if (o.spatialRadius < gRadiusMin) gRadiusMin = o.spatialRadius;
      if (o.spatialRadius > gRadiusMax) gRadiusMax = o.spatialRadius;
    }
  }
  if (!isFinite(gPcMin)) { gPcMin = 0; gPcMax = 1; }
  if (!isFinite(gSpeedMin)) { gSpeedMin = 0; gSpeedMax = 1; }
  if (!isFinite(gDensityMin)) { gDensityMin = 0; gDensityMax = 1; }
  if (!isFinite(gRadiusMin)) { gRadiusMin = 0; gRadiusMax = 1; }

  // ── EMA-blended envelope ranges ──────────────────────────────────
  // Blend this bar's observed ranges with the running EMA so that
  // normalization always produces a staccato↔sustained spread even
  // when all current organelles have similar properties.
  const prevRanges = prevState?.envelopeRanges ?? null;
  const rawRanges: EnvelopeRanges = {
    particleCountMin: gPcMin, particleCountMax: gPcMax,
    speedMin: gSpeedMin, speedMax: gSpeedMax,
    densityMin: gDensityMin, densityMax: gDensityMax,
    radiusMin: gRadiusMin, radiusMax: gRadiusMax,
  };
  const blendedRanges: EnvelopeRanges = prevRanges
    ? {
      particleCountMin: emaBlend(rawRanges.particleCountMin, prevRanges.particleCountMin),
      particleCountMax: emaBlend(rawRanges.particleCountMax, prevRanges.particleCountMax),
      speedMin: emaBlend(rawRanges.speedMin, prevRanges.speedMin),
      speedMax: emaBlend(rawRanges.speedMax, prevRanges.speedMax),
      densityMin: emaBlend(rawRanges.densityMin, prevRanges.densityMin),
      densityMax: emaBlend(rawRanges.densityMax, prevRanges.densityMax),
      radiusMin: emaBlend(rawRanges.radiusMin, prevRanges.radiusMin),
      radiusMax: emaBlend(rawRanges.radiusMax, prevRanges.radiusMax),
    }
    : rawRanges;

  // Enforce minimum range widths to prevent degenerate normalization
  const [ePcMin, ePcMax] = ensureMinWidth(blendedRanges.particleCountMin, blendedRanges.particleCountMax, MIN_RANGE_PC);
  const [eSpMin, eSpMax] = ensureMinWidth(blendedRanges.speedMin, blendedRanges.speedMax, MIN_RANGE_SPEED);
  const [eDnMin, eDnMax] = ensureMinWidth(blendedRanges.densityMin, blendedRanges.densityMax, MIN_RANGE_DENSITY);
  const [eRdMin, eRdMax] = ensureMinWidth(blendedRanges.radiusMin, blendedRanges.radiusMax, MIN_RANGE_RADIUS);
  const envelopeRanges: EnvelopeRanges = {
    particleCountMin: ePcMin, particleCountMax: ePcMax,
    speedMin: eSpMin, speedMax: eSpMax,
    densityMin: eDnMin, densityMax: eDnMax,
    radiusMin: eRdMin, radiusMax: eRdMax,
  };

  // ── Rank-based envelope normalization ──────────────────────────────
  // Ranks guarantee a staccato↔sustained spread: the organelle with the
  // lowest value always normalizes to 0, the highest to 1, regardless of
  // how close the absolute values are.  Single-organelle bars fall back
  // to EMA-based absolute normalization so physics still drives the shape.
  const organelleNorms = computeOrganelleRanks(allPlayable, envelopeRanges);

  // ── Schedule hits per organism ────────────────────────────────────
  const hits: ScheduledHit[] = [];

  // Occupancy tracking: maps quantized time offsets (relative to barStart)
  // to hit counts, so we prefer empty positions before doubling up.
  // Uses a tolerance of 0.001s (~1ms) to treat near-coincident times as same slot.
  const occupiedTimes: number[] = [];  // sorted list of occupied offsets
  const occupancyCounts = new Map<number, number>(); // offset → count

  /** Snap a raw bar-relative offset to the nearest position on the voice's
   *  own subdivision grid, preferring unoccupied positions. This preserves
   *  tuplet feel (triplets, quintuplets, etc.) by not pulling hits onto
   *  the 16th-note grid. */
  function quantizeToGrid(rawOffset: number, subdivisionGrid: readonly number[]): number {
    // Use only this voice's natural subdivision grid as candidates
    const candidates = new Set<number>();
    for (const p of subdivisionGrid) candidates.add(Math.round(p * 1000) / 1000);
    // Sort by distance from rawOffset
    const sorted = [...candidates].sort((a, b) =>
      Math.abs(a - rawOffset) - Math.abs(b - rawOffset)
    );
    // Pick the nearest unoccupied position
    for (const pos of sorted) {
      if (!occupancyCounts.has(pos)) {
        occupancyCounts.set(pos, 1);
        occupiedTimes.push(pos);
        return pos;
      }
    }
    // All positions occupied — double up on nearest
    const nearest = sorted[0];
    occupancyCounts.set(nearest, (occupancyCounts.get(nearest) ?? 0) + 1);
    return nearest;
  }

  // Pre-filter to qualified organisms so we know the total count for phase rotation.
  const qualifiedOrganisms = organisms.filter(org => {
    const ageInBars = (barStartTime - org.creationTime) / barDur;
    return ageInBars >= config.qualificationFraction;
  });
  const numOrganisms = qualifiedOrganisms.length;

  for (let orgIdx = 0; orgIdx < numOrganisms; orgIdx++) {
    const organism = qualifiedOrganisms[orgIdx];
    const ageInBars = (barStartTime - organism.creationTime) / barDur;

    // Age norm: saturation curve — half-life at 8 bars, asymptotes to 1.
    // Young organisms get short staccato sustains; old ones get legato holds.
    const AGE_HALF_LIFE = 8;
    const ageNorm = ageInBars / (ageInBars + AGE_HALF_LIFE);

    // Pan from organism centroid X position (§6.1)
    const pan = canvasWidth > 0
      ? (organism.centroidX / canvasWidth) * 2 - 1
      : 0;

    // ── Organism-wide subdivision ──────────────────────────────────────
    // Total organelle count (capped at MAX_SUBDIVISION) determines the
    // rhythmic grid.  Organelles arrive in BFS tree order from the snapshot
    // so the structural core of the organism plays the downbeat.
    const playable = organism.organelles.length <= MAX_SUBDIVISION
      ? organism.organelles
      : organism.organelles.slice(0, MAX_SUBDIVISION);
    const subdivision = playable.length;
    if (subdivision === 0) continue;

    // Per-organism particle count range for octave offset normalization.
    let orgPcMin = Infinity;
    let orgPcMax = -Infinity;
    for (const o of playable) {
      if (o.particleCount < orgPcMin) orgPcMin = o.particleCount;
      if (o.particleCount > orgPcMax) orgPcMax = o.particleCount;
    }
    if (!isFinite(orgPcMin)) { orgPcMin = 0; orgPcMax = 1; }

    // Blend per-organism and global ranges (50/50)
    const LOCAL_WEIGHT = 0.5;
    const bPcMin = LOCAL_WEIGHT * orgPcMin + (1 - LOCAL_WEIGHT) * gPcMin;
    const bPcMax = LOCAL_WEIGHT * orgPcMax + (1 - LOCAL_WEIGHT) * gPcMax;

    const rawTimes = subdivisionTimes(barStartTime, barDur, subdivision);
    // Build the organism's subdivision grid (bar-relative offsets)
    const subdivGrid: number[] = [];
    for (let s = 0; s < subdivision; s++) subdivGrid.push((s / subdivision) * barDur);
    const hitTimes = rawTimes.map(t =>
      barStartTime + quantizeToGrid((t - barStartTime + barDur) % barDur, subdivGrid)
    );

    for (let i = 0; i < subdivision; i++) {
      const organelle = playable[i];
      const typeId = organelle.typeId;

      // Harmonic phase from cross-type connectivity: more structural
      // connections unlock richer intervals from the overtone series.
      const harmonicPhase = crossTypeLinksToPhase(organelle.crossTypeLinks);

      // Pitch: type → relative interval from root → phase collapse → octave
      const typeKey = typeKeys[typeId];
      const typeRoot = typeRoots.get(typeKey ?? "") ?? 0;
      const relativeInterval = ((typeRoot - rootSemitone) % 12 + 12) % 12;
      const collapsed = collapseToPhase(relativeInterval, harmonicPhase);

      // Base MIDI = root + collapsed relative interval
      let midiNote = rootMidi + collapsed;

      // Octave offset from particle count (larger = lower), blended normalization
      midiNote += countToOctaveOffset(organelle.particleCount, bPcMin, bPcMax) * 12;

      // Clamp to register width
      midiNote = clampToRegister(midiNote, registerWidth);

      // Filter against active pitch classes (buffer bar constraint)
      const pc = ((midiNote % 12) + 12) % 12;
      if (!activePitchClasses.has(pc)) {
        let bestDist = 12;
        let bestPC = pc;
        for (const available of activePitchClasses) {
          const dist = Math.min(
            ((pc - available) % 12 + 12) % 12,
            ((available - pc) % 12 + 12) % 12,
          );
          if (dist < bestDist) { bestDist = dist; bestPC = available; }
        }
        const adjustment = bestPC - pc;
        midiNote += adjustment > 6 ? adjustment - 12 : adjustment < -6 ? adjustment + 12 : adjustment;
      }

      // Waveform from sociability
      const sociability = sociabilities.get(typeKey ?? "") ?? 0.5;
      const waveform = sociabilityToWaveform(sociability);

      // Compute norms with age for this organelle
      const hitNorms: EnvelopeNorms = { ...(organelleNorms.get(organelle.id) ?? { countNorm: 0.5, speedNorm: 0.5, densityNorm: 0.5, radiusNorm: 0.5, ageNorm: 0.5 }), ageNorm };
      const noteDur = computeNoteDuration(hitNorms);
      const beatDur = barDur / config.beatsPerBar;

      hits.push({
        time: hitTimes[i],
        organismId: organism.registryId,
        typeId,
        organelleIndex: i,
        midiNote,
        volume: densityToVolume(organelle.density) * (1 - 0.5 * hitNorms.countNorm),
        pan,
        filterCutoff: countToFilterCutoff(organelle.particleCount, bPcMin, bPcMax),
        envelope: computeEnvelope(hitNorms),
        noteDuration: noteDur,
        gateDuration: noteDurationToSeconds(noteDur, beatDur),
        waveform,
        vibratoDepth: hitNorms.speedNorm * 100,
      });
    }
  }

  // ── Bass arpeggio pool (ordered pitches from free particle types) ───
  const arpNotes: ArpNote[] = [];

  // Collect types with free particles, sorted by abundance (most first)
  const freeTypes: { typeKey: string; percent: number }[] = [];
  for (let i = 0; i < typeKeys.length; i++) {
    const percent = globalMetrics.freeParticlePercentByType.get(i) ?? 0;
    if (percent > 0) {
      freeTypes.push({ typeKey: typeKeys[i], percent });
    }
  }
  freeTypes.sort((a, b) => b.percent - a.percent);

  for (const { typeKey, percent } of freeTypes) {
    // Pitch: type's interval relative to root, placed in bass register
    // (matches organism hit logic: rootMidi + relativeInterval)
    const typeSemitone = typeRoots.get(typeKey) ?? 0;
    const relativeInterval = ((typeSemitone - rootSemitone) % 12 + 12) % 12;
    let midiNote = 36 + rootSemitone + relativeInterval;

    // Filter against active pitch classes (transition buffer constraint)
    const pc = ((midiNote % 12) + 12) % 12;
    if (!activePitchClasses.has(pc)) {
      let bestDist = 12;
      let bestPC = pc;
      for (const available of activePitchClasses) {
        const dist = Math.min(
          ((pc - available) % 12 + 12) % 12,
          ((available - pc) % 12 + 12) % 12,
        );
        if (dist < bestDist) { bestDist = dist; bestPC = available; }
      }
      // Use circular-aware adjustment to avoid octave jumps at the boundary
      const adjustment = bestPC - pc;
      midiNote += adjustment > 6 ? adjustment - 12 : adjustment < -6 ? adjustment + 12 : adjustment;
    }

    const sociability = sociabilities.get(typeKey) ?? 0.5;

    arpNotes.push({ midiNote, sociability, freePercent: percent });
  }

  // ── Bass update ──────────────────────────────────────────────────
  // No organisms → pedal on the tonic only; arpeggiation starts once
  // at least one organism is alive.  If organisms consumed all free
  // particles the arp pool is empty — fall back to tonic pedal so
  // the bass never goes silent.
  const tonicPedal: ArpNote[] = [{ midiNote: 36 + rootSemitone, sociability: 0.5, freePercent: 1 }];
  const bassArpNotes: ArpNote[] = organisms.length === 0
    ? tonicPedal
    : arpNotes.length > 0 ? arpNotes : tonicPedal;

  const bassUpdate: BassUpdate = {
    root: rootMidi,
    fifth: rootMidi + 7,
    mode,
    freeParticlePercentByType: globalMetrics.freeParticlePercentByType,
    isQuartalStack: organisms.length === 0,
    arpNotes: bassArpNotes,
    netStability,
    avgVelocity: globalMetrics.avgVelocity,
  };

  return {
    barNumber,
    startTime: barStartTime,
    duration: barDur,
    hits,
    mode,
    rootMidi,
    isBufferBar,
    bufferChord,
    bassUpdate,
    netStability,
    spatialEntropy: globalMetrics.spatialEntropy,
    envelopeRanges: blendedRanges,
    speciesCycle: prevState?.speciesCycle ?? { played: new Map() },
  };
}
