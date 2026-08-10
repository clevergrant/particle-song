/**
 * Bar-boundary hit scheduler (§3.2, §3.3).
 *
 * Pure functions: given a BarSnapshot + timing info + music state,
 * produce bar-level context and per-organism slot assignments for
 * the tuplet grid. Playback parameters are computed at play time
 * from live simulation state (see play-time-params.ts).
 */

import {
  MAX_SUBDIVISION,
  type BarSnapshot,
  type SlotNote,
  type MusicState,
  type ModeDefinition,
  type EnvelopeRanges,
  type TransitionChord,
  type SpeciesCycle,
} from "./types";

import { computeNetStability } from "./stability";
import { selectMode } from "./scale-selector";
import { diatonicTriad } from "./modes";
import { computeTypeRoots, deriveRoot, findOldestSpecies } from "./root-derivation";
import { computeTransitionBuffer, pitchClassSet } from "./transition-buffer";
import { collapseToPhase, crossTypeLinksToPhase } from "./overtone-phases";
import { computeAllSociabilities } from "./timbre";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Per-type pitch range from particle count                           */
/* ------------------------------------------------------------------ */

/** MIDI boundaries for the 2-octave sliding window. */
const PITCH_RANGE_LOW = 36;   // C2
const PITCH_RANGE_HIGH = 84;  // C6
const WINDOW_SIZE = 24;       // 2 octaves

export interface TypePitchRange {
  readonly lowMidi: number;
  readonly highMidi: number;
}

/**
 * Compute a 2-octave pitch window per organelle type.
 * More particles → lower range; fewer → higher.
 *
 * Uses rank-based quantization: types are sorted by average particle count
 * and assigned to evenly-spaced bands across the full pitch range.
 * This guarantees a good spread even when particle counts are similar.
 */
export function computeTypePitchRanges(
  organisms: readonly import("./types").SnapshotOrganism[],
): ReadonlyMap<number, TypePitchRange> {
  const typeSums = new Map<number, { total: number; count: number }>();

  for (const org of organisms) {
    for (const o of org.organelles) {
      const entry = typeSums.get(o.typeId);
      if (entry) {
        entry.total += o.particleCount;
        entry.count++;
      } else {
        typeSums.set(o.typeId, { total: o.particleCount, count: 1 });
      }
    }
  }

  // Rank types by average particle count (descending — most particles = lowest pitch)
  const ranked: { typeId: number; avg: number }[] = [];
  for (const [typeId, { total, count }] of typeSums) {
    ranked.push({ typeId, avg: total / count });
  }
  ranked.sort((a, b) => b.avg - a.avg); // most particles first → lowest band

  const result = new Map<number, TypePitchRange>();
  const n = ranked.length;
  if (n === 0) return result;

  // Slide range: how far the window center can move
  const slideRange = PITCH_RANGE_HIGH - WINDOW_SIZE - PITCH_RANGE_LOW;

  for (let i = 0; i < n; i++) {
    // Evenly space: rank 0 (most particles) → bottom, rank n-1 (fewest) → top
    const t = n === 1 ? 0.5 : i / (n - 1);
    const lowMidi = Math.round(PITCH_RANGE_LOW + t * slideRange);
    result.set(ranked[i].typeId, { lowMidi, highMidi: lowMidi + WINDOW_SIZE });
  }

  return result;
}

/* ------------------------------------------------------------------ */
/*  Bar context (pre-organism computation)                             */
/* ------------------------------------------------------------------ */

/** All bar-level musical decisions made before the per-organism loop. */
export interface BarContext {
  readonly mode: ModeDefinition;
  readonly rootMidi: number;
  readonly rootSemitone: number;
  readonly isBufferBar: boolean;
  readonly bufferChord: TransitionChord | null;
  readonly activePitchClasses: ReadonlySet<number>;
  /** This bar's chord (diatonic triad on the cycle's degree, or buffer chord). */
  readonly chordPitchClasses: ReadonlySet<number>;
  readonly typeRoots: ReadonlyMap<string, number>;
  readonly sociabilities: ReadonlyMap<string, number>;
  readonly typePitchRanges: ReadonlyMap<number, TypePitchRange>;
  readonly netStability: number;
  readonly envelopeRanges: EnvelopeRanges;
  readonly spatialEntropy: number;
  readonly speciesCycle: SpeciesCycle;
  readonly typeKeys: readonly string[];
  readonly canvasWidth: number;
}

export interface ScheduleConfig {
  readonly barsPerPhase: number;         // overtone phase rate
  readonly qualificationFraction: number; // fraction of a bar for organism qualification
  readonly hysteresisMargin?: number;
  readonly preferNiceModes?: boolean;    // divide thresholds by 2 → nicer modes at lower stability
}

/**
 * Compute bar-level context: mode, root, bass, ranges — everything
 * before the per-organism loop. Fast enough to post as `bar-meta`
 * immediately from the worker.
 */
export function computeBarContext(
  snapshot: BarSnapshot,
  barStartTime: number,
  barDur: number,
  prevState: MusicState | null,
  config: ScheduleConfig,
): BarContext {
  const { organisms, globalMetrics, forceMatrix, typeKeys, canvasWidth } = snapshot;

  const netStability = computeNetStability(globalMetrics);
  const prevMode = prevState?.currentMode ?? null;
  const mode = selectMode(netStability, prevMode, config.hysteresisMargin, config.preferNiceModes);

  const typeRoots = computeTypeRoots(forceMatrix, typeKeys);
  const rootSemitone = snapshot.rootOverride != null
    ? snapshot.rootOverride
    : deriveRoot(typeRoots, findOldestSpecies(organisms), typeKeys);
  const rootMidi = 60 + rootSemitone;

  let isBufferBar = false;
  let bufferChord: TransitionChord | null = null;
  if (prevState) {
    bufferChord = computeTransitionBuffer(
      prevState.currentMode, prevState.currentRootMidi % 12,
      mode, rootSemitone, prevState.bufferChord,
    );
    isBufferBar = bufferChord !== null;
  }

  const activePitchClasses = isBufferBar && bufferChord
    ? bufferChord.pitchClasses
    : pitchClassSet(mode, rootSemitone);

  // Bar chord: buffer chord during transitions, else the diatonic triad
  // on the organism cycle's chosen degree within the current key.
  const chordPitchClasses = isBufferBar && bufferChord
    ? bufferChord.pitchClasses
    : diatonicTriad(mode, rootSemitone, snapshot.chordDegree);

  const sociabilities = computeAllSociabilities(forceMatrix, typeKeys);
  // Pitch ranges scoped to the active species (one organism per bar)
  const activeOrganisms = snapshot.activeSpecies != null
    ? organisms.filter(o => o.colorSignature === snapshot.activeSpecies)
    : organisms;
  const typePitchRanges = computeTypePitchRanges(activeOrganisms);

  // Global ranges across all playable organelles
  let gPcMin = Infinity, gPcMax = -Infinity;
  let gSpeedMin = Infinity, gSpeedMax = -Infinity;
  let gDensityMin = Infinity, gDensityMax = -Infinity;
  let gRadiusMin = Infinity, gRadiusMax = -Infinity;
  for (const org of organisms) {
    const playable = org.organelles.length <= MAX_SUBDIVISION
      ? org.organelles : org.organelles.slice(0, MAX_SUBDIVISION);
    for (const o of playable) {
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

  return {
    mode, rootMidi, rootSemitone, isBufferBar, bufferChord,
    activePitchClasses, chordPitchClasses, typeRoots, sociabilities, typePitchRanges,
    netStability, envelopeRanges: blendedRanges,
    spatialEntropy: globalMetrics.spatialEntropy,
    speciesCycle: prevState?.speciesCycle ?? { played: new Map() },
    typeKeys, canvasWidth,
  };
}

/* ------------------------------------------------------------------ */
/*  Per-species slot computation                                       */
/* ------------------------------------------------------------------ */

/** Per-slot pitch offsets (semitones) cycled across a tuplet's slots. */
const ARPEGGIO_OFFSETS: readonly number[] = [0, 7, 4, 12];

/** A single tier's slot fill from one type-layer of a species' polyrhythm. */
export interface SpeciesSlotResult {
  readonly tierIndex: number;   // = subdivision - 1
  readonly slots: readonly { readonly slotIndex: number; readonly note: SlotNote }[];
}

/**
 * Maximally-even (Euclidean) onset pattern: k onsets across n slots,
 * onset o at slot floor(o·n/k). Slot 0 is always an onset.
 */
function euclideanOnsets(k: number, n: number): readonly number[] {
  const slots: number[] = [];
  for (let o = 0; o < k; o++) slots.push(Math.floor((o * n) / k));
  return slots;
}

/**
 * Smallest rotation ≥ `start` (mod n) that leaves slot 0 empty.
 * When k < n such a rotation always exists (n − k of the n rotations
 * put a gap on the downbeat); with a full pattern the start rotation
 * is returned unchanged.
 */
function rotationOffDownbeat(
  onsets: readonly number[],
  n: number,
  start: number,
): number {
  let r = ((start % n) + n) % n;
  for (let i = 0; i < n; i++) {
    if (!onsets.some(s => (s + r) % n === 0)) return r;
    r = (r + 1) % n;
  }
  return ((start % n) + n) % n;
}

/**
 * Compute slot placements for all species as interlocking polyrhythms.
 *
 * Groups organisms by colorSignature (species). For each species +
 * organelle type, the MAX organelle count across all organisms of that
 * species sets the tuplet grid (e.g. if organism A has 3 Red and
 * organism B has 5 Red, the species gets a quintuplet Red grid), and
 * cross-type connectivity sets how many of those slots actually sound:
 * onsets = 1 + crossTypeLinks, placed maximally evenly (Euclidean).
 *
 * The downbeat belongs to ONE anchor layer per species — the type with
 * the most particles (the lowest voice). Every other layer is rotated
 * so its pattern avoids slot 0, and lone-organelle non-anchor types are
 * promoted onto the species' widest grid so they spread across the bar
 * instead of stacking on the barline.
 *
 * Pitch is derived from the oldest organism in the species.
 *
 * Returns one SpeciesSlotResult per type per species.
 */
export function computeSpeciesSlots(
  snapshot: BarSnapshot,
  ctx: BarContext,
  barStartTime: number,
  barDur: number,
  config: ScheduleConfig,
  barNumber?: number,
): readonly SpeciesSlotResult[] {
  const { organisms } = snapshot;

  // Group organisms by species (colorSignature), filtered to active species if set
  const bySpecies = new Map<string, typeof organisms[number][]>();
  for (const org of organisms) {
    if (snapshot.activeSpecies != null && org.colorSignature !== snapshot.activeSpecies) continue;
    // Skip qualification on bar 0 — all organisms are born at t=0 so none can pass the age check
    if (barNumber !== 0) {
      const ageInBars = (barStartTime - org.creationTime) / barDur;
      if (ageInBars < config.qualificationFraction) continue;
    }
    let group = bySpecies.get(org.colorSignature);
    if (!group) { group = []; bySpecies.set(org.colorSignature, group); }
    group.push(org);
  }

  const results: SpeciesSlotResult[] = [];

  for (const [signature, speciesOrganisms] of bySpecies) {
    // Find oldest organism in species (for pitch derivation)
    let oldest = speciesOrganisms[0];
    for (let i = 1; i < speciesOrganisms.length; i++) {
      if (speciesOrganisms[i].creationTime < oldest.creationTime) {
        oldest = speciesOrganisms[i];
      }
    }

    // Collect all typeIds present across all organisms in this species
    const allTypeIds = new Set<number>();
    for (const org of speciesOrganisms) {
      for (const [typeId] of org.composition) allTypeIds.add(typeId);
    }

    // For each type, find max organelle count across all organisms
    const maxCountByType = new Map<number, number>();
    for (const typeId of allTypeIds) {
      let maxCount = 0;
      for (const org of speciesOrganisms) {
        const count = org.composition.get(typeId) ?? 0;
        if (count > maxCount) maxCount = count;
      }
      maxCountByType.set(typeId, maxCount);
    }

    // Rank types by average particle count within the species —
    // rank 0 (most particles, lowest pitch band) is the rhythmic anchor
    // and the only layer allowed to sound on the downbeat.
    const avgByType = new Map<number, number>();
    for (const typeId of allTypeIds) {
      let total = 0, count = 0;
      for (const org of speciesOrganisms) {
        for (const o of org.organelles) {
          if (o.typeId === typeId) { total += o.particleCount; count++; }
        }
      }
      avgByType.set(typeId, count > 0 ? total / count : 0);
    }
    const rankedTypes = [...allTypeIds].sort((a, b) =>
      (avgByType.get(b)! - avgByType.get(a)!) || a - b,
    );

    // Widest grid in the species — lone-organelle non-anchor layers are
    // promoted onto it so they don't all collapse onto tier 0's only slot.
    let widestGrid = 1;
    for (const typeId of allTypeIds) {
      const g = Math.min(maxCountByType.get(typeId)!, MAX_SUBDIVISION);
      if (g > widestGrid) widestGrid = g;
    }

    for (let rank = 0; rank < rankedTypes.length; rank++) {
      const typeId = rankedTypes[rank];
      const maxCount = maxCountByType.get(typeId)!;
      if (maxCount === 0) continue;
      const isAnchor = rank === 0;

      const subdivision = Math.min(maxCount, MAX_SUBDIVISION);
      const grid = !isAnchor && subdivision === 1
        ? Math.max(2, widestGrid)
        : subdivision;
      const tierIndex = grid - 1;

      // Pitch source: oldest organism's individual organelles of this type.
      // Each organelle's crossTypeLinks → harmonic phase → pitch.
      const oldestOrganelles = oldest.organelles.filter(o => o.typeId === typeId);
      const typeKey = ctx.typeKeys[typeId];
      const typeRoot = ctx.typeRoots.get(typeKey ?? "") ?? 0;
      const relativeInterval = ((typeRoot - ctx.rootSemitone) % 12 + 12) % 12;

      // Onset count from connectivity: organelles bonded to more types
      // fire more often. Non-anchor layers cap at grid − 1 so a rotation
      // that clears the downbeat always exists.
      let maxLinks = 0;
      for (const o of oldestOrganelles) {
        if (o.crossTypeLinks > maxLinks) maxLinks = o.crossTypeLinks;
      }
      const kCap = isAnchor ? grid : Math.max(1, grid - 1);
      const k = Math.max(1, Math.min(1 + maxLinks, kCap));

      const base = euclideanOnsets(k, grid);
      const rotation = isAnchor ? 0 : rotationOffDownbeat(base, grid, rank);
      const slotIndices = base
        .map(s => (s + rotation) % grid)
        .sort((a, b) => a - b);

      const slots: { slotIndex: number; note: SlotNote }[] = [];
      for (let i = 0; i < slotIndices.length; i++) {
        const s = slotIndices[i];
        // Pick organelle for this onset (round-robin if oldest has fewer)
        const organelle = oldestOrganelles.length > 0
          ? oldestOrganelles[i % oldestOrganelles.length]
          : null;

        const crossLinks = organelle ? organelle.crossTypeLinks : 0;

        const harmonicPhase = crossTypeLinksToPhase(crossLinks);
        const collapsed = collapseToPhase(relativeInterval, harmonicPhase);

        // Arpeggiate across the layer's onsets in temporal order:
        // same-type organelles usually share crossTypeLinks, which would
        // make every onset the identical pitch. Offsetting by onset
        // (root→fifth→third→octave, then snapped to the active scale
        // below) turns a monotone layer into a contour.
        const arpOffset = ARPEGGIO_OFFSETS[i % ARPEGGIO_OFFSETS.length];

        let midiNote = ctx.rootMidi + collapsed + arpOffset;
        // Clamp to per-type 2-octave pitch range (driven by particle count)
        const pitchRange = ctx.typePitchRanges.get(typeId);
        if (pitchRange) {
          while (midiNote < pitchRange.lowMidi) midiNote += 12;
          while (midiNote > pitchRange.highMidi) midiNote -= 12;
        }

        // Filter against active pitch classes
        const pc = ((midiNote % 12) + 12) % 12;
        if (!ctx.activePitchClasses.has(pc)) {
          let bestDist = 12, bestPC = pc;
          for (const available of ctx.activePitchClasses) {
            const dist = Math.min(
              ((pc - available) % 12 + 12) % 12,
              ((available - pc) % 12 + 12) % 12,
            );
            if (dist < bestDist) { bestDist = dist; bestPC = available; }
          }
          const adjustment = bestPC - pc;
          midiNote += adjustment > 6 ? adjustment - 12 : adjustment < -6 ? adjustment + 12 : adjustment;
        }

        slots.push({
          slotIndex: s,
          note: {
            midiNote,
            speciesSignature: signature,
            typeId,
            subdivisionIndex: s,
          },
        });
      }

      results.push({ tierIndex, slots });
    }
  }

  return results;
}

