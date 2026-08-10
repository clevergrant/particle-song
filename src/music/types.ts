/**
 * Shared readonly types for the music subsystem.
 *
 * Every type here is immutable. State is threaded through the simulation
 * as prev → next snapshots; nothing in this file is ever mutated.
 */

import type { ForceMatrix } from "../particles/particle";

/* ------------------------------------------------------------------ */
/*  Scale / Mode                                                       */
/* ------------------------------------------------------------------ */

export interface ModeDefinition {
  readonly name: string;
  readonly scaleSemitones: readonly number[];
}

/* ------------------------------------------------------------------ */
/*  Global Metrics                                                     */
/* ------------------------------------------------------------------ */

export type SimEventKind =
  | "organelle-formed"
  | "organelle-dissolved"
  | "organism-formed"
  | "organism-dissolved"
  | "organelle-joined"
  | "organelle-left";

export interface SimEvent {
  readonly kind: SimEventKind;
  readonly id: number;
  readonly typeId?: number;
  readonly signature?: string;
}

export interface GlobalMetrics {
  readonly freeParticleCount: number;
  readonly freeParticlePercentByType: ReadonlyMap<number, number>;
  readonly avgVelocity: number;
  readonly avgOrganelleDensity: number;
  readonly speciesCount: number;
  readonly organismCount: number;
  readonly spatialEntropy: number;
  readonly events: readonly SimEvent[];
  /** Fraction of organism-capable particles currently in organisms [0, 1]. */
  readonly organismFulfillment: number;
}

/* ------------------------------------------------------------------ */
/*  Stability                                                          */
/* ------------------------------------------------------------------ */

interface SigmoidConfig {
  readonly midpoint: number;
  readonly steepness: number;
}

interface StabilityConfig {
  readonly speciesDiversity: SigmoidConfig;
  readonly inverseVelocity: SigmoidConfig;
  readonly density: SigmoidConfig;
}

/* ------------------------------------------------------------------ */
/*  Waveform / Timbre                                                  */
/* ------------------------------------------------------------------ */

/** 0 = sawtooth, 0.33 = square, 0.66 = triangle, 1.0 = sine */
export interface WaveformParams {
  readonly sociability: number;
  readonly blend: number;
}

/* ------------------------------------------------------------------ */
/*  Envelope                                                           */
/* ------------------------------------------------------------------ */

/**
 * Curve shape for envelope segments (§8.3).
 * - "linear"      — straight line ramp
 * - "exponential"  — setTargetAtTime (fast start, slow tail or vice versa)
 * - "ease-in"      — slow start, fast finish (quadratic-ish via exponential τ)
 * - "ease-out"     — fast start, slow finish
 */
export type EnvelopeCurve = "linear" | "exponential" | "ease-in" | "ease-out";

export interface EnvelopeParams {
  readonly attackDuration: number;   // seconds — from centroid speed
  readonly attackCurve: EnvelopeCurve;
  readonly peakLevel: number;        // 0–1 — attack→decay boundary height
  readonly decayDuration: number;    // seconds — from density
  readonly decayCurve: EnvelopeCurve;
  readonly sustainLevel: number;     // 0–1 — decay→release boundary height (horizontal line)
  readonly releaseDuration: number;  // seconds — from spatial radius
  readonly releaseCurve: EnvelopeCurve;
}

/* ------------------------------------------------------------------ */
/*  Envelope Shape (user-editable bezier envelope)                     */
/* ------------------------------------------------------------------ */

/** A bezier control point for envelope curves. */
export interface EnvelopeNode {
  readonly x: number;  // 0–1 within section
  readonly y: number;  // 0–1 (amplitude)
  readonly handleInDx: number;
  readonly handleInDy: number;
  readonly handleOutDx: number;
  readonly handleOutDy: number;
}

/** One section of the envelope (attack, decay, or release). */
export interface EnvelopeSection {
  /** Proportion of total duration this section occupies (0–1). */
  readonly proportion: number;
  /** Bezier nodes in local [0,1]×[0,1] space. Sorted by x ascending. */
  readonly nodes: readonly EnvelopeNode[];
}

/**
 * User-editable ADSR envelope shape.
 * Attack: bezier curve, 0 → peakLevel.
 * Decay: bezier curve, peakLevel → sustainLevel.
 * Sustain: flat horizontal line at sustainLevel (duration is gate-driven).
 * Release: bezier curve, sustainLevel → 0.
 */
export interface EnvelopeShape {
  readonly attack: EnvelopeSection;
  readonly decay: EnvelopeSection;
  readonly sustainLevel: number;      // 0–1, flat line height
  readonly release: EnvelopeSection;
}

/** Musical note duration for gate-based envelope. */
export type NoteDuration = "whole" | "half" | "quarter" | "eighth" | "sixteenth";

/* ------------------------------------------------------------------ */
/*  Overtone Phase                                                     */
/* ------------------------------------------------------------------ */

/**
 * Phase 1 = fundamental only, through Phase 6 = upper partials.
 * Each phase unlocks additional intervals from the overtone series.
 */
export interface HarmonicPhase {
  readonly phase: number;                           // 1–6
  readonly availableIntervals: readonly number[];   // semitone offsets from root that are active
}

/* ------------------------------------------------------------------ */
/*  Tuplet Grid (bar-level note scheduling)                            */
/* ------------------------------------------------------------------ */

/** Minimal note data produced by the worker — just pitch + identity.
 *  Playback parameters (volume, envelope, etc.) are computed at play
 *  time on the main thread from live simulation state.
 *  Species-level: one voice per organelle type per species. */
export interface SlotNote {
  readonly midiNote: number;
  readonly speciesSignature: string;   // colorSignature — species identity
  readonly typeId: number;
  readonly subdivisionIndex: number;   // slot within tuplet (0..maxCount-1)
}

/** Pre-allocated grid: tiers[tierIndex][slotIndex] = notes for that slot.
 *  tierIndex = subdivision count − 1 (tier 0 = whole note, tier 2 = triplets, etc.)
 *  Each tier has (tierIndex + 1) evenly-spaced slots across the bar. */
export interface TupletGrid {
  readonly tiers: (SlotNote[] | null)[][];
}

/** Maximum number of subdivision tiers (= max organelles per organism). */
export const MAX_SUBDIVISION = 16;

/** Allocate an empty grid with all slots set to null. */
export function createTupletGrid(): TupletGrid {
  const tiers: (SlotNote[] | null)[][] = [];
  for (let t = 0; t < MAX_SUBDIVISION; t++) {
    const slots: (SlotNote[] | null)[] = [];
    for (let s = 0; s <= t; s++) slots.push(null);
    tiers.push(slots);
  }
  return { tiers };
}

/* ------------------------------------------------------------------ */
/*  Worker streaming messages                                          */
/* ------------------------------------------------------------------ */

/** Bar-level metadata — sent immediately when the worker starts. */
export interface ScheduleWorkerBarMeta {
  readonly kind: "bar-meta";
  readonly id: number;
  readonly barNumber: number;
  readonly mode: ModeDefinition;
  readonly rootMidi: number;
  readonly isBufferBar: boolean;
  readonly bufferChord: TransitionChord | null;
  readonly netStability: number;
  readonly spatialEntropy: number;
  readonly envelopeRanges: EnvelopeRanges;
  readonly speciesCycle: SpeciesCycle;
  /** This bar's chord pitch classes (diatonic triad, or buffer chord). */
  readonly chordPitchClasses: ReadonlySet<number>;
}

/** Per-organism slot fill — streamed as each organism is computed. */
export interface ScheduleWorkerSlotFill {
  readonly kind: "slot-fill";
  readonly id: number;
  readonly tierIndex: number;
  readonly slotIndex: number;
  readonly notes: readonly SlotNote[];
}

/** Signals that all organisms have been processed for this bar. */
export interface ScheduleWorkerDone {
  readonly kind: "done";
  readonly id: number;
}

/** Discriminated union of all worker→main messages. */
export type ScheduleWorkerMsg =
  | ScheduleWorkerBarMeta
  | ScheduleWorkerSlotFill
  | ScheduleWorkerDone;

/* ------------------------------------------------------------------ */
/*  Scheduled Bar                                                      */
/* ------------------------------------------------------------------ */

export interface TransitionChord {
  readonly name: string;              // e.g. "G dom7"
  readonly pitchClasses: ReadonlySet<number>;
}

/* ------------------------------------------------------------------ */
/*  Bar Snapshot (input to the scheduler)                               */
/* ------------------------------------------------------------------ */

export interface SnapshotOrganelle {
  readonly id: number;
  readonly typeId: number;
  readonly particleCount: number;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly centroidSpeed: number;    // scalar velocity of centroid
  readonly density: number;          // particleCount / area
  readonly spatialRadius: number;    // bounding radius or variance
  readonly angularOffset: number;    // angle from organism velocity vector (for visual only)
  readonly crossTypeLinks: number;   // number of distinct types directly bonded to
}

export interface SnapshotOrganism {
  readonly registryId: number;
  readonly colorSignature: string;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly velX: number;
  readonly velY: number;
  readonly creationTime: number;     // timestamp at formation
  readonly organelles: readonly SnapshotOrganelle[];
  readonly composition: ReadonlyMap<number, number>;  // typeId → count
  /** Type-adjacency fingerprint (cross-type edges from organelle tree). Wire format: number[]. */
  readonly typeAdjacency: ReadonlySet<number>;
}

export interface BarSnapshot {
  readonly organisms: readonly SnapshotOrganism[];
  readonly globalMetrics: GlobalMetrics;
  readonly forceMatrix: ForceMatrix;
  readonly typeKeys: readonly string[];
  readonly canvasWidth: number;
  /** If set, overrides the root derivation in computeBarContext (semitone 0-11). */
  readonly rootOverride: number | null;
  /** If set, only this species plays during this bar (organism-driven cycle). */
  readonly activeSpecies: string | null;
  /** Diatonic degree (0 = I) of this bar's chord within the current key. */
  readonly chordDegree: number;
}

/* ------------------------------------------------------------------ */
/*  Music State (threaded through update loop)                         */
/* ------------------------------------------------------------------ */

/**
 * Exponentially-smoothed min/max ranges for the four organelle properties
 * that drive envelope shape (staccato ↔ sustained).  Threaded between bars
 * so that normalization always produces a spread even when the current bar's
 * organelles are homogeneous.
 */
export interface EnvelopeRanges {
  readonly particleCountMin: number;
  readonly particleCountMax: number;
  readonly speedMin: number;
  readonly speedMax: number;
  readonly densityMin: number;
  readonly densityMax: number;
  readonly radiusMin: number;
  readonly radiusMax: number;
}

/**
 * Per-species round-robin cycle state.
 * Tracks which organism IDs have played so far within each species.
 * Once all have played, the set resets.
 */
export interface SpeciesCycle {
  /** Species signature → set of organism registryIds that have already played. */
  readonly played: ReadonlyMap<string, ReadonlySet<number>>;
}

export interface MusicState {
  readonly currentBarNumber: number;
  readonly currentMode: ModeDefinition;
  readonly currentRootMidi: number;
  readonly netStability: number;
  readonly isBufferBar: boolean;
  readonly bufferChord: TransitionChord | null;
  readonly envelopeRanges: EnvelopeRanges | null;
  readonly speciesCycle: SpeciesCycle;
  readonly organismCycleNumber: number;
}
