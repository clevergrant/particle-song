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

export interface SigmoidConfig {
  readonly midpoint: number;
  readonly steepness: number;
}

export interface StabilityConfig {
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

export interface RegisterWidth {
  readonly lowOctave: number;
  readonly highOctave: number;
}

/* ------------------------------------------------------------------ */
/*  Scheduled Hit                                                      */
/* ------------------------------------------------------------------ */

export interface ScheduledHit {
  readonly time: number;             // AudioContext time
  readonly organismId: number;
  readonly typeId: number;
  readonly organelleIndex: number;   // which organelle within this type's subdivision
  readonly midiNote: number;
  readonly volume: number;           // 0–1
  readonly pan: number;              // -1 (left) to +1 (right)
  readonly filterCutoff: number;     // Hz
  readonly envelope: EnvelopeParams;
  readonly noteDuration: NoteDuration;
  readonly gateDuration: number;     // seconds — how long the note is held before release
  readonly waveform: WaveformParams;
  readonly vibratoDepth: number;     // cents (0–100)
}

/* ------------------------------------------------------------------ */
/*  Scheduled Bar                                                      */
/* ------------------------------------------------------------------ */

/** A pitch in the bass arpeggio pool, ordered by free particle abundance. */
export interface ArpNote {
  /** MIDI note to play. */
  readonly midiNote: number;
  /** Sociability of this type (for waveform selection). */
  readonly sociability: number;
  /** Free particle percentage for this type (for volume). */
  readonly freePercent: number;
}

export interface BassUpdate {
  readonly root: number;             // MIDI note (base root from oldest species)
  readonly fifth: number;            // MIDI note
  readonly mode: ModeDefinition;
  readonly freeParticlePercentByType: ReadonlyMap<number, number>;
  readonly isQuartalStack: boolean;  // true when no organisms exist
  /** Ordered pitch pool for bass arpeggiator (most abundant type first). */
  readonly arpNotes: readonly ArpNote[];
  /** Simulation stability (0 = chaos, 1 = order) for interval selection. */
  readonly netStability: number;
  /** Global average particle velocity — drives bass staccato (higher = more staccato). */
  readonly avgVelocity: number;
}

export interface TransitionChord {
  readonly name: string;              // e.g. "G dom7"
  readonly pitchClasses: ReadonlySet<number>;
}

export interface ScheduledBar {
  readonly barNumber: number;
  readonly startTime: number;        // AudioContext time
  readonly duration: number;         // seconds
  readonly hits: readonly ScheduledHit[];
  readonly mode: ModeDefinition;
  readonly rootMidi: number;
  readonly isBufferBar: boolean;
  readonly bufferChord: TransitionChord | null;
  readonly bassUpdate: BassUpdate;
  readonly netStability: number;
  readonly spatialEntropy: number;
  readonly envelopeRanges: EnvelopeRanges;
  readonly speciesCycle: SpeciesCycle;
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
}

export interface BarSnapshot {
  readonly organisms: readonly SnapshotOrganism[];
  readonly globalMetrics: GlobalMetrics;
  readonly forceMatrix: ForceMatrix;
  readonly typeKeys: readonly string[];
  readonly canvasWidth: number;
  readonly beatsPerBar: number;
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
  readonly prevScheduledBar: ScheduledBar | null;
  readonly isBufferBar: boolean;
  readonly bufferChord: TransitionChord | null;
  readonly envelopeRanges: EnvelopeRanges | null;
  readonly speciesCycle: SpeciesCycle;
}
