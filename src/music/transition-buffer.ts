/**
 * Bridge-chord transition system for smooth key/scale changes (§3.6).
 *
 * When a scale or key change produces too much dissonance (≥3 differing
 * pitch classes), a bridge chord is selected that resolves well toward
 * both the outgoing and incoming tonalities.  The bridge chord is chosen
 * by scoring candidates on voice-leading smoothness, functional gravity
 * (dominant → tonic pull), and common-tone continuity.
 *
 * Pure functions — no side effects.
 */

import type { ModeDefinition, TransitionChord } from "./types";

/* ------------------------------------------------------------------ */
/*  Dissonance threshold                                               */
/* ------------------------------------------------------------------ */

export const DISSONANCE_THRESHOLD = 3;

/* ------------------------------------------------------------------ */
/*  Chord quality definitions                                          */
/* ------------------------------------------------------------------ */

/** Chord represented as a set of semitone offsets from its root. */
interface ChordTemplate {
  readonly name: string;
  readonly offsets: readonly number[];
}

const MAJOR: ChordTemplate   = { name: "maj",  offsets: [0, 4, 7] };
const MINOR: ChordTemplate   = { name: "min",  offsets: [0, 3, 7] };
const DOM7: ChordTemplate    = { name: "dom7", offsets: [0, 4, 7, 10] };
const DIM: ChordTemplate     = { name: "dim",  offsets: [0, 3, 6] };

const CHORD_TEMPLATES: readonly ChordTemplate[] = [MAJOR, MINOR, DOM7, DIM];

const NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"] as const;

/** A concrete candidate chord: root pitch class + quality + absolute pitch classes. */
interface CandidateChord {
  readonly root: number;
  readonly template: ChordTemplate;
  readonly pitchClasses: ReadonlySet<number>;
}

/** Build all candidate chords (48 total: 12 roots × 4 qualities). */
function buildCandidates(): readonly CandidateChord[] {
  const out: CandidateChord[] = [];
  for (let root = 0; root < 12; root++) {
    for (const template of CHORD_TEMPLATES) {
      out.push({
        root,
        template,
        pitchClasses: new Set(template.offsets.map(o => (root + o) % 12)),
      });
    }
  }
  return out;
}

const ALL_CANDIDATES: readonly CandidateChord[] = buildCandidates();

/* ------------------------------------------------------------------ */
/*  Pitch class set operations                                         */
/* ------------------------------------------------------------------ */

/** Get the absolute pitch classes for a mode rooted at a given semitone. */
export function pitchClassSet(mode: ModeDefinition, rootSemitone: number): ReadonlySet<number> {
  return new Set(mode.scaleSemitones.map(s => (s + rootSemitone) % 12));
}

/** Count of pitch classes in A that are not in B, plus those in B not in A. */
export function dissonanceScore(a: ReadonlySet<number>, b: ReadonlySet<number>): number {
  let diff = 0;
  for (const pc of a) if (!b.has(pc)) diff++;
  for (const pc of b) if (!a.has(pc)) diff++;
  return diff;
}

/** Intersection of two pitch class sets. */
export function intersectPitchClasses(
  a: ReadonlySet<number>,
  b: ReadonlySet<number>,
): ReadonlySet<number> {
  const result = new Set<number>();
  for (const pc of a) {
    if (b.has(pc)) result.add(pc);
  }
  return result;
}

/* ------------------------------------------------------------------ */
/*  Root triad extraction                                              */
/* ------------------------------------------------------------------ */

/**
 * Extract the root triad (1-3-5) from a mode at a given root semitone.
 * Returns absolute pitch classes for the triad.
 */
function rootTriad(mode: ModeDefinition, rootSemitone: number): ReadonlySet<number> {
  const scale = mode.scaleSemitones;
  // Degree 0 (root), degree 2 (third), degree 4 (fifth)
  const pcs = new Set<number>();
  pcs.add((rootSemitone + scale[0]) % 12);
  if (scale.length > 2) pcs.add((rootSemitone + scale[2]) % 12);
  if (scale.length > 4) pcs.add((rootSemitone + scale[4]) % 12);
  return pcs;
}

/* ------------------------------------------------------------------ */
/*  Scoring functions                                                  */
/* ------------------------------------------------------------------ */

/**
 * Voice-leading distance between two pitch class sets.
 * Sum of minimum semitone movements to get from each note in `from`
 * to the nearest note in `to`.  Lower = smoother.
 */
function voiceLeadingDistance(from: ReadonlySet<number>, to: ReadonlySet<number>): number {
  if (to.size === 0) return 12 * from.size;
  let total = 0;
  for (const pc of from) {
    let minDist = 12;
    for (const target of to) {
      const dist = Math.min(
        ((pc - target) % 12 + 12) % 12,
        ((target - pc) % 12 + 12) % 12,
      );
      if (dist < minDist) minDist = dist;
    }
    total += minDist;
  }
  return total;
}

/**
 * Count of common tones between two pitch class sets.
 */
function commonToneCount(a: ReadonlySet<number>, b: ReadonlySet<number>): number {
  let count = 0;
  for (const pc of a) if (b.has(pc)) count++;
  return count;
}

/**
 * Functional gravity bonus: does `candidate` act as a dominant of `target`?
 *
 * Returns a score 0–1:
 * - 1.0 if candidate root is a perfect 5th above target root (V → I)
 * - 0.7 if candidate root is a half-step above target root (tritone sub, bII → I)
 * - 0.4 if candidate is a diminished chord and its root is a half-step below target (vii° → I)
 * - 0.0 otherwise
 */
function functionalGravity(candidate: CandidateChord, targetRoot: number): number {
  const interval = ((candidate.root - targetRoot) % 12 + 12) % 12;

  // V → I: root a 5th above (interval = 7 semitones)
  if (interval === 7) {
    // Extra bonus for dom7 quality (strongest pull)
    return candidate.template === DOM7 ? 1.0 : 0.8;
  }
  // Tritone substitution: bII → I (interval = 1)
  if (interval === 1 && (candidate.template === DOM7 || candidate.template === MAJOR)) {
    return 0.7;
  }
  // Leading-tone diminished: vii° → I (interval = 11, half-step below)
  if (interval === 11 && candidate.template === DIM) {
    return 0.4;
  }
  // iv → I (subdominant, interval = 5) — mild pull
  if (interval === 5) {
    return 0.2;
  }
  return 0;
}

/**
 * Score a candidate bridge chord for how well it connects the outgoing
 * chord/scale to the incoming chord/scale.
 *
 * Higher score = better bridge.
 */
function scoreBridgeChord(
  candidate: CandidateChord,
  outgoingTriad: ReadonlySet<number>,
  outgoingScale: ReadonlySet<number>,
  incomingTriad: ReadonlySet<number>,
  incomingScale: ReadonlySet<number>,
  incomingRoot: number,
): number {
  const cpc = candidate.pitchClasses;

  // Voice-leading smoothness (lower distance = better, normalize to 0–1 score)
  // Normalize per-note (max 6 semitones each) so triads and 7th chords
  // are scored on the same scale.
  const vlFromOutgoing = voiceLeadingDistance(outgoingTriad, cpc);
  const vlToIncoming = voiceLeadingDistance(cpc, incomingTriad);
  const avgVL = (vlFromOutgoing / outgoingTriad.size + vlToIncoming / cpc.size) / 2;
  const vlScore = 1 - avgVL / 6;

  // Common tones with both neighbors
  const ctOutgoing = commonToneCount(cpc, outgoingScale);
  const ctIncoming = commonToneCount(cpc, incomingScale);
  const maxCT = candidate.template.offsets.length;
  const ctScore = (ctOutgoing + ctIncoming) / (2 * maxCT);

  // Functional gravity toward incoming key (dominant pull into target)
  const funcScore = functionalGravity(candidate, incomingRoot);

  // Weighted combination
  return (
    vlScore   * 0.40 +
    funcScore * 0.35 +
    ctScore   * 0.25
  );
}

/* ------------------------------------------------------------------ */
/*  Bridge chord selection                                             */
/* ------------------------------------------------------------------ */

/**
 * Select the best bridge chord to connect two tonalities.
 *
 * Returns the pitch class set of the winning chord.
 */
function selectBridgeChord(
  prevMode: ModeDefinition,
  prevRoot: number,
  nextMode: ModeDefinition,
  nextRoot: number,
): TransitionChord {
  const outgoingTriad = rootTriad(prevMode, prevRoot);
  const outgoingScale = pitchClassSet(prevMode, prevRoot);
  const incomingTriad = rootTriad(nextMode, nextRoot);
  const incomingScale = pitchClassSet(nextMode, nextRoot);

  let bestScore = -Infinity;
  let bestChord: CandidateChord = ALL_CANDIDATES[0];

  for (const candidate of ALL_CANDIDATES) {
    const score = scoreBridgeChord(
      candidate,
      outgoingTriad,
      outgoingScale,
      incomingTriad,
      incomingScale,
      nextRoot,
    );
    if (score > bestScore) {
      bestScore = score;
      bestChord = candidate;
    }
  }

  return {
    name: `${NOTE_NAMES[bestChord.root]} ${bestChord.template.name}`,
    pitchClasses: bestChord.pitchClasses,
  };
}

/* ------------------------------------------------------------------ */
/*  Buffer decision (public API — signature unchanged)                 */
/* ------------------------------------------------------------------ */

/**
 * Determine whether a transition buffer bar is needed and compute its pitch set.
 *
 * When a buffer is needed, selects a bridge chord that resolves well toward
 * both the outgoing and incoming tonalities, rather than simply intersecting
 * the two scales.
 *
 * @param prevMode       - Outgoing mode
 * @param prevRoot       - Outgoing root semitone (0–11)
 * @param nextMode       - Incoming mode
 * @param nextRoot       - Incoming root semitone (0–11)
 * @param currentBuffer  - Pitch set of current buffer bar (if already in a buffer)
 * @returns null if no buffer needed, otherwise the bridge chord's pitch class set
 */
export function computeTransitionBuffer(
  prevMode: ModeDefinition,
  prevRoot: number,
  nextMode: ModeDefinition,
  nextRoot: number,
  currentBuffer: TransitionChord | null = null,
): TransitionChord | null {
  const prevFull = pitchClassSet(prevMode, prevRoot);
  const incoming = pitchClassSet(nextMode, nextRoot);

  // Always compare full scales to decide if a buffer is needed.
  const score = dissonanceScore(prevFull, incoming);
  if (score < DISSONANCE_THRESHOLD) return null;

  // If already in a buffer bar, select a new bridge chord that connects
  // the *original* outgoing tonality to the incoming one.  The prevMode/prevRoot
  // still represent the pre-transition state (threaded via MusicState), so
  // the bridge chord is always chosen with full context of both endpoints.
  return selectBridgeChord(prevMode, prevRoot, nextMode, nextRoot);
}
