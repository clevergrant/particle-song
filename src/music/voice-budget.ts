/**
 * Voice budget — pure culling layer between scheduleBar() and playScheduledBar().
 *
 * When the simulation produces more voices than the browser can handle,
 * this module scores each hit by musical importance and keeps only the
 * top N within a configurable budget.  The scheduler remains unconstrained;
 * the audio graph remains unchanged; this is a clean functional filter.
 */

import type { ScheduledBar, ScheduledHit } from "./types";

/* ------------------------------------------------------------------ */
/*  Scoring weights                                                    */
/* ------------------------------------------------------------------ */

const W_BEAT   = 0.30;
const W_VOLUME = 0.20;
const W_ORG    = 0.25;
const W_PITCH  = 0.15;
const W_TIMBRE = 0.10;

/** Fraction of budget slots at the bottom that get volume attenuation. */
const FRINGE_FRACTION = 0.20;
/** Minimum volume multiplier for the lowest fringe voice. */
const FRINGE_MIN_VOLUME = 0.5;

/* ------------------------------------------------------------------ */
/*  Scoring helpers                                                    */
/* ------------------------------------------------------------------ */

/** Count how many hits share each pitch class (0–11). */
function pitchClassCounts(hits: readonly ScheduledHit[]): ReadonlyMap<number, number> {
  const counts = new Map<number, number>();
  for (const h of hits) {
    const pc = ((h.midiNote % 12) + 12) % 12;
    counts.set(pc, (counts.get(pc) ?? 0) + 1);
  }
  return counts;
}

/** Count how many hits share each waveform blend value (quantised to 2 decimals). */
function timbreCounts(hits: readonly ScheduledHit[]): ReadonlyMap<number, number> {
  const counts = new Map<number, number>();
  for (const h of hits) {
    const key = Math.round(h.waveform.blend * 100);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

/** Find the maximum organelleIndex across all hits (for beat-position normalisation). */
function maxSubdivision(hits: readonly ScheduledHit[]): number {
  let max = 1;
  for (const h of hits) {
    if (h.organelleIndex + 1 > max) max = h.organelleIndex + 1;
  }
  return max;
}

/* ------------------------------------------------------------------ */
/*  Core scoring                                                       */
/* ------------------------------------------------------------------ */

interface ScoredHit {
  readonly index: number;
  readonly organismId: number;
  score: number;  // mutable only during the two-pass scoring
}

/**
 * Score all hits and return scored entries sorted by descending score.
 *
 * Two-pass:
 * 1. Compute base score (beat + volume + pitch + timbre) for every hit.
 * 2. For each organism, add the organism-representation bonus to that
 *    organism's highest-scoring hit, guaranteeing it survives the cut.
 */
function scoreHits(hits: readonly ScheduledHit[]): ScoredHit[] {
  if (hits.length === 0) return [];

  const pcCounts = pitchClassCounts(hits);
  const tCounts  = timbreCounts(hits);
  const maxSub   = maxSubdivision(hits);

  // Pass 1: base scores
  const scored: ScoredHit[] = hits.map((h, i) => {
    const beatScore   = 1 - h.organelleIndex / maxSub;
    const volumeScore = h.volume;
    const pc          = ((h.midiNote % 12) + 12) % 12;
    const pitchScore  = 1 / (pcCounts.get(pc) ?? 1);
    const timbreKey   = Math.round(h.waveform.blend * 100);
    const timbreScore = 1 / (tCounts.get(timbreKey) ?? 1);

    return {
      index: i,
      organismId: h.organismId,
      score: W_BEAT * beatScore
           + W_VOLUME * volumeScore
           + W_PITCH * pitchScore
           + W_TIMBRE * timbreScore,
    };
  });

  // Pass 2: organism-representation bonus
  const bestByOrg = new Map<number, number>(); // organismId → index in scored[]
  for (let i = 0; i < scored.length; i++) {
    const s = scored[i];
    const prev = bestByOrg.get(s.organismId);
    if (prev === undefined || s.score > scored[prev].score) {
      bestByOrg.set(s.organismId, i);
    }
  }
  for (const idx of bestByOrg.values()) {
    scored[idx].score += W_ORG;
  }

  // Sort descending by score
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Apply a voice budget to a scheduled bar.
 *
 * If the bar already fits within the budget, the original object is
 * returned unchanged (zero allocation overhead in the common case).
 *
 * Otherwise, hits are scored by musical importance and the top
 * `melodyBudget` are kept.  Fringe voices (bottom 20% of budget)
 * get their volume attenuated so they fade in/out across bars
 * rather than appearing/disappearing abruptly.
 */
export function applyVoiceBudget(
  bar: ScheduledBar,
  melodyBudget: number,
): ScheduledBar {
  if (bar.hits.length <= melodyBudget) return bar;

  const scored = scoreHits(bar.hits);

  // Keep the top melodyBudget hits
  const kept = scored.slice(0, melodyBudget);

  // Determine the fringe threshold: voices in the bottom FRINGE_FRACTION
  // of the kept set get their volume attenuated.
  const fringeStart = Math.floor(melodyBudget * (1 - FRINGE_FRACTION));

  const culledHits: ScheduledHit[] = kept.map((s, rank) => {
    const original = bar.hits[s.index];
    if (rank < fringeStart) return original;

    // Linear fade from 1.0 at fringeStart to FRINGE_MIN_VOLUME at the end
    const fringePos = (rank - fringeStart) / (melodyBudget - fringeStart);
    const volumeScale = 1 - fringePos * (1 - FRINGE_MIN_VOLUME);
    return { ...original, volume: original.volume * volumeScale };
  });

  return { ...bar, hits: culledHits };
}

