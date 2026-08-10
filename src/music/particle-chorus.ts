/**
 * Particle chorus drones — a background wash of sine voices spanning the
 * entire frequency range, clustered on the current harmony.
 *
 * Every particle conceptually owns one equal subdivision of the range,
 * so the chorus reflects the population: few particles → a sparse wash;
 * many → thick clusters. Raw subdivision centers are NOT played directly
 * (evenly tiling the spectrum with sines is indistinguishable from
 * noise) — each voice snaps to the nearest chord tone, keeping a small
 * fraction of its offset as micro-detune. The result is an ensemble
 * chorus: several ±10–30-cent voices per chord tone per octave, denser
 * with more particles, gliding between chord tones as the harmony moves.
 *
 * Lower voices get higher gain via a pink-noise-style dB/octave tilt —
 * high sines cut through disproportionately, so the top of the range is
 * shelved down to keep the wash warm and even.
 *
 * Pure functions only — the imperative voice manager lives in
 * audio-graph.ts (updateDronePad).
 */

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface DroneVoice {
  readonly key: string;
  /** Fractional MIDI — chorus subdivisions are deliberately microtonal. */
  readonly midiNote: number;
  /** Relative voice level 0–1 (pad master level applied separately). */
  readonly gain: number;
}

/** Full range the chorus spans. */
const CHORUS_LOW_MIDI = 24;   // C1 (~32.7 Hz)
const CHORUS_HIGH_MIDI = 96;  // C7 (~2093 Hz)

/** Max realized oscillators — beyond this the range is saturated. */
const MAX_CHORUS_VOICES = 96;

/** Spectral tilt: gain falls by this many dB per octave above the low
 *  end (≈ pink noise), compensating for how easily high notes cut. */
const TILT_DB_PER_OCTAVE = 3;

/** Fraction of a voice's distance-to-chord-tone kept as micro-detune.
 *  Chord tones sit ~3–4 semitones apart, so residuals reach ±2 st;
 *  retaining 15% yields the ±10–30 cents of a real ensemble chorus. */
const DETUNE_RETAIN = 0.15;

/** Nearest MIDI position (any octave) whose pitch class is in `pcs`. */
function snapToChord(midi: number, pcs: ReadonlySet<number>): number {
  let best = midi;
  let bestDist = Infinity;
  for (const pc of pcs) {
    const candidate = pc + 12 * Math.round((midi - pc) / 12);
    const dist = Math.abs(midi - candidate);
    if (dist < bestDist) {
      bestDist = dist;
      best = candidate;
    }
  }
  return best;
}

/* ------------------------------------------------------------------ */
/*  Voice computation                                                  */
/* ------------------------------------------------------------------ */

/**
 * Compute the chorus voicing for the current particle population and
 * bar chord. Voice i starts from the center of subdivision i of the
 * range, then clusters onto the nearest chord tone with micro-detune;
 * keys are stable by index so existing oscillators glide when the
 * count or harmony changes.
 */
export function computeDroneVoices(
  totalParticleCount: number,
  chordPitchClasses: ReadonlySet<number> | null,
): readonly DroneVoice[] {
  const count = Math.max(0, Math.floor(totalParticleCount));
  const v = Math.min(count, MAX_CHORUS_VOICES);
  if (v === 0) return [];

  const span = CHORUS_HIGH_MIDI - CHORUS_LOW_MIDI;
  const voices: DroneVoice[] = [];

  for (let i = 0; i < v; i++) {
    const raw = CHORUS_LOW_MIDI + ((i + 0.5) / v) * span;

    let midiNote = raw;
    if (chordPitchClasses && chordPitchClasses.size > 0) {
      const snapped = snapToChord(raw, chordPitchClasses);
      midiNote = snapped + (raw - snapped) * DETUNE_RETAIN;
    }

    const octavesAboveLow = (midiNote - CHORUS_LOW_MIDI) / 12;
    const gain = Math.pow(10, (-TILT_DB_PER_OCTAVE * octavesAboveLow) / 20);
    voices.push({ key: `chorus-${i}`, midiNote, gain });
  }

  return voices;
}
