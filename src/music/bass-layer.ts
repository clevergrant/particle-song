/**
 * Bass layer — bass arpeggiator + quartal stack (§9).
 *
 * **Bass arpeggiator** (organisms exist):
 * Free particle types provide an ordered pitch pool (most abundant first).
 * The arpeggiator cycles through these pitches across the bar's beats,
 * creating a bassline whose complexity scales with free particle diversity.
 *
 * **Quartal stack** (no organisms — "orchestra warming up"):
 * Sustained voices stacked in perfect fourths, one per particle type.
 * Dense, ambiguous texture that collapses into the bass arpeggio once
 * the first organism forms.
 */

import type { BassUpdate, ArpNote, EnvelopeShape } from "./types";
import { waveformBlendToTypes, sociabilityGain } from "./timbre";
import { buildGateAwareLUT } from "../envelope-editor";
import { midiToFreq } from "./utils";

/* ------------------------------------------------------------------ */
/*  Bass-idiomatic interval weights                                    */
/* ------------------------------------------------------------------ */

/**
 * Weight per semitone-class interval (0–12).  Higher = more preferred.
 * Reflects bass voice-leading practice: P4/P5 leaps are the most
 * idiomatic bass motion, thirds are common chord-tone leaps, 6ths are
 * moderate, steps serve as approach tones, and tritone/7ths are tense.
 */
const BASS_INTERVAL_WEIGHTS: readonly number[] = [
  /* 0  unison  */ 0,
  /* 1  m2      */ 0.3,
  /* 2  M2      */ 0.4,
  /* 3  m3      */ 0.6,
  /* 4  M3      */ 0.6,
  /* 5  P4      */ 1.0,
  /* 6  tritone */ 0.1,
  /* 7  P5      */ 1.0,
  /* 8  m6      */ 0.5,
  /* 9  M6      */ 0.5,
  /* 10 m7      */ 0.2,
  /* 11 M7      */ 0.15,
  /* 12 octave  */ 0.8,
];

/** Semitone distance beyond which a leap should reverse direction. */
const LEAP_RECOVERY_THRESHOLD = 7; // > P5

/* ------------------------------------------------------------------ */
/*  Bass phrase density                                                */
/* ------------------------------------------------------------------ */

export type BassDensity = "W" | "H" | "Q" | "E";

export const DEFAULT_PHRASE_CELLS: readonly BassDensity[] = [
  "W", "W", "W", "W", "H", "H", "H", "H", "Q", "Q", "E", "E",
];

export function expandPhrase(
  cells: readonly BassDensity[],
  mirror: boolean,
): readonly BassDensity[] {
  if (!mirror || cells.length === 0) return cells;
  return [...cells, ...[...cells].reverse()];
}

function notesForDensity(
  density: BassDensity,
  barDur: number,
  beatsPerBar: number,
): { readonly notesPerBar: number; readonly noteDuration: number } {
  switch (density) {
    case "W": return { notesPerBar: 1, noteDuration: barDur };
    case "H": return { notesPerBar: 2, noteDuration: barDur / 2 };
    case "Q": return { notesPerBar: beatsPerBar, noteDuration: barDur / beatsPerBar };
    case "E": return { notesPerBar: beatsPerBar * 2, noteDuration: barDur / (beatsPerBar * 2) };
  }
}

/* ------------------------------------------------------------------ */
/*  Pure helpers                                                       */
/* ------------------------------------------------------------------ */

/** Build a quartal stack (perfect fourths, 5 semitones apart) from a root. */
function quartalStackPitches(rootMidi: number, count: number): readonly number[] {
  const base = 48 + (rootMidi % 12);          // anchor around octave 3-4
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    out.push(base + i * 5);
  }
  return out;
}

/**
 * Greedy nearest-neighbour voice leading.
 * Returns an assignment array: result[i] = the target MIDI for voice i.
 * Handles count mismatches by truncating or leaving extras unassigned.
 */
function assignVoiceLeading(
  currentMidis: readonly number[],
  targetMidis: readonly number[],
): readonly number[] {
  const used = new Set<number>();
  const assignment: number[] = [];

  // Sort pairs by distance, assign greedily
  const pairs: { vi: number; ti: number; dist: number }[] = [];
  for (let vi = 0; vi < currentMidis.length; vi++) {
    for (let ti = 0; ti < targetMidis.length; ti++) {
      pairs.push({ vi, ti, dist: Math.abs(currentMidis[vi] - targetMidis[ti]) });
    }
  }
  pairs.sort((a, b) => a.dist - b.dist);

  const assignedVoices = new Set<number>();
  for (const { vi, ti } of pairs) {
    if (assignedVoices.has(vi) || used.has(ti)) continue;
    assignment[vi] = targetMidis[ti];
    assignedVoices.add(vi);
    used.add(ti);
  }
  return assignment;
}

/* ------------------------------------------------------------------ */
/*  QuartalVoice — one sustained voice in the stack                    */
/* ------------------------------------------------------------------ */

interface QuartalVoice {
  readonly oscA: OscillatorNode;
  readonly oscB: OscillatorNode;
  readonly gainA: GainNode;
  readonly gainB: GainNode;
  readonly filter: BiquadFilterNode;
  readonly panner: StereoPannerNode;
  readonly envelope: GainNode;
  currentMidi: number;
  readonly typeId: number;
  /** audioContext.currentTime when this voice's slow attack finishes */
  readonly warmupUntil: number;
}

/* ------------------------------------------------------------------ */
/*  BassLayer class                                                   */
/* ------------------------------------------------------------------ */

export class BassLayer {
  private ctx: AudioContext | null = null;
  private output: GainNode | null = null;
  private bassEnvelopeShape: EnvelopeShape | null = null;
  private _staccatoLUT: Float32Array | null = null;

  /* ── Phrase cycle state ──────────────────────────────────────────── */
  private phraseCycleOrigin: number | null = null;
  private prevHadOrganisms = false;

  /* ── Quartal stack state ────────────────────────────────────────── */
  private quartalVoices: QuartalVoice[] = [];
  private quartalActive = false;
  private lastQuartalRoot = -1;
  private _maxQuartalVoices = 8;

  /* ── Lifecycle ───────────────────────────────────────────────────── */

  init(ctx: AudioContext, destination: AudioNode): void {
    this.ctx = ctx;
    this.output = ctx.createGain();
    this.output.gain.value = 0.08;
    this.output.connect(destination);
  }

  dispose(): void {
    this.teardownQuartalVoices();
    this.phraseCycleOrigin = null;
    this.prevHadOrganisms = false;
    this.output?.disconnect();
    this.output = null;
    this.ctx = null;
  }


  /** The bar number at which the current phrase cycle started, or null. */
  get cycleOrigin(): number | null {
    return this.phraseCycleOrigin;
  }

  /** Set the maximum number of quartal stack voices (caps particle types). */
  setMaxQuartalVoices(n: number): void {
    this._maxQuartalVoices = Math.max(1, n);
  }

  /** Set the staccato transfer curve LUT (x = normalized velocity, y = staccato amount 0–1). */
  setStaccatoLUT(lut: Float32Array): void {
    this._staccatoLUT = lut;
  }

  /** Set the bass envelope shape from the envelope editor. */
  setBassEnvelope(shape: EnvelopeShape): void {
    this.bassEnvelopeShape = shape;
  }

  /* ── Bar scheduling ─────────────────────────────────────────────── */

  /**
   * Schedule bass arpeggio for the upcoming bar, or manage quartal
   * stack voices when no organisms exist.
   */
  applyUpdate(
    update: BassUpdate,
    barStart: number,
    barDur: number,
    beatsPerBar: number,
    barNumber: number,
    phraseSequence: readonly BassDensity[],
  ): void {
    if (!this.ctx || !this.output) return;

    // ── Quartal stack mode ────────────────────────────────────────
    if (update.isQuartalStack) {
      this.prevHadOrganisms = false;
      if (!this.quartalActive) {
        this.activateQuartalStack(update, barStart, barDur);
      } else if (update.root !== this.lastQuartalRoot) {
        this.updateQuartalPitches(update, barStart, barDur);
      }
      this.lastQuartalRoot = update.root;
      return;                    // skip beat plucks while stack is active
    }

    // ── Deactivate quartal stack if organisms just appeared ───────
    if (this.quartalActive) {
      this.deactivateQuartalStack(barStart);
    }

    // ── Phrase cycle: reset origin on organism appearance ─────────
    if (!this.prevHadOrganisms) {
      this.phraseCycleOrigin = barNumber;
    }
    this.prevHadOrganisms = true;

    // ── Bass arpeggio with phrase-driven density + interval rules ──
    if (update.arpNotes.length === 0 || phraseSequence.length === 0) return;

    const seqLen = phraseSequence.length;
    const barInCycle = ((barNumber - this.phraseCycleOrigin!) % seqLen + seqLen) % seqLen;
    const density = phraseSequence[barInCycle];
    const slot = notesForDensity(density, barDur, beatsPerBar);
    let prevMidi = -1;
    let prevDirection = 0; // +1 ascending, -1 descending, 0 unknown
    let prevLeapSize = 0;

    // Find the arp note whose pitch class is closest to the tonic.
    // Used to anchor beat 1 (the downbeat) to the root.
    const rootPc = update.root % 12;
    let tonicNote = update.arpNotes[0];
    let tonicDist = 6; // max possible pitch-class distance
    for (const n of update.arpNotes) {
      const d = Math.min((n.midiNote % 12 - rootPc + 12) % 12, (rootPc - n.midiNote % 12 + 12) % 12);
      if (d < tonicDist) { tonicDist = d; tonicNote = n; }
    }

    for (let i = 0; i < slot.notesPerBar; i++) {
      // Downbeat always lands on the tonic (or nearest available pitch)
      let note = i === 0 ? tonicNote : update.arpNotes[i % update.arpNotes.length];

      if (prevMidi >= 0 && update.arpNotes.length > 1) {
        // Exclude same note as previous beat
        const available = update.arpNotes.filter(n => n.midiNote !== prevMidi);

        // Weighted selection using bass-idiomatic interval preferences.
        // Stability amplifies the weight hierarchy; chaos flattens it
        // toward uniform (any interval equally likely).
        let bestNote = available[0];
        let bestWeight = -1;

        for (const n of available) {
          const semitones = Math.abs(n.midiNote - prevMidi);
          const intervalClass = semitones % 12;
          const baseWeight = BASS_INTERVAL_WEIGHTS[intervalClass] ?? 0.1;

          // Stability shapes the weight curve: high stability preserves
          // the bass hierarchy, low stability flattens toward uniform.
          // exponent < 1 compresses differences, exponent > 1 amplifies.
          const exponent = 0.3 + 1.4 * update.netStability; // chaos 0.3, stable 1.7
          let w = Math.pow(baseWeight, exponent);

          // Leap recovery: after a large leap, bias toward opposite direction
          if (prevDirection !== 0 && prevLeapSize > LEAP_RECOVERY_THRESHOLD) {
            const dir = n.midiNote > prevMidi ? 1 : -1;
            if (dir === prevDirection) {
              w *= 0.3; // penalise continuing in same direction after big leap
            }
          }

          // Randomise within the weighted space
          const score = w * Math.random();
          if (score > bestWeight) {
            bestWeight = score;
            bestNote = n;
          }
        }

        note = bestNote;
      }

      // Track direction and leap size for recovery logic
      if (prevMidi >= 0) {
        const diff = note.midiNote - prevMidi;
        prevDirection = diff > 0 ? 1 : diff < 0 ? -1 : 0;
        prevLeapSize = Math.abs(diff);
      }
      prevMidi = note.midiNote;
      const time = barStart + i * slot.noteDuration;
      this.schedulePluck(note, time, slot.noteDuration, update.avgVelocity);
    }
  }

  /**
   * Continuously update quartal voice volumes from free-particle data.
   * Called every frame from the simulation (not bar-quantized).
   */
  updateFreeParticleVolumes(
    freePercentByType: ReadonlyMap<number, number>,
  ): void {
    if (!this.quartalActive || !this.ctx) return;
    const now = this.ctx.currentTime;
    const scale = 0.7 / Math.sqrt(Math.max(1, this.quartalVoices.length));
    for (const voice of this.quartalVoices) {
      // Don't override the slow warmup attack while it's still ramping
      if (now < voice.warmupUntil) continue;
      const percent = freePercentByType.get(voice.typeId) ?? 0;
      voice.envelope.gain.linearRampToValueAtTime(percent * scale, now + 0.1);
    }
  }

  /* ── Quartal stack management ───────────────────────────────────── */

  private activateQuartalStack(update: BassUpdate, time: number, barDur: number): void {
    if (!this.ctx || !this.output) return;

    const allTypeIds = [...update.freeParticlePercentByType.keys()];
    // Cap to voice budget — keep the types with the highest free particle %
    const typeIds = allTypeIds.length <= this._maxQuartalVoices
      ? allTypeIds
      : allTypeIds
          .sort((a, b) =>
            (update.freeParticlePercentByType.get(b) ?? 0) -
            (update.freeParticlePercentByType.get(a) ?? 0))
          .slice(0, this._maxQuartalVoices);
    const pitches = quartalStackPitches(update.root, typeIds.length);
    const scale = 0.7 / Math.sqrt(Math.max(1, typeIds.length));
    const { typeA, typeB, mix } = waveformBlendToTypes(0.7); // warm ambient blend

    // All voices must be fully present by 75% of the bar, leaving the
    // final quarter with the complete ensemble.  Entries are scattered
    // randomly across the first 75% — not quantized to beats.
    const deadlineFraction = 0.75;
    const deadlineTime = time + deadlineFraction * barDur;
    const n = typeIds.length;

    for (let i = 0; i < n; i++) {
      const midi = pitches[i];
      const freq = midiToFreq(midi);
      const freePercent = update.freeParticlePercentByType.get(typeIds[i]) ?? 1;
      // Random entry — true randomness so each activation sounds different
      const entryFraction = Math.random() * (deadlineFraction - 0.05);
      const entryTime = time + entryFraction * barDur;
      const attackDuration = Math.max(0.01, deadlineTime - entryTime);
      const warmupUntil = entryTime + attackDuration;           // == deadlineTime

      const oscA = this.ctx.createOscillator();
      oscA.type = typeA;
      oscA.frequency.value = freq;

      const oscB = this.ctx.createOscillator();
      oscB.type = typeB;
      oscB.frequency.value = freq;

      const gainA = this.ctx.createGain();
      gainA.gain.value = 1 - mix;

      const gainB = this.ctx.createGain();
      gainB.gain.value = mix;

      const filter = this.ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = 300;
      filter.Q.value = 0.5;

      // Orchestral seating: low voices left, high voices right (audience POV).
      // Map pitch position within the stack to -0.7 … +0.7
      const panPos = typeIds.length > 1
        ? -0.7 + (i / (typeIds.length - 1)) * 1.4
        : 0;
      const panner = this.ctx.createStereoPanner();
      panner.pan.value = panPos;

      const envelope = this.ctx.createGain();
      // Ease-in with a small audible floor so the voice is immediately
      // perceptible, then swells smoothly to full volume.
      const target = freePercent * scale;
      const floor = 0.08;                         // 8% — just barely audible
      const curveLen = Math.max(2, Math.ceil(attackDuration * 64));
      const curve = new Float32Array(curveLen);
      for (let j = 0; j < curveLen; j++) {
        const t = j / (curveLen - 1);            // 0 → 1
        curve[j] = (floor + (1 - floor) * t * t) * target;
      }
      envelope.gain.setValueAtTime(0, entryTime);
      envelope.gain.setValueCurveAtTime(curve, entryTime, attackDuration);

      oscA.connect(gainA);
      oscB.connect(gainB);
      gainA.connect(filter);
      gainB.connect(filter);
      filter.connect(panner);
      panner.connect(envelope);
      envelope.connect(this.output);

      oscA.start(entryTime);
      oscB.start(entryTime);

      this.quartalVoices.push({
        oscA, oscB, gainA, gainB, filter, panner, envelope,
        currentMidi: midi,
        typeId: typeIds[i],
        warmupUntil,
      });
    }

    this.quartalActive = true;
  }

  private deactivateQuartalStack(time: number): void {
    const fadeOut = 1.5;
    for (const voice of this.quartalVoices) {
      voice.envelope.gain.linearRampToValueAtTime(0, time + fadeOut);
      voice.oscA.stop(time + fadeOut + 0.1);
      voice.oscB.stop(time + fadeOut + 0.1);
    }
    // Clear references — stopped nodes will be GC'd by the browser
    this.quartalVoices = [];
    this.quartalActive = false;
    this.lastQuartalRoot = -1;
  }

  private updateQuartalPitches(update: BassUpdate, time: number, barDur: number): void {
    if (!this.ctx) return;

    const allTypeIds = [...update.freeParticlePercentByType.keys()];
    const typeIds = allTypeIds.length <= this._maxQuartalVoices
      ? allTypeIds
      : allTypeIds
          .sort((a, b) =>
            (update.freeParticlePercentByType.get(b) ?? 0) -
            (update.freeParticlePercentByType.get(a) ?? 0))
          .slice(0, this._maxQuartalVoices);
    const targetPitches = quartalStackPitches(update.root, typeIds.length);

    // If voice count changed, tear down and rebuild
    if (typeIds.length !== this.quartalVoices.length) {
      this.deactivateQuartalStack(time);
      this.activateQuartalStack(update, time + 0.05, barDur);
      return;
    }

    // Voice-lead existing voices to nearest targets
    const currentMidis = this.quartalVoices.map(v => v.currentMidi);
    const assignment = assignVoiceLeading(currentMidis, targetPitches);

    for (let i = 0; i < this.quartalVoices.length; i++) {
      const voice = this.quartalVoices[i];
      const target = assignment[i];
      if (target == null || target === voice.currentMidi) continue;

      const freq = midiToFreq(target);
      voice.oscA.frequency.exponentialRampToValueAtTime(freq, time + 0.2);
      voice.oscB.frequency.exponentialRampToValueAtTime(freq, time + 0.2);
      voice.currentMidi = target;
    }

    // Re-sort pan positions by pitch so spatial arrangement stays consistent
    const sorted = [...this.quartalVoices].sort((a, b) => a.currentMidi - b.currentMidi);
    const n = sorted.length;
    for (let i = 0; i < n; i++) {
      const panPos = n > 1 ? -0.7 + (i / (n - 1)) * 1.4 : 0;
      sorted[i].panner.pan.linearRampToValueAtTime(panPos, time + 0.2);
    }
  }

  private teardownQuartalVoices(): void {
    for (const voice of this.quartalVoices) {
      try { voice.oscA.stop(); } catch { /* already stopped */ }
      try { voice.oscB.stop(); } catch { /* already stopped */ }
      voice.oscA.disconnect();
      voice.oscB.disconnect();
      voice.gainA.disconnect();
      voice.gainB.disconnect();
      voice.filter.disconnect();
      voice.envelope.disconnect();
    }
    this.quartalVoices = [];
    this.quartalActive = false;
    this.lastQuartalRoot = -1;
  }

  /* ── Pluck scheduling ───────────────────────────────────────────── */

  private schedulePluck(note: ArpNote, time: number, beatDur: number, avgVelocity: number): void {
    if (!this.ctx || !this.output) return;

    // Keep bass below middle C (MIDI 60) — drop octaves until it fits
    let midi = note.midiNote;
    while (midi >= 60) midi -= 12;
    const freq = midiToFreq(midi);
    const { typeA, typeB, mix } = waveformBlendToTypes(note.sociability);

    // Oscillators
    const oscA = this.ctx.createOscillator();
    oscA.type = typeA;
    oscA.frequency.value = freq;

    const oscB = this.ctx.createOscillator();
    oscB.type = typeB;
    oscB.frequency.value = freq;

    // Waveform blend gains
    const gainA = this.ctx.createGain();
    gainA.gain.value = 1 - mix;

    const gainB = this.ctx.createGain();
    gainB.gain.value = mix;

    // Low-pass filter for dark bass tone
    const filter = this.ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 400;
    filter.Q.value = 0.5;

    // Amplitude envelope from editor shape
    const env = this.ctx.createGain();
    const volume = note.freePercent * sociabilityGain(note.sociability);

    // Staccato: higher average velocity → shorter gate relative to slot.
    // Normalize velocity to [0,1] then look up staccato amount from the curve LUT.
    // LUT x = normalized velocity, y = staccato intensity (0 = legato, 1 = max staccato).
    const velNorm = Math.min(avgVelocity / 300, 1);
    let staccato = 0;
    if (this._staccatoLUT && this._staccatoLUT.length > 0) {
      const idx = velNorm * (this._staccatoLUT.length - 1);
      const lo = Math.floor(idx);
      const hi = Math.min(lo + 1, this._staccatoLUT.length - 1);
      const t = idx - lo;
      staccato = this._staccatoLUT[lo] * (1 - t) + this._staccatoLUT[hi] * t;
    }
    const gateFraction = 1.0 - 0.8 * staccato;   // 1.0 (legato) … 0.2 (staccato)
    const totalDuration = beatDur * gateFraction;

    if (this.bassEnvelopeShape && totalDuration > 0) {
      // Bass pluck gate = one beat; A/D/R proportions from the beat duration
      const attackDur = totalDuration * 0.10;
      const decayDur = totalDuration * 0.20;
      const releaseDur = totalDuration * 0.30;
      const sustainDur = Math.max(0, totalDuration - attackDur - decayDur - releaseDur);
      const lutSize = Math.max(8, Math.min(8192, Math.ceil(totalDuration * this.ctx.sampleRate / 128)));
      const rawLut = buildGateAwareLUT(this.bassEnvelopeShape, attackDur, decayDur, sustainDur, releaseDur, lutSize);
      const curve = new Float32Array(rawLut.length);
      for (let i = 0; i < rawLut.length; i++) {
        curve[i] = rawLut[i] * volume;
      }
      if (curve.length >= 2) {
        env.gain.setValueAtTime(0, time);
        env.gain.setValueCurveAtTime(curve, time, totalDuration);
      }
    } else {
      // Fallback: simple pluck shape
      const attackEnd = Math.min(0.01, totalDuration * 0.1);
      const decayEnd = Math.min(0.16, totalDuration * 0.3);
      const releaseStart = Math.max(decayEnd, totalDuration * 0.7);
      env.gain.setValueAtTime(0, time);
      env.gain.linearRampToValueAtTime(volume, time + attackEnd);
      env.gain.linearRampToValueAtTime(volume * 0.4, time + decayEnd);
      env.gain.setValueAtTime(volume * 0.4, time + releaseStart);
      env.gain.linearRampToValueAtTime(0, time + totalDuration);
    }

    const stopTime = time + totalDuration + 0.05;

    // Connect: osc → blend gain → filter → envelope → output
    oscA.connect(gainA);
    oscB.connect(gainB);
    gainA.connect(filter);
    gainB.connect(filter);
    filter.connect(env);
    env.connect(this.output);

    oscA.start(time);
    oscB.start(time);
    oscA.stop(stopTime);
    oscB.stop(stopTime);
  }
}
