/**
 * Web Audio API playback engine (§3.2 Phase B).
 *
 * Imperative module — manages AudioContext, creates/destroys nodes,
 * schedules hits from ScheduledBar. Everything else in src/music/ is pure;
 * this is the only file that touches the Web Audio API.
 */

import type { ScheduledBar, ScheduledHit, EnvelopeCurve, EnvelopeShape } from "./types";
import { waveformBlendToTypes, sociabilityGain, waveformLowpassCutoff } from "./timbre";
import { buildGateAwareLUT } from "../envelope-editor";
import { midiToFreq } from "./utils";

/* ------------------------------------------------------------------ */
/*  AudioGraph class                                                   */
/* ------------------------------------------------------------------ */

export class AudioGraph {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private limiter: DynamicsCompressorNode | null = null;
  private masterLpf: BiquadFilterNode | null = null;
  private reverbWet: GainNode | null = null;
  private reverbDry: GainNode | null = null;
  private convolver: ConvolverNode | null = null;
  private enabled = false;
  private volume = 0.5;
  private volumeMultiplier = 1;
  private manualEnvelope: EnvelopeShape | null = null;
  private lpfLut: Float32Array | null = null;
  private lpfMinFreq = 200;
  private lpfMaxFreq = 20000;

  /* ── Lifecycle ───────────────────────────────────────────────────── */

  enable(): void {
    if (this.ctx) {
      if (this.ctx.state === "suspended") this.ctx.resume();
      this.enabled = true;
      return;
    }

    this.ctx = new AudioContext();

    // Master limiter
    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -6;
    this.limiter.knee.value = 10;
    this.limiter.ratio.value = 12;
    this.limiter.attack.value = 0.003;
    this.limiter.release.value = 0.25;

    // Master gain
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = this.volume * this.volumeMultiplier;

    // Master bus low-pass filter (curve-editor driven)
    this.masterLpf = this.ctx.createBiquadFilter();
    this.masterLpf.type = "lowpass";
    this.masterLpf.frequency.value = this.lpfMaxFreq;
    this.masterLpf.Q.value = 0.707; // Butterworth — flat passband

    // Reverb (synthetic IR)
    this.convolver = this.ctx.createConvolver();
    this.convolver.buffer = this.createReverbIR(2, 2);

    this.reverbWet = this.ctx.createGain();
    this.reverbWet.gain.value = 0.3;
    this.reverbDry = this.ctx.createGain();
    this.reverbDry.gain.value = 0.7;

    // Routing: master → LPF → dry → limiter → destination
    //          master → LPF → convolver → wet → limiter → destination
    this.masterGain.connect(this.masterLpf);
    this.masterLpf.connect(this.reverbDry);
    this.masterLpf.connect(this.convolver);
    this.convolver.connect(this.reverbWet);
    this.reverbDry.connect(this.limiter);
    this.reverbWet.connect(this.limiter);
    this.limiter.connect(this.ctx.destination);

    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
    if (this.ctx && this.ctx.state === "running") {
      this.ctx.suspend();
    }
  }

  dispose(): void {
    this.enabled = false;
    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
      this.masterGain = null;
      this.masterLpf = null;
      this.limiter = null;
      this.convolver = null;
      this.reverbWet = null;
      this.reverbDry = null;
    }
  }

  get isEnabled(): boolean { return this.enabled; }
  get currentTime(): number { return this.ctx?.currentTime ?? 0; }
  get context(): AudioContext | null { return this.ctx; }
  /** Destination for the bass layer — routes through master gain → reverb → limiter. */
  get masterDestination(): AudioNode | null { return this.masterGain; }

  /* ── Volume ──────────────────────────────────────────────────────── */

  setVolume(v: number): void {
    this.volume = Math.max(0, Math.min(1, v));
    if (this.masterGain) {
      this.masterGain.gain.value = this.volume * this.volumeMultiplier;
    }
  }

  setVolumeMultiplier(m: number): void {
    this.volumeMultiplier = Math.max(0, m);
    if (this.masterGain) {
      this.masterGain.gain.value = this.volume * this.volumeMultiplier;
    }
  }

  /* ── Master bus low-pass filter ───────────────────────────────────── */

  /**
   * Store a 0→1 LUT that maps bar-position → cutoff frequency.
   * Called once when the curve editor changes; sampled per-frame via
   * updateLpfFromBarPosition().
   */
  setLpfLUT(lut: Float32Array | null): void {
    this.lpfLut = lut;
  }

  /**
   * Set the resonance (Q) of the master bus LPF.
   */
  setLpfQ(q: number): void {
    if (this.masterLpf) this.masterLpf.Q.value = q;
  }

  /**
   * Sample the LPF curve at the current bar position and update cutoff.
   * @param barFraction  0→1 progress through the current bar
   */
  updateLpfFromBarPosition(barFraction: number): void {
    if (!this.masterLpf || !this.lpfLut) return;
    const idx = Math.min(
      this.lpfLut.length - 1,
      Math.max(0, Math.round(barFraction * (this.lpfLut.length - 1))),
    );
    const t = this.lpfLut[idx]; // 0→1
    // Log-scale interpolation between min and max frequency
    const freq =
      this.lpfMinFreq * Math.pow(this.lpfMaxFreq / this.lpfMinFreq, t);
    this.masterLpf.frequency.value = freq;
  }

  /* ── Manual envelope shape ────────────────────────────────────────── */

  setManualEnvelope(shape: EnvelopeShape | null): void {
    this.manualEnvelope = shape;
  }

  /* ── Reverb mix (driven by net stability) ────────────────────────── */

  /**
   * Set reverb wet/dry mix from spatial entropy (§6.2).
   * High entropy (spread out) = more reverb. Low entropy (clustered) = dry.
   */
  setReverbFromEntropy(spatialEntropy: number): void {
    const wet = spatialEntropy; // high entropy = wet
    if (this.reverbWet) this.reverbWet.gain.value = wet * 0.5;
    if (this.reverbDry) this.reverbDry.gain.value = 1 - wet * 0.3;
  }

  /* ── Bar playback ────────────────────────────────────────────────── */

  /**
   * Schedule all hits from a ScheduledBar for playback.
   * Fire-and-forget: nodes self-stop after their envelope completes.
   *
   * @param bar       - The scheduled bar to play
   * @param lookahead - Schedule hits this many seconds early (default 0.05)
   * @returns Mapping of hit index → { startTime, endTime } for visual sync
   */
  playScheduledBar(
    bar: ScheduledBar,
  ): readonly { readonly startTime: number; readonly endTime: number }[] {
    if (!this.ctx || !this.masterGain || !this.enabled) return [];

    this.setReverbFromEntropy(bar.spatialEntropy);

    const hitTimings: { startTime: number; endTime: number }[] = [];

    for (const hit of bar.hits) {
      const timing = this.scheduleHit(hit);
      hitTimings.push(timing);
    }

    return hitTimings;
  }

  /* ── Individual hit scheduling ───────────────────────────────────── */

  private scheduleHit(
    hit: ScheduledHit,
  ): { startTime: number; endTime: number } {
    const ctx = this.ctx!;
    const now = ctx.currentTime;
    const startTime = Math.max(hit.time, now + 0.001);
    const { envelope } = hit;

    // Gate-aware timing: sustain fills the gate after attack+decay
    let attackDur = envelope.attackDuration;
    let decayDur = envelope.decayDuration;
    const gateDur = hit.gateDuration;
    // If gate is shorter than attack+decay, compress both proportionally
    if (gateDur < attackDur + decayDur) {
      const ratio = gateDur / (attackDur + decayDur);
      attackDur *= ratio;
      decayDur *= ratio;
    }
    const sustainDur = Math.max(0, gateDur - attackDur - decayDur);
    const releaseDur = envelope.releaseDuration;
    const totalDuration = attackDur + decayDur + sustainDur + releaseDur;
    const endTime = startTime + totalDuration;

    // ── Waveform blend → two oscillators ──────────────────────────
    const { typeA, typeB, mix } = waveformBlendToTypes(hit.waveform.blend);
    const freq = midiToFreq(hit.midiNote);

    // Oscillator A
    const oscA = ctx.createOscillator();
    oscA.type = typeA;
    oscA.frequency.value = freq;

    const gainA = ctx.createGain();
    gainA.gain.value = 1 - mix;

    // Oscillator B
    const oscB = ctx.createOscillator();
    oscB.type = typeB;
    oscB.frequency.value = freq;

    const gainB = ctx.createGain();
    gainB.gain.value = mix;

    // ── Vibrato (LFO) ────────────────────────────────────────────
    // Vibrato starts at 0 and ramps to full depth by the end of the note.
    if (hit.vibratoDepth > 0) {
      const lfo = ctx.createOscillator();
      lfo.type = "sine";
      lfo.frequency.value = 5.5; // ~5.5 Hz vibrato rate
      const lfoGain = ctx.createGain();
      const maxDeviation = freq * (Math.pow(2, hit.vibratoDepth / 1200) - 1);
      const curveLen = 64;
      const curve = new Float32Array(curveLen);
      for (let i = 0; i < curveLen; i++) {
        const t = i / (curveLen - 1); // 0 → 1
        curve[i] = t * t * t * maxDeviation; // cubic ease-in
      }
      lfoGain.gain.setValueAtTime(0, startTime);
      lfoGain.gain.setValueCurveAtTime(curve, startTime, totalDuration);
      lfo.connect(lfoGain);
      lfoGain.connect(oscA.frequency);
      lfoGain.connect(oscB.frequency);
      lfo.start(startTime);
      lfo.stop(endTime);
    }

    // ── Bandpass filter ──────────────────────────────────────────
    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.value = hit.filterCutoff;
    filter.Q.value = 1;

    // ── Waveform-proportional lowpass (tame highs for harsh waveforms) ──
    const lpf = ctx.createBiquadFilter();
    lpf.type = "lowpass";
    lpf.frequency.value = waveformLowpassCutoff(hit.waveform.blend);
    lpf.Q.value = 0.707; // Butterworth — flat passband, no resonance

    // ── Envelope gain ────────────────────────────────────────────────
    const envGain = ctx.createGain();
    envGain.gain.value = 0;

    const peakVol = hit.volume * sociabilityGain(hit.waveform.sociability);

    if (this.manualEnvelope) {
      // Gate-aware LUT: editor bezier for A/D/R, flat sustain, durations from physics+gate
      const lutSize = Math.max(8, Math.min(8192, Math.ceil(totalDuration * ctx.sampleRate / 128)));
      const rawLut = buildGateAwareLUT(this.manualEnvelope, attackDur, decayDur, sustainDur, releaseDur, lutSize);
      const curve = new Float32Array(rawLut.length);
      for (let i = 0; i < rawLut.length; i++) {
        curve[i] = rawLut[i] * peakVol;
      }
      if (curve.length >= 2 && totalDuration > 0) {
        envGain.gain.setValueAtTime(0, startTime);
        envGain.gain.setValueCurveAtTime(curve, startTime, totalDuration);
      }
    } else {
      // Physics-derived ADSR envelope (fallback when no manual editor)
      envGain.gain.setValueAtTime(0, startTime);
      const peakAmp = peakVol * envelope.peakLevel;
      const sustainAmp = peakVol * envelope.sustainLevel;
      const attackEnd = startTime + attackDur;
      const decayEnd = attackEnd + decayDur;
      const sustainEnd = decayEnd + sustainDur;
      this.applyEnvelopeSegment(envGain.gain, startTime, attackEnd, 0, peakAmp, envelope.attackCurve);
      this.applyEnvelopeSegment(envGain.gain, attackEnd, decayEnd, peakAmp, sustainAmp, envelope.decayCurve);
      envGain.gain.setValueAtTime(sustainAmp, decayEnd);
      this.applyEnvelopeSegment(envGain.gain, sustainEnd, endTime, sustainAmp, 0, envelope.releaseCurve);
    }

    // ── Stereo panning ──────────────────────────────────────────
    const panner = ctx.createStereoPanner();
    panner.pan.value = Math.max(-1, Math.min(1, hit.pan));

    // ── Connect graph ────────────────────────────────────────────
    oscA.connect(gainA);
    oscB.connect(gainB);
    gainA.connect(filter);
    gainB.connect(filter);
    filter.connect(lpf);
    lpf.connect(envGain);
    envGain.connect(panner);
    panner.connect(this.masterGain!);

    // ── Start and auto-stop ─────────────────────────────────────
    oscA.start(startTime);
    oscA.stop(endTime + 0.01);
    oscB.start(startTime);
    oscB.stop(endTime + 0.01);

    return { startTime, endTime };
  }

  /* ── Envelope curve helpers (§8.3) ────────────────────────────────── */

  /**
   * Apply a shaped envelope segment to an AudioParam.
   * Uses setValueAtTime + setTargetAtTime for exponential curves,
   * linearRampToValueAtTime for linear, and combinations for ease-in/out.
   */
  private applyEnvelopeSegment(
    param: AudioParam,
    startTime: number,
    endTime: number,
    fromValue: number,
    toValue: number,
    curve: EnvelopeCurve,
  ): void {
    const duration = endTime - startTime;
    if (duration <= 0) return;

    // Ensure we have a known starting value at startTime
    param.setValueAtTime(fromValue, startTime);

    switch (curve) {
      case "linear":
        param.linearRampToValueAtTime(toValue, endTime);
        break;

      case "exponential":
        // setTargetAtTime with τ = duration/3 reaches ~95% at endTime
        // Then snap to exact value
        param.setTargetAtTime(toValue, startTime, duration / 3);
        param.setValueAtTime(toValue, endTime);
        break;

      case "ease-in":
        // Slow start: use setTargetAtTime with large τ (sluggish), then
        // linear ramp for the final portion to hit the target
        param.setTargetAtTime(toValue, startTime, duration / 1.5);
        param.setValueAtTime(toValue, endTime);
        break;

      case "ease-out":
        // Fast start: exponential with short τ, settles quickly
        param.setTargetAtTime(toValue, startTime, duration / 5);
        param.setValueAtTime(toValue, endTime);
        break;
    }
  }

  /* ── Reverb IR generation ────────────────────────────────────────── */

  private createReverbIR(duration: number, sampleRate: number = 2): AudioBuffer {
    const rate = this.ctx!.sampleRate;
    const length = rate * duration;
    const buffer = this.ctx!.createBuffer(2, length, rate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buffer.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2);
      }
    }
    return buffer;
  }
}
