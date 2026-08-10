/**
 * Web Audio API playback engine (§3.2 Phase B).
 *
 * Imperative module — manages AudioContext, creates/destroys nodes,
 * and plays notes via a tuplet grid scrubber. Everything else in
 * src/music/ is pure; this is the only file that touches the Web Audio API.
 */

import { buildGateAwareLUT } from "../envelope-editor"
import type { DroneVoice } from "./particle-chorus"
import type { PlaybackParams } from "./play-time-params"
import {
	sociabilityGain,
	waveformBlendToTypes,
	waveformLowpassCutoff,
} from "./timbre"
import type {
	EnvelopeCurve,
	EnvelopeShape,
	SlotNote,
	TupletGrid,
} from "./types"
import { MAX_SUBDIVISION, createTupletGrid } from "./types"
import { midiToFreq } from "./utils"

/** One bar's worth of tuplet grid plus its playback cursors. Two of these
 *  can be live at once (current bar + next bar) so scheduling the next bar
 *  early never truncates the tail of the current one. */
interface ActiveGrid {
	readonly grid: TupletGrid
	readonly cursors: number[]
	readonly barStart: number
	readonly barDur: number
}

/** Persistent AudioNodes for one chorus drone voice (always sine —
 *  the wash stays timbrally neutral background glue). */
interface PadVoiceNodes {
	readonly osc: OscillatorNode
	readonly voiceGain: GainNode
}

/* ------------------------------------------------------------------ */
/*  AudioGraph class                                                   */
/* ------------------------------------------------------------------ */

export class AudioGraph {
	private ctx: AudioContext | null = null
	private masterGain: GainNode | null = null
	private reverbWet: GainNode | null = null
	private reverbDry: GainNode | null = null
	private convolver: ConvolverNode | null = null
	private enabled = false
	private volume = 0.5
	private volumeMultiplier = 1
	private manualEnvelope: EnvelopeShape | null = null

	/** How far ahead (seconds) to create AudioNodes before a hit's start time. */
	private static readonly JIT_LOOKAHEAD = 0.1 // 100ms — ~6 frames of headroom
	/** Max hits to realize per frame — keeps node creation under the glitch threshold. */
	private static readonly MAX_PER_TICK = 32

	/* ── Tuplet grid state ──────────────────────────────────────────── */

	/** Live grids, oldest first (at most current bar + next bar). */
	private grids: ActiveGrid[] = []

	/* ── Lifecycle ───────────────────────────────────────────────────── */

	enable(): void {
		if (this.ctx) {
			if (this.ctx.state === "suspended") this.ctx.resume()
			this.enabled = true
			return
		}

		this.ctx = new AudioContext()

		// Master gain
		this.masterGain = this.ctx.createGain()
		this.applyMasterGain()

		// Reverb (synthetic IR)
		this.convolver = this.ctx.createConvolver()
		this.convolver.buffer = this.createReverbIR(2, 2)

		this.reverbWet = this.ctx.createGain()
		this.reverbWet.gain.value = 0.3
		this.reverbDry = this.ctx.createGain()
		this.reverbDry.gain.value = 0.7

		// Routing: master → dry → destination
		//          master → convolver → wet → destination
		this.masterGain.connect(this.reverbDry)
		this.masterGain.connect(this.convolver)
		this.convolver.connect(this.reverbWet)
		this.reverbDry.connect(this.ctx.destination)
		this.reverbWet.connect(this.ctx.destination)

		this.enabled = true
	}

	disable(): void {
		this.enabled = false
		if (this.ctx && this.ctx.state === "running") {
			this.ctx.suspend()
		}
	}

	dispose(): void {
		this.enabled = false
		this.padVoices.clear()
		this.padMaster = null
		this.grids = []
		if (this.ctx) {
			this.ctx.close()
			this.ctx = null
			this.masterGain = null
			this.convolver = null
			this.reverbWet = null
			this.reverbDry = null
		}
	}

	get isEnabled(): boolean {
		return this.enabled
	}
	get currentTime(): number {
		return this.ctx?.currentTime ?? 0
	}
	get context(): AudioContext | null {
		return this.ctx
	}

	/* ── Volume ──────────────────────────────────────────────────────── */

	/** Base output gain the whole mix is scaled by — per-voice levels are
	 *  conservative, so the overall loudness is lifted here (single knob;
	 *  note voices, pad, and reverb all sit under masterGain, so relative
	 *  balance is preserved). */
	private static readonly BASE_GAIN = 4

	private applyMasterGain(): void {
		if (this.masterGain) {
			this.masterGain.gain.value =
				AudioGraph.BASE_GAIN * this.volume * this.volumeMultiplier
		}
	}

	setVolume(v: number): void {
		this.volume = Math.max(0, Math.min(1, v))
		this.applyMasterGain()
	}

	setVolumeMultiplier(m: number): void {
		this.volumeMultiplier = Math.max(0, m)
		this.applyMasterGain()
	}

	/* ── Manual envelope shape ────────────────────────────────────────── */

	setManualEnvelope(shape: EnvelopeShape | null): void {
		this.manualEnvelope = shape
	}

	/* ── Reverb mix (driven by net stability) ────────────────────────── */

	/**
	 * Set reverb wet/dry mix from spatial entropy (§6.2).
	 * High entropy (spread out) = more reverb. Low entropy (clustered) = dry.
	 */
	setReverbFromEntropy(spatialEntropy: number, atTime?: number): void {
		const wet = spatialEntropy // high entropy = wet
		const t = atTime ?? this.ctx?.currentTime ?? 0
		// Schedule at bar start so the change aligns with new notes, not with
		// worker-response time (which can be hundreds of ms earlier).
		if (this.reverbWet) this.reverbWet.gain.setTargetAtTime(wet * 0.5, t, 0.05)
		if (this.reverbDry)
			this.reverbDry.gain.setTargetAtTime(1 - wet * 0.3, t, 0.05)
	}

	/* ── Particle chorus drones (always-on full-range wash) ───────────── */

	private padMaster: GainNode | null = null
	private padVoices = new Map<string, PadVoiceNodes>()

	/** Overall pad level, well under the note voices. */
	private static readonly PAD_LEVEL = 0.02
	/** Pitch glide time constant (s) — voice-leading slides, not jumps. */
	private static readonly PAD_FREQ_TAU = 0.0625
	/** Gain breathing time constant (s). */
	private static readonly PAD_GAIN_TAU = 1.2

	/**
	 * Reconcile the pad against a freshly computed voicing. Existing voices
	 * glide to new pitches/levels; new voices fade in; missing voices fade
	 * out and are reclaimed. Call once per bar (bar-meta).
	 */
	updateDronePad(voices: readonly DroneVoice[], atTime?: number): void {
		const ctx = this.ctx
		if (!ctx || !this.masterGain) return
		const t = Math.max(atTime ?? ctx.currentTime, ctx.currentTime)

		if (!this.padMaster) {
			this.padMaster = ctx.createGain()
			this.padMaster.gain.value = 0
			this.padMaster.connect(this.masterGain)
		}

		// Normalize so a many-type pad doesn't get louder than a two-type pad
		const audible = voices.filter((v) => v.gain > 0.001).length
		this.padMaster.gain.setTargetAtTime(
			AudioGraph.PAD_LEVEL / Math.sqrt(Math.max(1, audible)),
			t,
			AudioGraph.PAD_GAIN_TAU,
		)

		const seen = new Set<string>()
		for (const voice of voices) {
			seen.add(voice.key)
			const freq = midiToFreq(voice.midiNote)
			const existing = this.padVoices.get(voice.key)

			if (existing) {
				existing.osc.frequency.setTargetAtTime(freq, t, AudioGraph.PAD_FREQ_TAU)
				existing.voiceGain.gain.setTargetAtTime(
					voice.gain,
					t,
					AudioGraph.PAD_GAIN_TAU,
				)
				continue
			}

			const voiceGain = ctx.createGain()
			voiceGain.gain.value = 0
			voiceGain.gain.setTargetAtTime(voice.gain, t, AudioGraph.PAD_GAIN_TAU)

			const osc = ctx.createOscillator()
			osc.type = "sine"
			osc.frequency.value = freq
			osc.connect(voiceGain)
			voiceGain.connect(this.padMaster)
			osc.start(t)

			this.padVoices.set(voice.key, { osc, voiceGain })
		}

		// Fade out voices for types no longer present
		for (const [key, nodes] of this.padVoices) {
			if (seen.has(key)) continue
			nodes.voiceGain.gain.setTargetAtTime(0, t, AudioGraph.PAD_GAIN_TAU)
			const { osc, voiceGain } = nodes
			setTimeout(() => {
				try {
					osc.stop()
					voiceGain.disconnect()
				} catch {
					/* context may have closed */
				}
			}, 6000)
			this.padVoices.delete(key)
		}
	}

	/* ── Tuplet grid API ─────────────────────────────────────────────── */

	/** Notes older than this past the slot time are dropped, not rushed. */
	private static readonly LATE_GRACE = 0.08

	/**
	 * Allocate a fresh empty grid for a new bar. The previous bar's grid
	 * stays live until it finishes playing, so scheduling the next bar
	 * early never truncates the current one.
	 */
	initGrid(barStart: number, barDur: number): void {
		this.grids.push({
			grid: createTupletGrid(),
			cursors: new Array(MAX_SUBDIVISION).fill(0),
			barStart,
			barDur,
		})
		while (this.grids.length > 2) this.grids.shift()
	}

	/**
	 * Replay the newest grid's notes as a fresh bar starting at `barStart`
	 * — holds a computed melody for multiple bars without recalculating.
	 * Slot arrays are copied so the source bar's slots stay untouched.
	 * Returns false when there is no grid to repeat.
	 */
	repeatGrid(barStart: number, barDur: number): boolean {
		const source = this.grids[this.grids.length - 1]
		if (!source) return false
		this.grids.push({
			grid: {
				tiers: source.grid.tiers.map((tier) =>
					tier.map((slot) => (slot ? [...slot] : null)),
				),
			},
			cursors: new Array(MAX_SUBDIVISION).fill(0),
			barStart,
			barDur,
		})
		while (this.grids.length > 2) this.grids.shift()
		return true
	}

	/**
	 * Discard all live grids. Must be called whenever the bar timeline is
	 * re-anchored (BPM/time-multiplier change, sound re-enable) — otherwise
	 * the old timeline's future-dated slots keep playing on top of the new
	 * bar 0.
	 */
	clearGrids(): void {
		this.grids = []
	}

	/** Append notes to the newest grid's slot (slot-fill messages always
	 *  target the most recently scheduled bar). */
	fillSlot(
		tierIndex: number,
		slotIndex: number,
		notes: readonly SlotNote[],
	): void {
		const active = this.grids[this.grids.length - 1]
		if (!active) return
		const tier = active.grid.tiers[tierIndex]
		if (!tier || slotIndex >= tier.length) return
		const existing = tier[slotIndex]
		if (existing) {
			existing.push(...notes)
		} else {
			tier[slotIndex] = [...notes]
		}
	}

	/** Read the grid covering `atTime` (for visualization); falls back to
	 *  the newest grid. */
	getGrid(atTime?: number): TupletGrid | null {
		if (this.grids.length === 0) return null
		if (atTime !== undefined) {
			for (const ag of this.grids) {
				if (atTime >= ag.barStart && atTime < ag.barStart + ag.barDur) {
					return ag.grid
				}
			}
		}
		return this.grids[this.grids.length - 1].grid
	}

	/**
	 * Tick the grid-based scrubber. Call every frame.
	 *
	 * Drains every due slot in every live grid (so a slow frame doesn't
	 * strand slots behind a once-per-tick cursor), resolves playback params
	 * from live state, and creates AudioNodes for each note.
	 *
	 * @param resolveParams — Given a SlotNote, returns PlaybackParams or null
	 *   (null = organism is dead, skip the note).
	 */
	tickGridScheduler(
		resolveParams: (note: SlotNote) => PlaybackParams | null,
		onPlay?: (note: SlotNote, params: PlaybackParams, time: number) => void,
	): void {
		if (!this.ctx || !this.enabled || this.grids.length === 0) return
		const now = this.ctx.currentTime
		const horizon = now + AudioGraph.JIT_LOOKAHEAD
		let scheduled = 0

		for (const ag of this.grids) {
			for (
				let t = 0;
				t < MAX_SUBDIVISION && scheduled < AudioGraph.MAX_PER_TICK;
				t++
			) {
				const tier = ag.grid.tiers[t]
				const tierSize = t + 1 // tier 0 = 1 slot, tier 1 = 2 slots, etc.

				// Drain every due slot (not just one per tick)
				while (
					ag.cursors[t] < tierSize &&
					scheduled < AudioGraph.MAX_PER_TICK
				) {
					const cursor = ag.cursors[t]
					const slotTime = ag.barStart + (cursor / tierSize) * ag.barDur
					if (slotTime > horizon) break // not yet time

					// Defer a multi-note slot that won't fit this tick's budget
					// rather than dropping its tail mid-slot (the cursor only
					// ever advances, so half-played slots are unrecoverable).
					// A fresh tick (scheduled === 0) always proceeds so a slot
					// larger than the whole budget can't stall forever.
					const slot = tier[cursor]
					if (
						slot &&
						slot.length > 0 &&
						scheduled > 0 &&
						scheduled + slot.length > AudioGraph.MAX_PER_TICK
					) {
						break
					}

					// Advance cursor even if slot is empty or notes fail to resolve
					ag.cursors[t] = cursor + 1

					if (!slot || slot.length === 0) continue

					if (slotTime < now - AudioGraph.LATE_GRACE) {
						console.warn(
							`  ⚠️ TOO LATE tier=${t} slot=${cursor} slotTime=${slotTime.toFixed(3)} < now=${now.toFixed(3)}`,
						)
						continue
					}
					// Slightly late notes play now instead of being dropped
					const playTime = Math.max(slotTime, now + 0.005)

					for (const note of slot) {
						if (scheduled >= AudioGraph.MAX_PER_TICK) {
							console.warn(
								`  ⚠️ MAX_PER_TICK (${AudioGraph.MAX_PER_TICK}) hit — dropping midi=${note.midiNote} tier=${t} slot=${cursor}`,
							)
							break
						}
						const params = resolveParams(note)
						if (!params) continue // organism dead or not found

						this.playNote(note.midiNote, playTime, params)
						if (onPlay) onPlay(note, params, playTime)
						scheduled++
					}
				}
			}
		}

		// Drop grids that are fully in the past
		this.grids = this.grids.filter((ag) => ag.barStart + ag.barDur > now - 0.5)
	}

	/** Create AudioNodes for a single note with pre-computed playback params. */
	private playNote(
		midiNote: number,
		time: number,
		params: PlaybackParams,
	): void {
		const ctx = this.ctx!
		const {
			envelope,
			waveform,
			volume,
			pan,
			filterCutoff,
			gateDuration,
			vibratoDepth,
		} = params

		// Jitter startTime by up to 3ms to break oscillator phase alignment
		const startTime = time + Math.random() * 0.003

		// Gate-aware timing
		let attackDur = Math.max(envelope.attackDuration, 0.05)
		let decayDur = envelope.decayDuration
		if (gateDuration < attackDur) {
			attackDur = gateDuration
			decayDur = 0
		} else if (gateDuration < attackDur + decayDur) {
			decayDur = gateDuration - attackDur
		}
		const sustainDur = Math.max(0, gateDuration - attackDur - decayDur)
		const releaseDur = envelope.releaseDuration
		const totalDuration = attackDur + decayDur + sustainDur + releaseDur
		const endTime = startTime + totalDuration

		// Waveform blend → two oscillators
		const { typeA, typeB, mix } = waveformBlendToTypes(waveform.blend)
		const freq = midiToFreq(midiNote)

		const oscA = ctx.createOscillator()
		oscA.type = typeA
		oscA.frequency.value = freq
		const gainA = ctx.createGain()
		gainA.gain.value = 1 - mix

		const oscB = ctx.createOscillator()
		oscB.type = typeB
		oscB.frequency.value = freq
		const gainB = ctx.createGain()
		gainB.gain.value = mix

		// Vibrato (LFO)
		if (vibratoDepth > 0) {
			const lfo = ctx.createOscillator()
			lfo.type = "sine"
			lfo.frequency.value = 5.5
			const lfoGain = ctx.createGain()
			const maxDeviation = freq * (Math.pow(2, vibratoDepth / 1200) - 1)
			const curveLen = 64
			const curve = new Float32Array(curveLen)
			for (let i = 0; i < curveLen; i++) {
				curve[i] = (i / (curveLen - 1)) ** 3 * maxDeviation
			}
			lfoGain.gain.setValueAtTime(0, startTime)
			lfoGain.gain.setValueCurveAtTime(curve, startTime, totalDuration)
			lfo.connect(lfoGain)
			lfoGain.connect(oscA.frequency)
			lfoGain.connect(oscB.frequency)
			lfo.start(startTime)
			lfo.stop(endTime)
		}

		// Bandpass filter
		const filter = ctx.createBiquadFilter()
		filter.type = "bandpass"
		filter.frequency.value = filterCutoff
		filter.Q.value = 1

		// Waveform-proportional lowpass
		const lpf = ctx.createBiquadFilter()
		lpf.type = "lowpass"
		lpf.frequency.value = waveformLowpassCutoff(waveform.blend)
		lpf.Q.value = 0.707

		// Envelope gain
		const envGain = ctx.createGain()
		envGain.gain.value = 0
		// Floor the resolved volume so every scheduled note is audible, THEN
		// apply waveform-loudness compensation so harsh (low-sociability)
		// voices sit lower in the mix. With the compensation inside the max(),
		// the floor re-boosted saws to the same peak as sines — and equal
		// amplitude reads ~2x louder for a sawtooth, so harsh voices stuck out.
		const peakVol =
			Math.max(0.18, volume) * sociabilityGain(waveform.sociability)

		if (this.manualEnvelope) {
			const lutSize = Math.max(
				8,
				Math.min(8192, Math.ceil((totalDuration * ctx.sampleRate) / 128)),
			)
			const rawLut = buildGateAwareLUT(
				this.manualEnvelope,
				attackDur,
				decayDur,
				sustainDur,
				releaseDur,
				lutSize,
			)
			const envCurve = new Float32Array(rawLut.length)
			for (let i = 0; i < rawLut.length; i++) envCurve[i] = rawLut[i] * peakVol
			if (envCurve.length >= 2 && totalDuration > 0) {
				envGain.gain.setValueAtTime(0, startTime)
				envGain.gain.setValueCurveAtTime(envCurve, startTime, totalDuration)
			}
		} else {
			envGain.gain.setValueAtTime(0, startTime)
			const peakAmp = peakVol * envelope.peakLevel
			const sustainAmp = peakVol * envelope.sustainLevel
			const attackEnd = startTime + attackDur
			const decayEnd = attackEnd + decayDur
			const sustainEnd = decayEnd + sustainDur
			this.applyEnvelopeSegment(
				envGain.gain,
				startTime,
				attackEnd,
				0,
				peakAmp,
				envelope.attackCurve,
			)
			this.applyEnvelopeSegment(
				envGain.gain,
				attackEnd,
				decayEnd,
				peakAmp,
				sustainAmp,
				envelope.decayCurve,
			)
			envGain.gain.setValueAtTime(sustainAmp, decayEnd)
			this.applyEnvelopeSegment(
				envGain.gain,
				sustainEnd,
				endTime,
				sustainAmp,
				0,
				envelope.releaseCurve,
			)
		}

		// Stereo panning
		const panner = ctx.createStereoPanner()
		panner.pan.value = Math.max(-1, Math.min(1, pan))

		// Connect graph
		oscA.connect(gainA)
		oscB.connect(gainB)
		gainA.connect(filter)
		gainB.connect(filter)
		filter.connect(lpf)
		lpf.connect(envGain)
		envGain.connect(panner)
		panner.connect(this.masterGain!)

		// Start and auto-stop
		oscA.start(startTime)
		oscA.stop(endTime + 0.01)
		oscB.start(startTime)
		oscB.stop(endTime + 0.01)
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
		const duration = endTime - startTime
		if (duration <= 0) return

		// Ensure we have a known starting value at startTime
		param.setValueAtTime(fromValue, startTime)

		switch (curve) {
			case "linear":
				param.linearRampToValueAtTime(toValue, endTime)
				break

			case "exponential":
				// setTargetAtTime with τ = duration/3 reaches ~95% at endTime
				// Then snap to exact value
				param.setTargetAtTime(toValue, startTime, duration / 3)
				param.setValueAtTime(toValue, endTime)
				break

			case "ease-in":
				// Slow start: use setTargetAtTime with large τ (sluggish), then
				// linear ramp for the final portion to hit the target
				param.setTargetAtTime(toValue, startTime, duration / 1.5)
				param.setValueAtTime(toValue, endTime)
				break

			case "ease-out":
				// Fast start: exponential with short τ, settles quickly
				param.setTargetAtTime(toValue, startTime, duration / 5)
				param.setValueAtTime(toValue, endTime)
				break
		}
	}

	/* ── Reverb IR generation ────────────────────────────────────────── */

	private createReverbIR(
		duration: number,
		sampleRate: number = 2,
	): AudioBuffer {
		const rate = this.ctx!.sampleRate
		const length = rate * duration
		const buffer = this.ctx!.createBuffer(2, length, rate)
		for (let ch = 0; ch < 2; ch++) {
			const data = buffer.getChannelData(ch)
			for (let i = 0; i < length; i++) {
				data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2)
			}
		}
		return buffer
	}
}
