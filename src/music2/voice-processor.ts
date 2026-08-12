/**
 * PolyVoiceProcessor — one AudioWorkletProcessor that owns every voice.
 *
 * Runs on the audio thread at sample rate. Never starts or stops anything;
 * "triggering a voice" is just a message that jumps the target amplitude
 * back up to the peak. All parameter changes (freq, amp, timbre) glide
 * smoothly per-sample toward their targets, so there is nothing anywhere
 * in the audio path that can produce a click.
 *
 * Message protocol from main thread (via `node.port.postMessage`):
 *   { kind: 'trigger', slot, freq, peak, triMix, sustainSec }
 *   { kind: 'set-master', gain }
 *   { kind: 'fade-all', factor }
 *
 * This file is bundled by Vite as an ES-module worklet (see
 * `audio-graph.ts` for the `?worker&url` import). Because it's compiled
 * from TypeScript, it can import shared tunables from `./constants`
 * directly — no source-string trickery, no duplicated magic values.
 */

import {
	WORKLET_AMP_FLOOR,
	WORKLET_AMP_GLIDE_TAU_SEC,
	WORKLET_DEFAULT_SLOT_FREQ_HZ,
	WORKLET_FREQ_GLIDE_TAU_SEC,
	WORKLET_INTERNAL_GAIN,
	WORKLET_MIX_GLIDE_TAU_SEC,
	WORKLET_NODE_NAME,
	WORKLET_SLOT_COUNT,
} from "../constants"

// AudioWorkletGlobalScope globals (not in the DOM lib because they only
// exist inside a worklet). Declared locally so this file remains a
// module and the names don't leak into the main-thread build.
declare const sampleRate: number
declare class AudioWorkletProcessor {
	readonly port: MessagePort
	constructor()
}
declare function registerProcessor(
	name: string,
	ctor: new () => AudioWorkletProcessor,
): void

interface Slot {
	phase: number
	freq: number
	targetFreq: number
	amp: number
	targetAmp: number
	triMix: number
	targetTriMix: number
	decayPerSample: number
}

type TriggerMessage = {
	kind: "trigger"
	slot: number
	freq: number
	peak: number
	triMix: number
	sustainSec: number
}
type SetMasterMessage = { kind: "set-master"; gain: number }
type FadeAllMessage = { kind: "fade-all"; factor: number }
type InboundMessage = TriggerMessage | SetMasterMessage | FadeAllMessage

class PolyVoiceProcessor extends AudioWorkletProcessor {
	private readonly slots: Slot[]
	private masterGain = WORKLET_INTERNAL_GAIN
	private readonly freqGlide: number
	private readonly ampGlide: number
	private readonly mixGlide: number

	constructor() {
		super()
		this.slots = new Array(WORKLET_SLOT_COUNT)
		for (let i = 0; i < WORKLET_SLOT_COUNT; i++) {
			this.slots[i] = {
				phase: 0,
				freq: WORKLET_DEFAULT_SLOT_FREQ_HZ,
				targetFreq: WORKLET_DEFAULT_SLOT_FREQ_HZ,
				amp: 0,
				targetAmp: 0,
				triMix: 0,
				targetTriMix: 0,
				decayPerSample: 1, // 1 = no decay until first trigger
			}
		}

		// Pre-compute per-sample glide coefficients (constant over the session).
		const sr = sampleRate
		this.freqGlide = 1 - Math.exp(-1 / (WORKLET_FREQ_GLIDE_TAU_SEC * sr))
		this.ampGlide = 1 - Math.exp(-1 / (WORKLET_AMP_GLIDE_TAU_SEC * sr))
		this.mixGlide = 1 - Math.exp(-1 / (WORKLET_MIX_GLIDE_TAU_SEC * sr))

		this.port.onmessage = (e: MessageEvent<InboundMessage>) => {
			const msg = e.data
			if (msg.kind === "trigger") {
				const s = this.slots[msg.slot]
				if (!s) return
				s.targetFreq = msg.freq
				s.targetTriMix = msg.triMix
				s.targetAmp = msg.peak
				// Exponential decay: targetAmp reaches WORKLET_AMP_FLOOR after sustainSec.
				// factor^N = WORKLET_AMP_FLOOR/peak  →  factor = (WORKLET_AMP_FLOOR/peak)^(1/N)
				const N = Math.max(1, msg.sustainSec * sr)
				s.decayPerSample = Math.pow(WORKLET_AMP_FLOOR / msg.peak, 1 / N)
			} else if (msg.kind === "set-master") {
				this.masterGain = msg.gain
			} else if (msg.kind === "fade-all") {
				// Scale every slot's targetAmp so the whole mix ducks — used
				// as a musical breath between sections when the key changes.
				const factor = Math.max(0, Math.min(1, msg.factor))
				for (const s of this.slots) s.targetAmp *= factor
			}
		}
	}

	process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
		const output = outputs[0]
		if (!output || output.length === 0) return true
		const channel = output[0]
		const nSamples = channel.length
		const master = this.masterGain
		const slots = this.slots
		const freqGlide = this.freqGlide
		const ampGlide = this.ampGlide
		const mixGlide = this.mixGlide
		const sr = sampleRate
		const twoPi = Math.PI * 2

		for (let i = 0; i < nSamples; i++) {
			let sum = 0
			for (let si = 0; si < WORKLET_SLOT_COUNT; si++) {
				const s = slots[si]

				// Decay the envelope target toward the floor
				s.targetAmp *= s.decayPerSample

				// Glide rendered params toward targets
				s.freq += (s.targetFreq - s.freq) * freqGlide
				s.amp += (s.targetAmp - s.amp) * ampGlide
				s.triMix += (s.targetTriMix - s.triMix) * mixGlide

				// Skip inaudible slots
				if (s.amp < WORKLET_AMP_FLOOR && s.targetAmp < WORKLET_AMP_FLOOR) continue

				// Advance phase
				s.phase += s.freq / sr
				if (s.phase >= 1) s.phase -= 1

				// Blend sine ⇄ triangle
				const sine = Math.sin(s.phase * twoPi)
				const tri = s.phase < 0.5 ? -1 + 4 * s.phase : 3 - 4 * s.phase
				sum += (sine * (1 - s.triMix) + tri * s.triMix) * s.amp
			}
			channel[i] = sum * master
		}

		// Mirror to any additional channels
		for (let c = 1; c < output.length; c++) {
			output[c].set(channel)
		}
		return true
	}
}

registerProcessor(WORKLET_NODE_NAME, PolyVoiceProcessor)
