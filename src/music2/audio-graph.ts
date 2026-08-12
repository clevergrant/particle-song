import {
	BASE_MASTER_GAIN,
	MIN_NOTE_GAP_RANGE,
	VOICE_PEAK,
	VOLUME_RAMP_TAU_SEC,
	VOLUME_RANGE,
	WORKLET_NODE_NAME,
} from "../constants"
import { Diagnostics } from "./diagnostics"
import type { NoteParams } from "./types"
import { clamp, midiToFreq } from "./utils"
// Vite bundles the worklet as a separate ES-module chunk and returns the
// URL of the emitted file. The worklet code can import from the rest of
// the project (e.g. `./constants`) — Vite inlines everything the worklet
// touches into its own bundle at build time.
import voiceProcessorUrl from "./voice-processor.ts?worker&url"

/**
 * Web Audio wrapper. The entire synthesis lives inside a single
 * PolyVoiceProcessor running in an AudioWorklet — this file's only job
 * is to load the worklet, own the master gain, and post trigger messages.
 *
 * There is nothing in the audio path that can produce a click:
 *  - no OscillatorNode start/stop
 *  - no AudioParam automation ramps
 *  - all glides happen sample-by-sample inside the worklet
 */
export class AudioGraph2 {
	private ctx: AudioContext | null = null
	private masterGain: GainNode | null = null
	private polyVoice: AudioWorkletNode | null = null
	private workletReady = false
	private workletLoadInFlight: Promise<void> | null = null
	private lastTriggerTime = new Map<number, number>()
	private volume = 0.5
	private minNoteGapSec = 0.6
	private enabled = false
	readonly diagnostics = new Diagnostics()

	enable(): void {
		if (this.ctx) {
			if (this.ctx.state === "suspended") this.ctx.resume()
			this.enabled = true
			this.diagnostics.enable(this.ctx.currentTime)
			return
		}
		this.ctx = new AudioContext()
		this.masterGain = this.ctx.createGain()
		this.masterGain.gain.value = BASE_MASTER_GAIN * this.volume
		this.masterGain.connect(this.ctx.destination)
		if (this.ctx.state === "suspended") this.ctx.resume()
		this.enabled = true
		this.diagnostics.enable(this.ctx.currentTime)
		void this.loadWorklet()
	}

	disable(): void {
		this.enabled = false
		if (this.ctx) {
			this.diagnostics.disable(this.ctx.currentTime)
			if (this.ctx.state === "running") this.ctx.suspend()
		}
	}

	dispose(): void {
		this.enabled = false
		if (this.polyVoice) {
			this.polyVoice.port.onmessage = null
			this.polyVoice.disconnect()
			this.polyVoice = null
		}
		this.workletReady = false
		this.workletLoadInFlight = null
		this.lastTriggerTime.clear()
		if (this.ctx) {
			this.ctx.close()
			this.ctx = null
			this.masterGain = null
		}
	}

	get isEnabled(): boolean {
		return this.enabled
	}
	get currentTime(): number {
		return this.ctx?.currentTime ?? 0
	}

	setVolume(v: number): void {
		this.volume = clamp(v, VOLUME_RANGE.min, VOLUME_RANGE.max)
		if (this.masterGain) {
			this.masterGain.gain.setTargetAtTime(
				BASE_MASTER_GAIN * this.volume,
				this.ctx?.currentTime ?? 0,
				VOLUME_RAMP_TAU_SEC,
			)
		}
	}

	setMinNoteGap(sec: number): void {
		this.minNoteGapSec = Math.max(MIN_NOTE_GAP_RANGE.min, sec)
	}

	/**
	 * Duck every active voice by `factor` (0..1). Used as a musical
	 * "breath" between sections — call e.g. with 0.25 when the key
	 * changes so the new context re-emerges naturally through fresh
	 * triggers instead of jump-cutting from the old one.
	 */
	fadeAll(factor: number): void {
		if (!this.workletReady || !this.polyVoice) return
		this.polyVoice.port.postMessage({ kind: "fade-all", factor })
	}

	/**
	 * Re-attack the voice for `typeId`. Returns `true` if the message was
	 * posted to the worklet, `false` if it was coalesced by the debounce
	 * or the worklet isn't ready yet. Callers use the return to know
	 * whether to emit a visual pulse.
	 */
	triggerVoice(typeId: number, params: NoteParams): boolean {
		if (!this.enabled || !this.ctx || !this.workletReady || !this.polyVoice) {
			return false
		}
		const t = this.ctx.currentTime
		const last = this.lastTriggerTime.get(typeId) ?? -Infinity
		if (t - last < this.minNoteGapSec) return false
		this.lastTriggerTime.set(typeId, t)
		const freq = midiToFreq(params.midiNote)
		this.diagnostics.recordTrigger(
			t,
			typeId,
			params.midiNote,
			freq,
			params.sustainSec,
			params.triangleBlend,
		)
		this.polyVoice.port.postMessage({
			kind: "trigger",
			slot: typeId,
			freq,
			peak: VOICE_PEAK,
			triMix: params.triangleBlend,
			sustainSec: params.sustainSec,
		})
		return true
	}

	private async loadWorklet(): Promise<void> {
		if (!this.ctx) return
		if (this.workletReady || this.workletLoadInFlight) {
			await this.workletLoadInFlight
			return
		}
		const ctx = this.ctx
		this.workletLoadInFlight = (async () => {
			await ctx.audioWorklet.addModule(voiceProcessorUrl)
			if (!this.ctx || this.ctx !== ctx) return // disposed mid-load
			this.polyVoice = new AudioWorkletNode(ctx, WORKLET_NODE_NAME, {
				numberOfInputs: 0,
				numberOfOutputs: 1,
				outputChannelCount: [1],
			})
			if (this.masterGain) this.polyVoice.connect(this.masterGain)
			this.workletReady = true
		})()
		await this.workletLoadInFlight
	}
}
