/**
 * Diagnostic recorder for music2. Streams every voice trigger, every
 * settings change, and a ~20 Hz sample of both pre- and post-compressor
 * peak levels to `music-log.jsonl` via the Vite dev plugin at
 * POST /__music-log.
 *
 * Records are batched and flushed on a short interval so we don't drown
 * the dev server with one request per event. On disable, a final flush
 * ships anything still pending.
 *
 * Everything here is dev-only tooling; production builds still function
 * (the POST will simply 404) and no audio path depends on it.
 */

import {
	DIAG_ENDPOINT,
	DIAG_FLUSH_INTERVAL_MS,
	DIAG_RESET_ENDPOINT,
	DIAG_ROUND_FACTOR,
} from "../constants"
import type { KeyDerivation, MusicSettings } from "./types"

export interface TriggerRecord {
	readonly kind: "trigger"
	readonly t: number
	readonly typeId: number
	readonly midi: number
	readonly freq: number
	readonly sustainSec: number
	readonly triangleBlend: number
	readonly deltaSincePrev: number | null
}

export interface VoiceCreateRecord {
	readonly kind: "voice-create"
	readonly t: number
	readonly typeId: number
}

export interface LifecycleRecord {
	readonly kind: "audio-enable" | "audio-disable"
	readonly t: number
}

export interface PeakRecord {
	readonly kind: "peak"
	readonly t: number
	readonly rawPeak: number
	readonly outputPeak: number
}

export interface SettingsSnapshotRecord {
	readonly kind: "settings-snapshot"
	readonly t: number
	readonly settings: MusicSettings
}

export interface SettingsChangeRecord {
	readonly kind: "settings-change"
	readonly t: number
	readonly field: keyof MusicSettings
	readonly value: number
}

export interface KeyChangeRecord {
	readonly kind: "key-change"
	readonly t: number
	readonly tonicTypeIdx: number
	readonly modeName: string
	readonly modeIntervals: readonly number[]
}

export interface ChordChangeRecord {
	readonly kind: "chord-change"
	readonly t: number
	/** Semitones from tonic, 0..11. */
	readonly chordRoot: number
}

export interface TransitionRecord {
	readonly kind: "key-transition"
	readonly t: number
	readonly fromTonicTypeIdx: number
	readonly toTonicTypeIdx: number
	readonly fromMode: string
	readonly toMode: string
}

export type Record =
	| TriggerRecord
	| VoiceCreateRecord
	| LifecycleRecord
	| PeakRecord
	| SettingsSnapshotRecord
	| SettingsChangeRecord
	| KeyChangeRecord
	| ChordChangeRecord
	| TransitionRecord

export class Diagnostics {
	private enabled = false
	private lastTriggerByType = new Map<number, number>()
	private pending: Record[] = []
	private flushTimerId: number | null = null

	enable(t: number): void {
		this.enabled = true
		this.pending = []
		this.lastTriggerByType.clear()
		void this.reset()
		this.push({ kind: "audio-enable", t })
		this.startFlushTimer()
	}

	disable(t: number): void {
		if (!this.enabled) return
		this.push({ kind: "audio-disable", t })
		this.stopFlushTimer()
		void this.flush()
		this.enabled = false
	}

	get isEnabled(): boolean {
		return this.enabled
	}

	recordTrigger(
		t: number,
		typeId: number,
		midi: number,
		freq: number,
		sustainSec: number,
		triangleBlend: number,
	): void {
		if (!this.enabled) return
		const prev = this.lastTriggerByType.get(typeId) ?? null
		const deltaSincePrev = prev == null ? null : +(t - prev).toFixed(4)
		this.lastTriggerByType.set(typeId, t)
		this.push({
			kind: "trigger",
			t: round(t),
			typeId,
			midi,
			freq: round(freq),
			sustainSec: round(sustainSec),
			triangleBlend: round(triangleBlend),
			deltaSincePrev,
		})
	}

	recordVoiceCreate(t: number, typeId: number): void {
		if (!this.enabled) return
		this.push({ kind: "voice-create", t: round(t), typeId })
	}

	recordPeaks(t: number, rawPeak: number, outputPeak: number): void {
		if (!this.enabled) return
		this.push({
			kind: "peak",
			t: round(t),
			rawPeak: round(rawPeak),
			outputPeak: round(outputPeak),
		})
	}

	recordSettingsSnapshot(t: number, settings: MusicSettings): void {
		if (!this.enabled) return
		this.push({ kind: "settings-snapshot", t: round(t), settings: { ...settings } })
	}

	recordSettingsChange(
		t: number,
		field: keyof MusicSettings,
		value: number,
	): void {
		if (!this.enabled) return
		this.push({
			kind: "settings-change",
			t: round(t),
			field,
			value: round(value),
		})
	}

	recordKeyChange(t: number, key: KeyDerivation): void {
		if (!this.enabled) return
		this.push({
			kind: "key-change",
			t: round(t),
			tonicTypeIdx: key.tonicTypeIdx,
			modeName: key.modeName,
			modeIntervals: key.modeIntervals,
		})
	}

	recordChordChange(t: number, chordRoot: number): void {
		if (!this.enabled) return
		this.push({ kind: "chord-change", t: round(t), chordRoot })
	}

	recordKeyTransition(
		t: number,
		from: KeyDerivation,
		to: KeyDerivation,
	): void {
		if (!this.enabled) return
		this.push({
			kind: "key-transition",
			t: round(t),
			fromTonicTypeIdx: from.tonicTypeIdx,
			toTonicTypeIdx: to.tonicTypeIdx,
			fromMode: from.modeName,
			toMode: to.modeName,
		})
	}

	private push(record: Record): void {
		this.pending.push(record)
	}

	private startFlushTimer(): void {
		this.stopFlushTimer()
		this.flushTimerId = window.setInterval(() => {
			void this.flush()
		}, DIAG_FLUSH_INTERVAL_MS)
	}

	private stopFlushTimer(): void {
		if (this.flushTimerId != null) {
			window.clearInterval(this.flushTimerId)
			this.flushTimerId = null
		}
	}

	private async flush(): Promise<void> {
		if (this.pending.length === 0) return
		const batch = this.pending
		this.pending = []
		const body = batch.map((r) => JSON.stringify(r)).join("\n")
		try {
			await fetch(DIAG_ENDPOINT, { method: "POST", body, keepalive: true })
		} catch {
			/* dev endpoint unavailable (prod build) — silently drop */
		}
	}

	private async reset(): Promise<void> {
		try {
			await fetch(DIAG_RESET_ENDPOINT, { method: "POST" })
		} catch {
			/* dev endpoint unavailable — fine */
		}
	}
}

function round(n: number): number {
	return Math.round(n * DIAG_ROUND_FACTOR) / DIAG_ROUND_FACTOR
}
