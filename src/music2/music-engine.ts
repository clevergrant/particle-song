import type { OrganelleState } from "../detection"
import { AudioGraph2 } from "./audio-graph"
import {
	BLEND_RANGE,
	CHORD_CYCLE_RANGE,
	KEY_TRANSITION_FADE,
	LOOPS_TO_RANDOMIZE_RANGE,
	MIN_NOTE_GAP_RANGE,
	MUSIC_DEFAULT_SETTINGS,
	SUSTAIN_MIN_FLOOR_SEC,
} from "../constants"
import { buildProgression, deriveKey, pickQuality } from "./key-derivation"
import { computeNoteParams } from "./organelle-signals"
import type {
	ChordSpec,
	HarmonicContext,
	KeyDerivation,
	MusicSettings,
} from "./types"
import { clamp } from "./utils"

export interface FormationEvent {
	readonly typeId: number
	readonly organelleId: number
	readonly particleIndices: Uint32Array
	readonly sustainSec: number
	readonly midi: number
}

/**
 * Top-level orchestrator. Owns the audio graph, the derived key, the
 * per-type chord progression, and the per-type "count high-water" state.
 *
 * Trigger discipline (two-tick confirmation):
 *   A type's voice re-attacks only when its organelle count has been
 *   above its previous high-water mark for at least TWO consecutive
 *   detection ticks — single-tick blips never fire.
 *
 * Progression:
 *   One chord per particle type, walked in type order. Each chord is
 *   picked from the previous one plus how friendly the two adjacent
 *   types are in the matrix (see `pickNextChord`). Every
 *   `chordCycleSec` seconds the engine advances to the next slot.
 *
 * Randomize request:
 *   After `loopsToRandomize` complete passes through the progression the
 *   engine calls `onRandomizeRequest`. The outer sim listens and
 *   re-randomizes the force matrix — new matrix → new tonic + progression
 *   fall out naturally. 0 disables the request.
 *
 * Key transition:
 *   When the force matrix produces a new tonic or mode, all active
 *   voices are ducked (KEY_TRANSITION_FADE) for one musical "breath"
 *   so the new key emerges through fresh triggers instead of jump-cutting.
 */
export class MusicEngine {
	private readonly audio = new AudioGraph2()
	private key: KeyDerivation = deriveKey([])
	private forceMatrixDim = 0
	private progression: readonly ChordSpec[] = []
	private harmony: HarmonicContext = {
		chord: { rootSemitones: 0, quality: pickQuality(0, this.key) },
		slotIdx: 0,
		cycleAt: 0,
	}
	private loopsCompleted = 0

	private readonly countHighWaterByType = new Map<number, number>()
	private readonly pendingCountByType = new Map<number, number>()
	private readonly settings: MusicSettings = { ...MUSIC_DEFAULT_SETTINGS }

	/** Fired when the progression has looped `loopsToRandomize` times.
	 *  Outer sim wires this up to re-randomize its force matrix. */
	onRandomizeRequest: (() => void) | null = null

	enable(): void {
		this.audio.enable()
		this.audio.setMinNoteGap(this.settings.minNoteGapSec)
		const t = this.audio.currentTime
		this.harmony = {
			chord: this.progression[0] ?? this.harmony.chord,
			slotIdx: 0,
			cycleAt: t + this.settings.chordCycleSec,
		}
		this.loopsCompleted = 0
		this.audio.diagnostics.recordSettingsSnapshot(t, this.settings)
		this.audio.diagnostics.recordKeyChange(t, this.key)
		this.audio.diagnostics.recordChordChange(t, this.harmony.chord.rootSemitones)
	}
	disable(): void {
		this.audio.disable()
		this.countHighWaterByType.clear()
		this.pendingCountByType.clear()
	}
	dispose(): void {
		this.audio.dispose()
		this.countHighWaterByType.clear()
		this.pendingCountByType.clear()
	}
	get isEnabled(): boolean {
		return this.audio.isEnabled
	}

	getKey(): KeyDerivation {
		return this.key
	}
	getSettings(): Readonly<MusicSettings> {
		return this.settings
	}
	getHarmony(): Readonly<HarmonicContext> {
		return this.harmony
	}
	getProgression(): readonly ChordSpec[] {
		return this.progression
	}

	setVolume(v: number): void {
		this.settings.volume = v
		this.audio.setVolume(v)
		this.logSettingsChange("volume", v)
	}
	setTonicMidi(m: number): void {
		this.settings.tonicMidi = m
		this.logSettingsChange("tonicMidi", m)
	}
	setSustainRange(minSec: number, maxSec: number): void {
		const lo = Math.max(SUSTAIN_MIN_FLOOR_SEC, Math.min(minSec, maxSec))
		const hi = Math.max(lo, maxSec)
		this.settings.sustainMinSec = lo
		this.settings.sustainMaxSec = hi
		this.logSettingsChange("sustainMinSec", lo)
		this.logSettingsChange("sustainMaxSec", hi)
	}
	setTimbreRange(minBlend: number, maxBlend: number): void {
		const lo = clamp(minBlend, BLEND_RANGE.min, BLEND_RANGE.max)
		const hi = clamp(Math.max(lo, maxBlend), BLEND_RANGE.min, BLEND_RANGE.max)
		this.settings.timbreMinBlend = lo
		this.settings.timbreMaxBlend = hi
		this.logSettingsChange("timbreMinBlend", lo)
		this.logSettingsChange("timbreMaxBlend", hi)
	}
	setMinNoteGap(sec: number): void {
		const clamped = clamp(sec, MIN_NOTE_GAP_RANGE.min, MIN_NOTE_GAP_RANGE.max)
		this.settings.minNoteGapSec = clamped
		this.audio.setMinNoteGap(clamped)
		this.logSettingsChange("minNoteGapSec", clamped)
	}
	setChordCycleSec(sec: number): void {
		const clamped = clamp(sec, CHORD_CYCLE_RANGE.min, CHORD_CYCLE_RANGE.max)
		this.settings.chordCycleSec = clamped
		this.logSettingsChange("chordCycleSec", clamped)
	}
	setLoopsToRandomize(n: number): void {
		const clamped = Math.round(
			clamp(n, LOOPS_TO_RANDOMIZE_RANGE.min, LOOPS_TO_RANDOMIZE_RANGE.max),
		)
		this.settings.loopsToRandomize = clamped
		this.loopsCompleted = 0
		this.logSettingsChange("loopsToRandomize", clamped)
	}

	/**
	 * Called by the sim whenever the force matrix changes. Rebuilds the
	 * per-type progression. If the derived key or mode changed too, fades
	 * every active voice as a musical breath into the new key.
	 */
	setForceMatrix(matrix: readonly (readonly number[])[]): void {
		const prevKey = this.key
		const newKey = deriveKey(matrix)
		this.key = newKey
		this.forceMatrixDim = matrix.length
		this.progression = buildProgression(matrix, newKey)
		const keyMoved =
			prevKey.tonicTypeIdx !== newKey.tonicTypeIdx ||
			prevKey.modeName !== newKey.modeName
		if (keyMoved) {
			const t = this.audio.currentTime
			this.audio.diagnostics.recordKeyChange(t, newKey)
			if (this.audio.isEnabled) {
				this.audio.fadeAll(KEY_TRANSITION_FADE)
				this.audio.diagnostics.recordKeyTransition(t, prevKey, newKey)
				// Reset the progression to slot 0 so the new key emerges
				// starting from the tonic chord.
				this.loopsCompleted = 0
				const chord = this.progression[0] ?? this.harmony.chord
				this.harmony = {
					chord,
					slotIdx: 0,
					cycleAt: t + this.settings.chordCycleSec,
				}
				this.audio.diagnostics.recordChordChange(t, chord.rootSemitones)
			}
		}
	}

	/**
	 * Called each frame with the current organelle set from detection.
	 * Returns the formation events that actually fired a voice this tick
	 * (post-debounce, post-confirmation) so the caller can drive
	 * organelle-pulse visualisations.
	 */
	tick(organelles: readonly OrganelleState[]): FormationEvent[] {
		if (!this.audio.isEnabled) return []

		// Advance to the next chord slot if the timer has run out.
		const t = this.audio.currentTime
		const progLen = this.progression.length
		if (progLen > 0 && t >= this.harmony.cycleAt) {
			const nextSlot = (this.harmony.slotIdx + 1) % progLen
			const chord = this.progression[nextSlot]
			this.harmony = {
				chord,
				slotIdx: nextSlot,
				cycleAt: t + this.settings.chordCycleSec,
			}
			this.audio.diagnostics.recordChordChange(t, chord.rootSemitones)
			// Wrap-around → one full loop completed.
			if (nextSlot === 0) {
				this.loopsCompleted++
				const target = this.settings.loopsToRandomize
				if (target > 0 && this.loopsCompleted >= target) {
					this.loopsCompleted = 0
					this.onRandomizeRequest?.()
				}
			}
		}

		// Bucket by type; keep the largest organelle in each bucket.
		const byType = new Map<number, { count: number; largest: OrganelleState }>()
		for (const org of organelles) {
			const entry = byType.get(org.typeId)
			if (!entry) {
				byType.set(org.typeId, { count: 1, largest: org })
				continue
			}
			entry.count++
			if (org.particleIndices.length > entry.largest.particleIndices.length) {
				entry.largest = org
			}
		}

		const events: FormationEvent[] = []

		for (const [typeId, { count, largest }] of byType) {
			const highWater = this.countHighWaterByType.get(typeId) ?? 0
			const pending = this.pendingCountByType.get(typeId) ?? null

			if (count > highWater) {
				if (pending !== null && pending > highWater) {
					this.countHighWaterByType.set(typeId, count)
					this.pendingCountByType.set(typeId, count)
					const params = computeNoteParams(
						largest,
						this.key,
						Math.max(1, this.forceMatrixDim),
						this.harmony,
						this.settings,
					)
					const fired = this.audio.triggerVoice(typeId, params)
					if (fired) {
						events.push({
							typeId,
							organelleId: largest.id,
							particleIndices: largest.particleIndices,
							sustainSec: params.sustainSec,
							midi: params.midiNote,
						})
					}
				} else {
					this.pendingCountByType.set(typeId, count)
				}
			} else {
				this.pendingCountByType.set(typeId, count)
				if (count < highWater) this.countHighWaterByType.set(typeId, count)
			}
		}

		// Types absent this tick reset both trackers.
		for (const typeId of this.countHighWaterByType.keys()) {
			if (!byType.has(typeId)) {
				this.countHighWaterByType.set(typeId, 0)
				this.pendingCountByType.set(typeId, 0)
			}
		}

		return events
	}

	private logSettingsChange(field: keyof MusicSettings, value: number): void {
		if (this.audio.isEnabled) {
			this.audio.diagnostics.recordSettingsChange(
				this.audio.currentTime,
				field,
				value,
			)
		}
	}
}
