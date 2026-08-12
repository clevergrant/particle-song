import {
	CHORD_MOVE_SEMITONES,
	CHORD_MOVE_THRESHOLDS,
	DEFAULT_MODE_NAME,
	MODES,
} from "../constants"
import type { ChordQuality, ChordSpec, KeyDerivation } from "./types"

const DEFAULT_MODE = MODES.find((m) => m.name === DEFAULT_MODE_NAME) ?? MODES[0]

const EMPTY_KEY: KeyDerivation = {
	tonicTypeIdx: 0,
	modeName: DEFAULT_MODE.name,
	modeIntervals: DEFAULT_MODE.intervals,
}

/**
 * Derive a key + mode from a force matrix.
 *
 *   Tonic = friendliest type — the one whose outgoing forces sum to the
 *           most net attraction. Feels warm/outward: the type that likes
 *           everyone else the most anchors the key.
 *   Mode  = brightness read from the tonic's outgoing row. Magnitude-
 *           weighted: signedSum / absSum ∈ [-1, +1] captures both sign
 *           and intensity. All-friendly outgoing → Lydian; all-hostile →
 *           Locrian; mixed → mid modes.
 *
 *   We deliberately use only the tonic's row (not the whole matrix).
 *   Whole-matrix means concentrate near 0.5 for randomized matrices
 *   (law of large numbers over ~N² cells) and pin the mode to Dorian.
 *   The tonic's row is small enough to actually vary, and reading "how
 *   home feels about everyone" is the more musical statistic anyway.
 *
 * O(N²) in the number of types, evaluated only when the matrix changes.
 */
export function deriveKey(
	matrix: readonly (readonly number[])[],
): KeyDerivation {
	const n = matrix.length
	if (n === 0) return EMPTY_KEY

	let tonicTypeIdx = 0
	let bestRowSum = -Infinity
	for (let i = 0; i < n; i++) {
		let rowSum = 0
		for (let j = 0; j < n; j++) {
			if (i === j) continue
			rowSum += matrix[i]?.[j] ?? 0
		}
		if (rowSum > bestRowSum) {
			bestRowSum = rowSum
			tonicTypeIdx = i
		}
	}

	let signedSum = 0
	let absSum = 0
	for (let j = 0; j < n; j++) {
		if (j === tonicTypeIdx) continue
		const v = matrix[tonicTypeIdx]?.[j] ?? 0
		signedSum += v
		absSum += Math.abs(v)
	}
	// signedSum/absSum ∈ [-1, +1]; map to brightness ∈ [0, 1].
	const brightness = absSum > 0 ? (signedSum / absSum + 1) / 2 : 0.5

	// brightness ∈ [0, 1] → mode index (bright=0 → dark=len-1)
	const idx = Math.max(
		0,
		Math.min(MODES.length - 1, Math.round((1 - brightness) * (MODES.length - 1))),
	)

	return {
		tonicTypeIdx,
		modeName: MODES[idx].name,
		modeIntervals: MODES[idx].intervals,
	}
}

/**
 * Pick a triad quality that keeps the third and fifth inside the current
 * mode's pitch classes when possible. Falls back to a diatonic guess for
 * chromatic roots so the chord still sounds like it belongs.
 */
export function pickQuality(
	rootSemitones: number,
	key: KeyDerivation,
): ChordQuality {
	const modeSet = new Set(key.modeIntervals.map((s) => s % 12))
	const has3 = modeSet.has((rootSemitones + 3) % 12)
	const has4 = modeSet.has((rootSemitones + 4) % 12)
	const has6 = modeSet.has((rootSemitones + 6) % 12)
	const has7 = modeSet.has((rootSemitones + 7) % 12)
	if (has4 && has7) return "maj"
	if (has3 && has7) return "min"
	if (has3 && has6) return "dim"
	return has4 ? "maj" : "min"
}

/**
 * Pick the next chord given the previous chord and a friendliness score
 * in [-1, +1]. Friendly moves stay close (fifth/fourth motion); hostile
 * moves leap far (tritone, borrowed bVI). Roots stay in [0, 12), quality
 * comes from `pickQuality` so distant roots still fit the current mode.
 */
export function pickNextChord(
	prev: ChordSpec,
	friendliness: number,
	key: KeyDerivation,
): ChordSpec {
	const t = CHORD_MOVE_THRESHOLDS
	const s = CHORD_MOVE_SEMITONES
	const step =
		friendliness > t.veryFriendly
			? s.veryFriendly
			: friendliness > t.friendly
				? s.friendly
				: friendliness > t.hostile
					? s.neutral
					: friendliness > t.veryHostile
						? s.hostile
						: s.veryHostile
	const rootSemitones = (prev.rootSemitones + step) % 12
	return { rootSemitones, quality: pickQuality(rootSemitones, key) }
}

/**
 * Build the full per-type chord progression from the current matrix.
 * Type 0's chord is derived from how friendly type 0 is with the tonic
 * type; each subsequent type's chord is derived from its friendliness
 * with the previous type. Friendliness is symmetric — the mean of both
 * matrix directions between the two types.
 *
 * O(N) in the number of types. Called only when the matrix changes.
 */
export function buildProgression(
	matrix: readonly (readonly number[])[],
	key: KeyDerivation,
): readonly ChordSpec[] {
	const n = matrix.length
	if (n === 0) return []
	const progression: ChordSpec[] = new Array(n)
	const tonicChord: ChordSpec = {
		rootSemitones: 0,
		quality: pickQuality(0, key),
	}
	let prev = tonicChord
	let prevType = key.tonicTypeIdx
	for (let i = 0; i < n; i++) {
		const friendliness =
			((matrix[prevType]?.[i] ?? 0) + (matrix[i]?.[prevType] ?? 0)) / 2
		const chord =
			i === key.tonicTypeIdx ? tonicChord : pickNextChord(prev, friendliness, key)
		progression[i] = chord
		prev = chord
		prevType = i
	}
	return progression
}
