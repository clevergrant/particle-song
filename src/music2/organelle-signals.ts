import type { OrganelleState } from "../detection"
import {
	LOOSENESS_REFERENCE,
	MAX_SIZE_OCTAVE_DROP,
	SIZE_OCTAVE_DROP_THRESHOLD,
	SIZE_REFERENCE,
	SPEED_REFERENCE,
} from "../constants"
import type {
	HarmonicContext,
	KeyDerivation,
	MusicSettings,
	NoteParams,
} from "./types"
import { clamp01, clampMidi, lerp } from "./utils"

/** bbox-area / particle-count. Compact ≈ 1, loose ≫ 1. */
export function computeLooseness(org: OrganelleState): number {
	const w = org.maxCol - org.minCol + 1
	const h = org.maxRow - org.minRow + 1
	const area = Math.max(1, w * h)
	return area / Math.max(1, org.particleIndices.length)
}

/** Centroid speed in simulation units per second. */
export function computeCentroidSpeed(org: OrganelleState): number {
	return Math.hypot(org.avgVelX, org.avgVelY)
}

/**
 * Turn an organelle into a note. Called once per formation event, so this
 * runs O(new organelles) per tick — bounded, cheap.
 *
 * Pitch model: the ensemble sits on one chord (root/3rd/5th) whose root
 * and quality are chosen by the current progression slot. Each type
 * occupies a fixed role in the chord — `voicePos = typeDegree % 3` picks
 * root vs third vs fifth, and `octaveLayer = floor(typeDegree / 3)`
 * stacks types across octaves. As the progression walks to the next
 * chord every `chordCycleSec`, every voice's pitch shifts to match —
 * same type sounds different in different chords.
 */
export function computeNoteParams(
	org: OrganelleState,
	key: KeyDerivation,
	numTypes: number,
	harmony: HarmonicContext,
	settings: MusicSettings,
): NoteParams {
	const nt = Math.max(1, numTypes)
	const degree = (((org.typeId - key.tonicTypeIdx) % nt) + nt) % nt

	// Triad intervals from the chord's root, in semitones.
	const q = harmony.chord.quality
	const thirdInterval = q === "maj" ? 4 : 3
	const fifthInterval = q === "dim" ? 6 : 7
	const chordOffsets = [0, thirdInterval, fifthInterval]

	const voicePos = degree % 3
	const octaveLayer = Math.floor(degree / 3)

	// Bigger organelle = one octave down, but only past a threshold so
	// small size variation doesn't jump register on every trigger.
	const size = org.particleIndices.length
	const sizeOctaves =
		size >= SIZE_REFERENCE * SIZE_OCTAVE_DROP_THRESHOLD
			? -MAX_SIZE_OCTAVE_DROP
			: 0

	const midiNote = clampMidi(
		settings.tonicMidi +
			harmony.chord.rootSemitones +
			chordOffsets[voicePos] +
			12 * (octaveLayer + sizeOctaves),
	)

	// Sustain: slow centroid = long note, fast = short
	const speedT = clamp01(computeCentroidSpeed(org) / SPEED_REFERENCE)
	const sustainSec = lerp(settings.sustainMaxSec, settings.sustainMinSec, speedT)

	// Timbre: compact = pure sine, loose = triangle
	const looseT = clamp01((computeLooseness(org) - 1) / LOOSENESS_REFERENCE)
	const triangleBlend = lerp(
		settings.timbreMinBlend,
		settings.timbreMaxBlend,
		looseT,
	)

	return { midiNote, sustainSec, triangleBlend }
}
