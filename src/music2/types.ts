/**
 * Public types for the music2 engine.
 *
 * The whole engine is sim-driven: each frame it observes the current
 * organelle set, spots newly-formed organelles per type, and re-attacks
 * a single mono voice per type. Rhythm, pitch, sustain, and timbre are
 * all emergent from simulation state — there is no grid, tempo, or
 * schedule of any kind.
 *
 * All defaults, ranges, and tunables live in `./constants`.
 */

export interface KeyDerivation {
	/** The type whose net outgoing force is strongest — its position defines "home". */
	readonly tonicTypeIdx: number
	readonly modeName: string
	/** Semitone offsets from the tonic, one per scale degree. */
	readonly modeIntervals: readonly number[]
}

export interface MusicSettings {
	/** 0..1 master gain. */
	volume: number
	/** MIDI note of the tonic (e.g. 48 = C3). */
	tonicMidi: number
	/** Sustain seconds for the fastest-moving organelles. */
	sustainMinSec: number
	/** Sustain seconds for the slowest-moving organelles. */
	sustainMaxSec: number
	/** Waveform blend for the most compact (least stressed) organelles. 0 = pure sine. */
	timbreMinBlend: number
	/** Waveform blend for the most loose (most stressed) organelles. 1 = pure triangle. */
	timbreMaxBlend: number
	/** Minimum seconds between successive notes on the same voice. Also
	 *  the coarse tempo control — larger gap = sparser music. */
	minNoteGapSec: number
	/** Seconds each chord holds before the harmonic context advances to
	 *  the next chord in the progression. */
	chordCycleSec: number
	/** Full progression loops before the engine requests a fresh matrix
	 *  from the outer sim. 0 = never request (progression just cycles). */
	loopsToRandomize: number
}

export interface NoteParams {
	readonly midiNote: number
	readonly sustainSec: number
	/** 0 = pure sine, 1 = pure triangle. */
	readonly triangleBlend: number
}

export type ChordQuality = "maj" | "min" | "dim"

/**
 * A single chord in the per-type progression. Root is in semitones from
 * the tonic (0..11) — this allows chromatic movements like tritone and
 * borrowed-chord roots that scale degrees can't express directly.
 */
export interface ChordSpec {
	readonly rootSemitones: number
	readonly quality: ChordQuality
}

/**
 * Rotating harmonic context. The progression has one chord per particle
 * type, walked in type order. Each chord is picked based on how friendly
 * the current type is with the next — friendly = close move, hostile =
 * far move (see CHORD_MOVE_* in constants).
 */
export interface HarmonicContext {
	readonly chord: ChordSpec
	/** Which slot of the per-type progression is currently sounding. */
	readonly slotIdx: number
	/** AudioContext time when this chord should give way to the next. */
	readonly cycleAt: number
}
