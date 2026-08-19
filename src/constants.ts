/**
 * Central tunables for the entire project.
 *
 * Every numeric/behavioural knob lives here so re-shaping the feel of
 * the sim, detection, music, or UI interaction is a one-file edit.
 *
 * What DOES belong here:
 *   scalars, thresholds, references, limits, default-config objects,
 *   scale/mode tables, endpoint strings — anything you'd tune.
 *
 * What does NOT belong here:
 *   inline CSS, inline WGSL source, WebGPU pipeline-state objects,
 *   per-widget SVG geometry, storage keys — those stay next to the
 *   component that owns them.
 */

import type { DetectionConfig } from "./detection"
import type { MusicSettings } from "./music2/types"

// ══════════════════════════════════════════════════════════════════════
//  Simulation
// ══════════════════════════════════════════════════════════════════════

export const MAX_PARTICLES = 10000
/** Cap on distinct particle types. Also mirrored in
 *  `src/shaders/particles.compute.wgsl` (const `MAX_TYPES = 32u`) and
 *  in `WORKLET_SLOT_COUNT` below — the audio worklet needs one slot
 *  per possible type. Bump all three together. */
export const MAX_TYPES = 32
/** Particles per pixel at scale 1.0. */
export const DENSITY_TARGET = 0.0017
export const MIN_PER_ACTIVE_TYPE = 300
/** Bytes per Particle struct — must match the WGSL layout. */
export const PARTICLE_STRIDE = 48
export const WORKGROUP_SIZE = 64

// ══════════════════════════════════════════════════════════════════════
//  Physics / auto-balance
// ══════════════════════════════════════════════════════════════════════

export const AUTO_BASE_RADIUS_FACTOR = 4.0
export const AUTO_BASE_STRENGTH = 200
export const AUTO_MIN_AFFECT_RADIUS = 20
export const AUTO_MAX_AFFECT_RADIUS = 300
export const AUTO_MIN_CROWD_LIMIT = 8

// ══════════════════════════════════════════════════════════════════════
//  Main loop
// ══════════════════════════════════════════════════════════════════════

export const FIXED_DT = 1 / 60
export const MAX_STEPS_PER_FRAME = 3

// ══════════════════════════════════════════════════════════════════════
//  Detection
// ══════════════════════════════════════════════════════════════════════

export const DEFAULT_DETECTION_CONFIG: DetectionConfig = {
	proximityRadius: 18,
	coherenceThreshold: 45,
	minOrganelleSize: 3,
	organelleLatchSeconds: 0.5,
	organismProximityRadius: 60,
	organismCoherenceThreshold: 80,
}
/** EMA weight for stability tracking. */
export const STABILITY_ALPHA = 0.02
/** Threshold below which a stability value is treated as zero. */
export const STABILITY_EPSILON = 0.001

// ── Organism prediction ──────────────────────────────────────────────
export const MAX_ORGANISMS = 50
export const MIN_PARTICLES_FOR_VIABILITY = 3
/** Cross-type pairs need meaningful net mutual attraction to bind.
 *  Barely-positive pairs won't overcome particle velocity / damping. */
export const MUTUAL_ATTRACTION_THRESHOLD = 0.15
/** If one direction is strongly repulsive, the pair won't bind. */
export const ONE_SIDED_REPULSION_CUTOFF = -0.35
/** Predictions scoring below this fraction of the top score are pruned. */
export const RELATIVE_SCORE_CUTOFF = 0.2
/** Maximum sub-clique size to enumerate from each maximal clique. */
export const MAX_SUB_CLIQUE_SIZE = 8

// ══════════════════════════════════════════════════════════════════════
//  UI interaction
// ══════════════════════════════════════════════════════════════════════

/** Pixels of pointer movement before a drag starts (avoids hijacking clicks). */
export const DRAG_THRESHOLD = 3
/** Pixels of proximity to a viewport edge that triggers snap. */
export const SNAP_THRESHOLD = 8

// ── Curve editor ─────────────────────────────────────────────────────
export const CURVE_EDITOR_NODE_RADIUS = 5
export const CURVE_EDITOR_HANDLE_RADIUS = 4
export const CURVE_EDITOR_HIT_RADIUS = 10
export const CURVE_EDITOR_MARGIN = 14
export const CURVE_EDITOR_LUT_SIZE = 256

// ══════════════════════════════════════════════════════════════════════
//  Music — defaults exposed via the settings panel
// ══════════════════════════════════════════════════════════════════════

export const MUSIC_DEFAULT_SETTINGS: MusicSettings = {
	volume: 0.5,
	tonicMidi: 45,
	sustainMinSec: 0.6,
	sustainMaxSec: 4,
	timbreMinBlend: 0.15,
	timbreMaxBlend: 0.6,
	minNoteGapSec: 0.5,
	chordCycleSec: 2,
	/** Full progression loops that pass before the engine emits a
	 *  randomize-matrix request. 0 = never emit. */
	loopsToRandomize: 2,
}

// ── Setting clamp ranges ──────────────────────────────────────────────
export const VOLUME_RANGE = { min: 0, max: 1 } as const
export const BLEND_RANGE = { min: 0, max: 1 } as const
export const MIN_NOTE_GAP_RANGE = { min: 0.05, max: 5 } as const
export const CHORD_CYCLE_RANGE = { min: 1, max: 60 } as const
export const LOOPS_TO_RANDOMIZE_RANGE = { min: 0, max: 32 } as const
/** Absolute floor for sustainMinSec — below this the envelope can't
 *  meaningfully decay without clicking. */
export const SUSTAIN_MIN_FLOOR_SEC = 0.02

// ── MIDI ──────────────────────────────────────────────────────────────
export const MIDI_RANGE = { min: 36, max: 96 } as const
export const A4_MIDI = 69
export const A4_FREQ_HZ = 440

// ── Harmony ───────────────────────────────────────────────────────────
/**
 * The progression is generated one-chord-per-type from the force matrix.
 * The move from chord[i] to chord[i+1] is chosen by how "friendly" type i
 * is with type i+1 (mean of the two matrix cells, in [-1, +1]):
 *
 *   very friendly  → perfect 4th up  (subdominant motion, IV-ish)
 *   friendly       → perfect 5th up  (dominant motion, V-ish)
 *   neutral        → whole step up   (step motion, ii-ish)
 *   mildly hostile → minor 6th up    (bVI-borrowed feel)
 *   very hostile   → tritone         (bV, dramatic)
 *
 * Thresholds slice friendliness in [-1, +1]; offsets are semitones from
 * the previous chord's root, mod 12. Chord quality (maj/min/dim) is picked
 * so the triad's third + fifth land inside the current mode when possible.
 */
export const CHORD_MOVE_THRESHOLDS = {
	veryFriendly: 0.5,
	friendly: 0.1,
	hostile: -0.1,
	veryHostile: -0.5,
} as const
export const CHORD_MOVE_SEMITONES = {
	veryFriendly: 5,
	friendly: 7,
	neutral: 2,
	hostile: 8,
	veryHostile: 6,
} as const

// ── Organelle → note signal mappings ──────────────────────────────────
/** Speed (units/sec) at which sustain reaches its minimum. */
export const SPEED_REFERENCE = 250
/** Looseness value at which timbre reaches its maximum blend. */
export const LOOSENESS_REFERENCE = 6
/** Smallest expected organelle size, in particles. */
export const SIZE_REFERENCE = 3
/** Maximum number of octaves a large organelle can drop the pitch by. */
export const MAX_SIZE_OCTAVE_DROP = 1
/** Doubling factor over SIZE_REFERENCE that triggers the octave drop. */
export const SIZE_OCTAVE_DROP_THRESHOLD = 4

// ── Audio graph ───────────────────────────────────────────────────────
/** Peak amplitude a single triggered voice targets. Kept low so many
 *  voices summing in-phase stays comfortably below unity. */
export const VOICE_PEAK = 0.28
export const BASE_MASTER_GAIN = 1.0
/** setTargetAtTime time-constant for the master-volume ramp. */
export const VOLUME_RAMP_TAU_SEC = 0.02

// ── Worklet ───────────────────────────────────────────────────────────
/** Number of voice slots inside the worklet — one per particle type.
 *  Must be ≥ `MAX_TYPES` above. */
export const WORKLET_SLOT_COUNT = 32
/** Identifier registered inside the worklet. */
export const WORKLET_NODE_NAME = "poly-voice"
/** How fast rendered params catch up to their targets, in seconds. */
export const WORKLET_FREQ_GLIDE_TAU_SEC = 0.012
export const WORKLET_AMP_GLIDE_TAU_SEC = 0.03
export const WORKLET_MIX_GLIDE_TAU_SEC = 0.05
/** Envelope decay floor — targetAmp decays toward this rather than 0
 *  so it never becomes truly silent unless the slot is untriggered. */
export const WORKLET_AMP_FLOOR = 1e-5
/** Initial per-slot resting frequency, before any trigger arrives. */
export const WORKLET_DEFAULT_SLOT_FREQ_HZ = 220
/** Initial per-processor internal gain (multiplied by the master gain node
 *  the worklet is routed through — kept < 1 for extra headroom). */
export const WORKLET_INTERNAL_GAIN = 0.9

// ── Key transition ────────────────────────────────────────────────────
/** Amp multiplier applied to every voice on a key change — a musical
 *  "breath" so the new key emerges through fresh triggers. */
export const KEY_TRANSITION_FADE = 0.25

// ── Key derivation ────────────────────────────────────────────────────
export const MODES: readonly { name: string; intervals: readonly number[] }[] = [
	{ name: "Lydian", intervals: [0, 2, 4, 6, 7, 9, 11] },
	{ name: "Ionian", intervals: [0, 2, 4, 5, 7, 9, 11] },
	{ name: "Mixolydian", intervals: [0, 2, 4, 5, 7, 9, 10] },
	{ name: "Dorian", intervals: [0, 2, 3, 5, 7, 9, 10] },
	{ name: "Aeolian", intervals: [0, 2, 3, 5, 7, 8, 10] },
	{ name: "Phrygian", intervals: [0, 1, 3, 5, 7, 8, 10] },
	{ name: "Locrian", intervals: [0, 1, 3, 5, 6, 8, 10] },
]
/** Fallback mode name used when the force matrix is empty. */
export const DEFAULT_MODE_NAME = "Aeolian"
