import type { RandomDots } from "../basic-particles"
import { createNumberGroup, createToggleGroup } from "../../ui/ui-helpers"
import { makeSection } from "./section"

/** Chord-root label by semitone offset from tonic (0..11). */
const CHORD_ROOT_NAMES = [
	"I", "bII", "II", "bIII", "III", "IV",
	"bV", "V", "bVI", "VI", "bVII", "VII",
]

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const
const midiToNoteName = (midi: number) => {
	const note = NOTE_NAMES[((midi % 12) + 12) % 12]
	const octave = Math.floor(midi / 12) - 1
	return `${note}${octave}`
}

function makeStatusCell(label: string) {
	const cell = document.createElement("div")
	cell.className = "status-cell"
	const lbl = document.createElement("div")
	lbl.className = "status-cell-label"
	lbl.textContent = label
	const val = document.createElement("div")
	val.className = "status-cell-value"
	val.textContent = "—"
	cell.appendChild(lbl)
	cell.appendChild(val)
	return { cell, val }
}

export function buildMusicWindow(dots: RandomDots, container: HTMLElement) {
	// --- Status Bar ---
	const statusBar = document.createElement("div")

	const soundToggle = createToggleGroup({
		label: "Enable Sound",
		checked: false,
		setting: "soundEnabled",
		onChange: (v) => {
			if (v) dots.music.enable()
			else dots.music.disable()
		},
	})
	statusBar.appendChild(soundToggle.group)

	const toggleSeparator = document.createElement("div")
	toggleSeparator.style.height = "8px"
	statusBar.appendChild(toggleSeparator)

	const keyCell = makeStatusCell("Key")
	const modeCell = makeStatusCell("Mode")
	const chordCell = makeStatusCell("Chord")
	const statusGrid = document.createElement("div")
	statusGrid.className = "status-grid"
	statusGrid.appendChild(keyCell.cell)
	statusGrid.appendChild(modeCell.cell)
	statusGrid.appendChild(chordCell.cell)
	statusBar.appendChild(statusGrid)

	setInterval(() => {
		const key = dots.music.getKey()
		const settings = dots.music.getSettings()
		const tonicPc = ((settings.tonicMidi + key.tonicTypeIdx) % 12 + 12) % 12
		keyCell.val.textContent = NOTE_NAMES[tonicPc]
		modeCell.val.textContent = key.modeName
		const chord = dots.music.getHarmony().chord
		const rootLabel = CHORD_ROOT_NAMES[chord.rootSemitones] ?? String(chord.rootSemitones)
		chordCell.val.textContent = `${rootLabel} ${chord.quality}`
	}, 250)

	container.appendChild(statusBar)

	// --- Mix ---
	const mix = makeSection("Mix", false)
	mix.body.appendChild(
		createNumberGroup({
			label: "Volume",
			value: Math.round(dots.music.getSettings().volume * 100),
			setting: "soundVolume",
			min: 0,
			max: 100,
			step: 1,
			suffix: "%",
			onInput: (v) => dots.music.setVolume(v / 100),
		}),
	)
	container.appendChild(mix.section)

	// --- Pitch ---
	const pitch = makeSection("Pitch", false)
	pitch.body.appendChild(
		createNumberGroup({
			label: "Tonic MIDI",
			value: dots.music.getSettings().tonicMidi,
			setting: "musicTonicMidi",
			min: 24,
			max: 72,
			step: 1,
			suffix: "midi",
			onInput: (v) => dots.music.setTonicMidi(v),
		}),
	)
	container.appendChild(pitch.section)

	// --- Note density ---
	const density = makeSection("Note density", false)
	density.body.appendChild(
		createNumberGroup({
			label: "Min gap per voice",
			value: dots.music.getSettings().minNoteGapSec,
			setting: "musicMinNoteGapSec",
			min: 0.1,
			max: 5,
			step: 0.05,
			suffix: "s",
			onInput: (v) => dots.music.setMinNoteGap(v),
		}),
	)
	container.appendChild(density.section)

	// --- Chord cycle ---
	const chord = makeSection("Chord cycle", false)
	chord.body.appendChild(
		createNumberGroup({
			label: "Chord duration",
			value: dots.music.getSettings().chordCycleSec,
			setting: "musicChordCycleSec",
			min: 1,
			max: 60,
			step: 0.5,
			suffix: "s",
			onInput: (v) => dots.music.setChordCycleSec(v),
		}),
	)
	container.appendChild(chord.section)

	// --- Key rotation ---
	const rotation = makeSection("Key rotation", false)
	rotation.body.appendChild(
		createNumberGroup({
			label: "Loops per randomize",
			value: dots.music.getSettings().loopsToRandomize,
			setting: "musicLoopsToRandomize",
			min: 0,
			max: 32,
			step: 1,
			suffix: "",
			onInput: (v) => dots.music.setLoopsToRandomize(v),
		}),
	)
	container.appendChild(rotation.section)

	// --- Sustain (from centroid velocity) ---
	const sustain = makeSection("Sustain (centroid velocity)", false)
	const settings = dots.music.getSettings()
	sustain.body.appendChild(
		createNumberGroup({
			label: "Fast movers",
			value: settings.sustainMinSec,
			setting: "musicSustainMinSec",
			min: 0.02,
			max: 2,
			step: 0.01,
			suffix: "s",
			onInput: (v) => {
				dots.music.setSustainRange(v, dots.music.getSettings().sustainMaxSec)
			},
		}),
	)
	sustain.body.appendChild(
		createNumberGroup({
			label: "Slow movers",
			value: settings.sustainMaxSec,
			setting: "musicSustainMaxSec",
			min: 0.1,
			max: 8,
			step: 0.1,
			suffix: "s",
			onInput: (v) => {
				dots.music.setSustainRange(dots.music.getSettings().sustainMinSec, v)
			},
		}),
	)
	container.appendChild(sustain.section)

	// --- Timbre (from organelle looseness = stress proxy) ---
	const timbre = makeSection("Timbre (stress)", false)
	timbre.body.appendChild(
		createNumberGroup({
			label: "Compact blend",
			value: settings.timbreMinBlend,
			setting: "musicTimbreMinBlend",
			min: 0,
			max: 1,
			step: 0.01,
			suffix: "",
			onInput: (v) => {
				dots.music.setTimbreRange(v, dots.music.getSettings().timbreMaxBlend)
			},
		}),
	)
	timbre.body.appendChild(
		createNumberGroup({
			label: "Loose blend",
			value: settings.timbreMaxBlend,
			setting: "musicTimbreMaxBlend",
			min: 0,
			max: 1,
			step: 0.01,
			suffix: "",
			onInput: (v) => {
				dots.music.setTimbreRange(dots.music.getSettings().timbreMinBlend, v)
			},
		}),
	)
	container.appendChild(timbre.section)
}
