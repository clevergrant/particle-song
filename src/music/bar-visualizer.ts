/**
 * Bar Visualizer — bottom-of-screen scrubber showing current bar progress
 * and colored organelle note dots stacked vertically at their grid positions.
 *
 * Pure DOM rendering; updated each frame from the simulation loop.
 */

import type { TupletGrid } from "./types"
import { MAX_SUBDIVISION } from "./types"
import { attachNumberDrag } from "../number-drag"

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface BarVisualizerState {
  readonly grid: TupletGrid | null
  readonly barNumber: number
  readonly barStartTime: number
  readonly barDuration: number
  readonly bpm: number
  readonly now: number // AudioContext currentTime
  readonly groupColors: ReadonlyMap<string, readonly [number, number, number]>
  readonly typeKeys: readonly string[]
  /** Info text fields (from bar-meta). */
  readonly rootMidi: number | null
  readonly modeName: string | null
  readonly bufferChordName: string | null
  readonly isBufferBar: boolean
}

/* ------------------------------------------------------------------ */
/*  DOM bootstrap                                                      */
/* ------------------------------------------------------------------ */

let container: HTMLDivElement | null = null
let track: HTMLDivElement | null = null
let scrubber: HTMLDivElement | null = null
let dotsContainer: HTMLDivElement | null = null
let infoLeft: HTMLSpanElement | null = null
let bpmInput: HTMLInputElement | null = null
let playBtn: HTMLButtonElement | null = null
let isPlaying = false
let cyclesInput: HTMLInputElement | null = null
let barsInput: HTMLInputElement | null = null
let niceModeCheckbox: HTMLInputElement | null = null

// Cache to avoid re-creating dots every frame
let cachedBarNumber = -1
let cachedGridNoteCount = -1

function ensureDOM(): void {
  if (container) return

  container = document.createElement("div")
  container.id = "bar-visualizer"

  // Info row
  const infoRow = document.createElement("div")
  infoRow.className = "bv-info"

  playBtn = document.createElement("button")
  playBtn.className = "bv-play-btn"
  playBtn.innerHTML = PLAY_SVG
  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying
    playBtn!.innerHTML = isPlaying ? PAUSE_SVG : PLAY_SVG
    playOnToggle?.(isPlaying)
  })
  infoRow.appendChild(playBtn)

  infoLeft = document.createElement("span")
  infoLeft.className = "bv-info-left"
  infoRow.appendChild(infoLeft)

  const controlsGroup = document.createElement("div")
  controlsGroup.className = "bv-controls-group"
  attachNumberDrag(controlsGroup)

  const niceModeWrapper = document.createElement("div")
  niceModeWrapper.className = "bv-nice-toggle"
  const niceOff = document.createElement("span")
  niceOff.textContent = "\u{1F610}"
  niceOff.className = "bv-nice-label"
  niceModeCheckbox = document.createElement("input")
  niceModeCheckbox.type = "checkbox"
  niceModeCheckbox.addEventListener("change", () => {
    niceModeOnChange?.(niceModeCheckbox!.checked)
  })
  const niceOn = document.createElement("span")
  niceOn.textContent = "\u{1F642}"
  niceOn.className = "bv-nice-label"
  niceModeWrapper.appendChild(niceOff)
  niceModeWrapper.appendChild(niceModeCheckbox)
  niceModeWrapper.appendChild(niceOn)
  controlsGroup.appendChild(niceModeWrapper)

  bpmInput = makeCompactInput("bpm", 90, 20, 300, 5, (v) => bpmOnChange?.(v))
  controlsGroup.appendChild(bpmInput.parentElement!)

  barsInput = makeCompactInput("bars", 2, 1, 16, 1, (v) => barsPerMelodyOnChange?.(v))
  barsInput.parentElement!.title =
    "Bars per melody — how many bars each computed melody plays before the next is calculated"
  controlsGroup.appendChild(barsInput.parentElement!)

  cyclesInput = makeCompactInput("cycles", 3, 1, 20, 1, (v) => cyclesOnChange?.(v))
  controlsGroup.appendChild(cyclesInput.parentElement!)

  infoRow.appendChild(controlsGroup)

  container.appendChild(infoRow)

  // Track (the scrub bar area)
  track = document.createElement("div")
  track.className = "bv-track"

  dotsContainer = document.createElement("div")
  dotsContainer.className = "bv-dots"
  track.appendChild(dotsContainer)

  scrubber = document.createElement("div")
  scrubber.className = "bv-scrubber"
  track.appendChild(scrubber)

  container.appendChild(track)

  container.style.display = "none"
  document.body.appendChild(container)
}

/* ------------------------------------------------------------------ */
/*  Note dots (inside track, vertically stacked)                       */
/* ------------------------------------------------------------------ */

/**
 * Asymptotic height offset for stacked dots.
 * f(i) = maxHeight * (1 - 1/(1 + k*i))
 * Dots near the bottom are evenly spaced; dots near the top crowd together.
 */
function stackOffset(index: number, maxHeight: number): number {
  const k = 0.6
  return maxHeight * (1 - 1 / (1 + k * index))
}

/** Render dots from the tuplet grid — each filled slot becomes a colored
 *  dot inside the track, stacked vertically when multiple notes share a position. */
function renderGridDots(
  grid: TupletGrid,
  barNumber: number,
  groupColors: ReadonlyMap<string, readonly [number, number, number]>,
  typeKeys: readonly string[],
): void {
  if (!dotsContainer) return

  // Count total notes in the grid to detect changes
  let noteCount = 0
  for (let t = 0; t < MAX_SUBDIVISION; t++) {
    const tier = grid.tiers[t]
    for (let s = 0; s <= t; s++) {
      const slot = tier[s]
      if (slot) noteCount += slot.length
    }
  }

  // Skip re-render if bar and note count haven't changed
  if (barNumber === cachedBarNumber && noteCount === cachedGridNoteCount) return
  cachedBarNumber = barNumber
  cachedGridNoteCount = noteCount

  dotsContainer.innerHTML = ""

  // Group notes by their horizontal fraction so we can stack them
  const columns = new Map<number, { typeKey: string; rgb: readonly [number, number, number] }[]>()

  for (let t = 0; t < MAX_SUBDIVISION; t++) {
    const tier = grid.tiers[t]
    const tierSize = t + 1
    for (let s = 0; s < tierSize; s++) {
      const slot = tier[s]
      if (!slot || slot.length === 0) continue

      // Quantize fraction to avoid floating-point column splits
      const frac = Math.round((s / tierSize) * 10000) / 10000

      let col = columns.get(frac)
      if (!col) { col = []; columns.set(frac, col) }

      for (const note of slot) {
        const typeKey = typeKeys[note.typeId] ?? ""
        const rgb = groupColors.get(typeKey) ?? [1, 1, 1]
        col.push({ typeKey, rgb })
      }
    }
  }

  // Track height for stacking (the track is 24px by default)
  const trackHeight = 24
  const dotSize = 6

  for (const [frac, notes] of columns) {
    // Render top-most dots first so they sit behind lower ones in DOM order
    for (let i = notes.length - 1; i >= 0; i--) {
      const { rgb } = notes[i]
      const dot = document.createElement("div")
      dot.className = "bv-dot"
      dot.style.left = `${frac * 100}%`
      // Stack from bottom up with asymptotic compression
      const bottomOffset = stackOffset(i, trackHeight - dotSize)
      dot.style.bottom = `${bottomOffset}px`
      dot.style.backgroundColor = `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`
      dotsContainer.appendChild(dot)
    }
  }
}

let playOnToggle: ((playing: boolean) => void) | null = null
let bpmOnChange: ((bpm: number) => void) | null = null
let cyclesOnChange: ((cycles: number) => void) | null = null
let barsPerMelodyOnChange: ((bars: number) => void) | null = null
let niceModeOnChange: ((nice: boolean) => void) | null = null

const PLAY_SVG = `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><polygon points="6,3 20,12 6,21"/></svg>`
const PAUSE_SVG = `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="5" y="3" width="4" height="18"/><rect x="15" y="3" width="4" height="18"/></svg>`

/** Register a callback for when the user toggles play/pause. */
export function onPlayToggle(cb: (playing: boolean) => void): void {
  playOnToggle = cb
}

/** Sync the play button state from external changes (e.g. the settings checkbox). */
export function setPlayState(playing: boolean): void {
  ensureDOM()
  isPlaying = playing
  if (playBtn) playBtn.innerHTML = playing ? PAUSE_SVG : PLAY_SVG
}

/** Register a callback for when the user changes BPM. */
export function onBpmChange(cb: (bpm: number) => void): void {
  bpmOnChange = cb
}

/** Sync the BPM input from external changes. */
export function setBpm(bpm: number): void {
  ensureDOM()
  if (bpmInput) bpmInput.value = String(bpm)
}

/** Register a callback for when the user changes cycles before randomize. */
export function onCyclesChange(cb: (cycles: number) => void): void {
  cyclesOnChange = cb
}

/** Sync the cycles input from external changes. */
export function setCycles(cycles: number): void {
  ensureDOM()
  if (cyclesInput) cyclesInput.value = String(cycles)
}

/** Register a callback for when the user changes bars per melody. */
export function onBarsPerMelodyChange(cb: (bars: number) => void): void {
  barsPerMelodyOnChange = cb
}

/** Sync the bars-per-melody input from external changes. */
export function setBarsPerMelody(bars: number): void {
  ensureDOM()
  if (barsInput) barsInput.value = String(bars)
}

/** Register a callback for when the user toggles nice modes. */
export function onNiceModeChange(cb: (nice: boolean) => void): void {
  niceModeOnChange = cb
}

/** Sync the nice-mode toggle from external changes. */
export function setNiceMode(nice: boolean): void {
  ensureDOM()
  if (niceModeCheckbox) niceModeCheckbox.checked = nice
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"] as const

function midiToNoteName(midi: number): string {
  return NOTE_NAMES[((midi % 12) + 12) % 12]
}

function makeCompactInput(
  label: string, initial: number, min: number, max: number, step: number,
  onChange: (v: number) => void,
): HTMLInputElement {
  const wrapper = document.createElement("div")
  wrapper.className = "bv-compact-input"

  const lbl = document.createElement("span")
  lbl.className = "bv-compact-label"
  lbl.textContent = label
  wrapper.appendChild(lbl)

  const input = document.createElement("input")
  input.type = "number"
  input.min = String(min)
  input.max = String(max)
  input.step = String(step)
  input.value = String(initial)
  const handle = () => {
    let v = Number(input.value)
    v = Math.round(v / step) * step
    v = Math.max(min, Math.min(max, v))
    input.value = String(v)
    onChange(v)
  }
  input.addEventListener("change", handle)
  input.addEventListener("input", handle)
  wrapper.appendChild(input)

  return input
}

/* ------------------------------------------------------------------ */
/*  Public update (called each frame)                                  */
/* ------------------------------------------------------------------ */

export function updateBarVisualizer(state: BarVisualizerState): void {
  ensureDOM()

  const { grid, barNumber, barStartTime, barDuration: barDur, now, groupColors, typeKeys } = state

  // Progress fraction [0, 1]
  const progress = barDur > 0 ? Math.max(0, Math.min(1, (now - barStartTime) / barDur)) : 0

  // Scrubber position
  if (scrubber) {
    scrubber.style.left = `${progress * 100}%`
  }

  // Buffer bar indicator
  if (track) {
    track.classList.toggle("bv-buffer-bar", state.isBufferBar)
  }

  // Hit dots from tuplet grid
  if (grid) {
    renderGridDots(grid, barNumber, groupColors, typeKeys)
  }

  // Info text (from bar-meta)
  if (infoLeft) {
    if (state.rootMidi != null && state.modeName) {
      const rootName = midiToNoteName(state.rootMidi)
      const bufferTag = state.bufferChordName ? `  ·  ${state.bufferChordName}` : ""
      infoLeft.textContent = `${rootName}  ·  ${state.modeName}${bufferTag}`
    }
  }
  // Keep BPM input in sync
  if (bpmInput && Number(bpmInput.value) !== state.bpm) {
    bpmInput.value = String(state.bpm)
  }
}

/** Hide the visualizer (e.g. when audio is off). */
export function hideBarVisualizer(): void {
  if (container) container.style.display = "none"
}

/** Show the visualizer. */
export function showBarVisualizer(): void {
  if (container) container.style.display = ""
}
