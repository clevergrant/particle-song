/**
 * Bar Visualizer — bottom-of-screen scrubber showing current bar progress,
 * beat lines, and colored organelle hit dots.
 *
 * Pure DOM rendering; updated each frame from the simulation loop.
 */

import type { ScheduledBar } from "./types"
import type { BassDensity } from "./bass-layer"

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface BarVisualizerState {
  readonly scheduledBar: ScheduledBar | null
  readonly barStartTime: number
  readonly barDuration: number
  readonly beatsPerBar: number
  readonly bpm: number
  readonly now: number // AudioContext currentTime
  readonly groupColors: ReadonlyMap<string, readonly [number, number, number]>
  readonly typeKeys: readonly string[]
  readonly phraseBarInCycle: number // -1 if inactive
  readonly phraseSequenceLength: number // expanded sequence length
  readonly phraseIndices: readonly number[] // maps sequence position → original cell index
}

/* ------------------------------------------------------------------ */
/*  DOM bootstrap                                                      */
/* ------------------------------------------------------------------ */

let container: HTMLDivElement | null = null
let track: HTMLDivElement | null = null
let scrubber: HTMLDivElement | null = null
let beatLinesContainer: HTMLDivElement | null = null
let dotsContainer: HTMLDivElement | null = null
let infoLeft: HTMLSpanElement | null = null
let beatsInput: HTMLInputElement | null = null
let bpmInput: HTMLInputElement | null = null
let playBtn: HTMLButtonElement | null = null
let isPlaying = false
let phraseStrip: HTMLDivElement | null = null
let phraseCells: HTMLDivElement[] = []
let phraseMirrorCheckbox: HTMLInputElement | null = null
let niceModeCheckbox: HTMLInputElement | null = null

// Cache to avoid re-creating dots every frame
let cachedBarNumber = -1
let cachedPhraseActiveCell = -1

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

  beatsInput = makeCompactInput("beats", 4, 2, 8, 1, (v) => beatsOnChange?.(v))
  controlsGroup.appendChild(beatsInput.parentElement!)

  infoRow.appendChild(controlsGroup)

  container.appendChild(infoRow)

  // Phrase strip row
  const phraseRow = document.createElement("div")
  phraseRow.className = "bv-phrase-row"

  phraseStrip = document.createElement("div")
  phraseStrip.className = "bv-phrase-strip"
  phraseRow.appendChild(phraseStrip)

  const mirrorLabel = document.createElement("label")
  mirrorLabel.className = "bv-phrase-mirror"
  phraseMirrorCheckbox = document.createElement("input")
  phraseMirrorCheckbox.type = "checkbox"
  phraseMirrorCheckbox.addEventListener("change", () => {
    const cells = phraseCells.map(c => c.dataset.density as BassDensity)
    phraseOnChange?.(cells, phraseMirrorCheckbox!.checked)
  })
  const mirrorText = document.createElement("span")
  mirrorText.textContent = "Mirror"
  mirrorLabel.appendChild(mirrorText)
  mirrorLabel.appendChild(phraseMirrorCheckbox)
  phraseRow.appendChild(mirrorLabel)

  container.appendChild(phraseRow)

  // Track (the scrub bar area)
  const trackWrapper = document.createElement("div")
  trackWrapper.className = "bv-track-wrapper"

  dotsContainer = document.createElement("div")
  dotsContainer.className = "bv-dots"
  trackWrapper.appendChild(dotsContainer)

  track = document.createElement("div")
  track.className = "bv-track"

  beatLinesContainer = document.createElement("div")
  beatLinesContainer.className = "bv-beat-lines"
  track.appendChild(beatLinesContainer)

  scrubber = document.createElement("div")
  scrubber.className = "bv-scrubber"
  track.appendChild(scrubber)

  trackWrapper.appendChild(track)
  container.appendChild(trackWrapper)

  container.style.display = "none"
  document.body.appendChild(container)
}

/* ------------------------------------------------------------------ */
/*  Beat lines                                                         */
/* ------------------------------------------------------------------ */

let cachedBeatsPerBar = -1

function renderBeatLines(beatsPerBar: number): void {
  if (!beatLinesContainer || beatsPerBar === cachedBeatsPerBar) return
  cachedBeatsPerBar = beatsPerBar

  beatLinesContainer.innerHTML = ""

  // Show beat lines for quarter-note downbeats only. Tuplet subdivisions
  // don't align to a fixed grid, so we just mark the beats.
  for (let i = 0; i <= beatsPerBar; i++) {
    const line = document.createElement("div")
    const pct = (i / beatsPerBar) * 100
    line.style.left = `${pct}%`
    line.className = i === 0 ? "bv-beat-line bv-downbeat" : "bv-beat-line"
    beatLinesContainer.appendChild(line)
  }
}

/* ------------------------------------------------------------------ */
/*  Hit dots                                                           */
/* ------------------------------------------------------------------ */

function renderDots(
  bar: ScheduledBar,
  groupColors: ReadonlyMap<string, readonly [number, number, number]>,
  typeKeys: readonly string[],
): void {
  if (!dotsContainer) return
  if (bar.barNumber === cachedBarNumber) return
  cachedBarNumber = bar.barNumber

  dotsContainer.innerHTML = ""

  const barStart = bar.startTime
  const barDur = bar.duration
  if (barDur <= 0) return

  // Cap at 16 dots per type to match the max subdivision count
  const typeCounts = new Map<number, number>()

  for (const hit of bar.hits) {
    const count = typeCounts.get(hit.typeId) ?? 0
    if (count >= 16) continue
    typeCounts.set(hit.typeId, count + 1)

    // Place dots at their actual scheduled time — the scheduler already
    // quantized to the voice's own subdivision grid (triplets, quintuplets, etc.)
    const frac = (hit.time - barStart) / barDur
    if (frac < 0 || frac > 1) continue

    const typeKey = typeKeys[hit.typeId] ?? ""
    const rgb = groupColors.get(typeKey) ?? [1, 1, 1]

    const dot = document.createElement("div")
    dot.className = "bv-dot"
    dot.style.left = `${frac * 100}%`
    dot.style.backgroundColor = `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`
    dotsContainer.appendChild(dot)
  }
}

/* ------------------------------------------------------------------ */
/*  Phrase strip                                                       */
/* ------------------------------------------------------------------ */

const DENSITY_CYCLE: readonly BassDensity[] = ["W", "H", "Q", "E", "X"]
const DENSITY_GLYPH: Record<BassDensity, string> = {
  W: "\uD834\uDD5D",  // 𝅝  whole note
  H: "\uD834\uDD5E",  // 𝅗𝅥  half note
  Q: "\u2669",         // ♩  quarter note
  E: "\u266A",         // ♪  eighth note
  X: "\u2715",         // ✕  disabled
}
const DENSITY_BG: Record<BassDensity, string> = {
  W: "rgba(255,255,255,0.08)",
  H: "rgba(255,255,255,0.14)",
  Q: "rgba(255,255,255,0.22)",
  E: "rgba(255,255,255,0.32)",
  X: "rgba(255,255,255,0.03)",
}

let phraseOnChange: ((cells: readonly BassDensity[], mirror: boolean) => void) | null = null
let playOnToggle: ((playing: boolean) => void) | null = null
let beatsOnChange: ((beats: number) => void) | null = null
let bpmOnChange: ((bpm: number) => void) | null = null
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

/** Register a callback for when the user changes beats per bar. */
export function onBeatsChange(cb: (beats: number) => void): void {
  beatsOnChange = cb
}

/** Sync the beats input from external changes. */
export function setBeatsPerBar(beats: number): void {
  ensureDOM()
  if (beatsInput) beatsInput.value = String(beats)
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

/** Register a callback for when the user toggles nice modes. */
export function onNiceModeChange(cb: (nice: boolean) => void): void {
  niceModeOnChange = cb
}

/** Sync the nice-mode toggle from external changes. */
export function setNiceMode(nice: boolean): void {
  ensureDOM()
  if (niceModeCheckbox) niceModeCheckbox.checked = nice
}

/** Register a callback for when the user edits the phrase strip. */
export function onPhraseChange(
  cb: (cells: readonly BassDensity[], mirror: boolean) => void,
): void {
  phraseOnChange = cb
}

/** Rebuild phrase strip cells from the given pattern. */
export function setPhraseStripCells(cells: readonly BassDensity[], mirror: boolean): void {
  ensureDOM()
  if (!phraseStrip) return

  phraseStrip.innerHTML = ""
  phraseCells = []

  for (let i = 0; i < cells.length; i++) {
    const cell = document.createElement("div")
    cell.className = "bv-phrase-cell"
    cell.dataset.setting = `phraseCell${i}`
    cell.dataset.density = cells[i]
    cell.textContent = DENSITY_GLYPH[cells[i]]
    cell.style.background = DENSITY_BG[cells[i]]

    const cycleCell = (dir: 1 | -1) => {
      const cur = cell.dataset.density as BassDensity
      const idx = DENSITY_CYCLE.indexOf(cur)
      const next = DENSITY_CYCLE[((idx + dir) % DENSITY_CYCLE.length + DENSITY_CYCLE.length) % DENSITY_CYCLE.length]
      cell.dataset.density = next
      cell.textContent = DENSITY_GLYPH[next]
      cell.style.background = DENSITY_BG[next]
      const updated = phraseCells.map(c => c.dataset.density as BassDensity)
      phraseOnChange?.(updated, phraseMirrorCheckbox?.checked ?? false)
    }
    cell.addEventListener("click", () => cycleCell(1))
    cell.addEventListener("contextmenu", (e) => { e.preventDefault(); cycleCell(-1) })

    phraseStrip.appendChild(cell)
    phraseCells.push(cell)
  }

  if (phraseMirrorCheckbox) {
    phraseMirrorCheckbox.checked = mirror
  }
}

function updatePhraseHighlight(barInCycle: number, indices: readonly number[]): void {
  if (barInCycle === cachedPhraseActiveCell) return
  cachedPhraseActiveCell = barInCycle

  const cellCount = phraseCells.length
  if (cellCount === 0) return

  // Use the precomputed index mapping (handles X-filtered + mirrored sequences)
  const cellIdx = barInCycle >= 0 && barInCycle < indices.length
    ? indices[barInCycle]
    : -1

  for (let i = 0; i < cellCount; i++) {
    phraseCells[i].classList.toggle("active", i === cellIdx)
  }
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
  input.addEventListener("change", () => {
    let v = Number(input.value)
    v = Math.round(v / step) * step
    v = Math.max(min, Math.min(max, v))
    input.value = String(v)
    onChange(v)
  })
  wrapper.appendChild(input)

  return input
}

/* ------------------------------------------------------------------ */
/*  Public update (called each frame)                                  */
/* ------------------------------------------------------------------ */

export function updateBarVisualizer(state: BarVisualizerState): void {
  ensureDOM()

  const { scheduledBar, barStartTime, barDuration: barDur, beatsPerBar, now, groupColors, typeKeys } = state

  // Progress fraction [0, 1]
  const progress = barDur > 0 ? Math.max(0, Math.min(1, (now - barStartTime) / barDur)) : 0

  // Scrubber position
  if (scrubber) {
    scrubber.style.left = `${progress * 100}%`
  }

  // Beat lines
  renderBeatLines(beatsPerBar)

  // Buffer bar indicator
  if (track) {
    track.classList.toggle("bv-buffer-bar", scheduledBar?.isBufferBar ?? false)
  }

  // Hit dots
  if (scheduledBar) {
    renderDots(scheduledBar, groupColors, typeKeys)
  }

  // Phrase strip highlight
  updatePhraseHighlight(state.phraseBarInCycle, state.phraseIndices)

  // Info text
  if (infoLeft && scheduledBar) {
    const rootName = midiToNoteName(scheduledBar.rootMidi)
    const modeName = scheduledBar.mode.name
    const bufferTag = scheduledBar.bufferChord
      ? `  ·  ${scheduledBar.bufferChord.name}`
      : ""
    infoLeft.textContent = `${rootName}  ·  ${modeName}${bufferTag}`
  }
  // Keep inputs in sync with current values
  if (beatsInput && Number(beatsInput.value) !== beatsPerBar) {
    beatsInput.value = String(beatsPerBar)
  }
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
