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
  readonly now: number // AudioContext currentTime
  readonly groupColors: ReadonlyMap<string, readonly [number, number, number]>
  readonly typeKeys: readonly string[]
  readonly phraseBarInCycle: number // -1 if inactive
  readonly phraseSequenceLength: number // expanded sequence length
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
let infoRight: HTMLSpanElement | null = null
let phraseStrip: HTMLDivElement | null = null
let phraseCells: HTMLDivElement[] = []
let phraseMirrorCheckbox: HTMLInputElement | null = null

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
  infoLeft = document.createElement("span")
  infoLeft.className = "bv-info-left"
  infoRight = document.createElement("span")
  infoRight.className = "bv-info-right"
  infoRow.appendChild(infoLeft)
  infoRow.appendChild(infoRight)
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

const DENSITY_CYCLE: readonly BassDensity[] = ["W", "H", "Q", "E"]
const DENSITY_GLYPH: Record<BassDensity, string> = {
  W: "\uD834\uDD5D",  // 𝅝  whole note
  H: "\uD834\uDD5E",  // 𝅗𝅥  half note
  Q: "\u2669",         // ♩  quarter note
  E: "\u266A",         // ♪  eighth note
}
const DENSITY_BG: Record<BassDensity, string> = {
  W: "rgba(255,255,255,0.08)",
  H: "rgba(255,255,255,0.14)",
  Q: "rgba(255,255,255,0.22)",
  E: "rgba(255,255,255,0.32)",
}

let phraseOnChange: ((cells: readonly BassDensity[], mirror: boolean) => void) | null = null

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

    cell.addEventListener("click", () => {
      const cur = cell.dataset.density as BassDensity
      const next = DENSITY_CYCLE[(DENSITY_CYCLE.indexOf(cur) + 1) % DENSITY_CYCLE.length]
      cell.dataset.density = next
      cell.textContent = DENSITY_GLYPH[next]
      cell.style.background = DENSITY_BG[next]
      // Read back all cells
      const updated = phraseCells.map(c => c.dataset.density as BassDensity)
      phraseOnChange?.(updated, phraseMirrorCheckbox?.checked ?? false)
    })

    phraseStrip.appendChild(cell)
    phraseCells.push(cell)
  }

  if (phraseMirrorCheckbox) {
    phraseMirrorCheckbox.checked = mirror
  }
}

function updatePhraseHighlight(barInCycle: number, seqLen: number): void {
  if (barInCycle === cachedPhraseActiveCell) return
  cachedPhraseActiveCell = barInCycle

  // The barInCycle is within the expanded (possibly mirrored) sequence.
  // Map it back to the cell index: if mirrored, second half maps backwards.
  const cellCount = phraseCells.length
  if (cellCount === 0) return

  let cellIdx: number
  if (seqLen > cellCount && barInCycle >= cellCount) {
    // Mirrored half — map backwards
    cellIdx = seqLen - 1 - barInCycle
  } else {
    cellIdx = barInCycle % cellCount
  }

  for (let i = 0; i < cellCount; i++) {
    phraseCells[i].classList.toggle("active", i === cellIdx)
  }
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
  updatePhraseHighlight(state.phraseBarInCycle, state.phraseSequenceLength)

  // Info text
  if (infoLeft && scheduledBar) {
    const barNum = scheduledBar.barNumber
    const modeName = scheduledBar.mode.name
    const bufferTag = scheduledBar.isBufferBar ? "  ·  transition" : ""
    infoLeft.textContent = `Bar ${barNum + 1}  ·  ${modeName}${bufferTag}`
  }
  if (infoRight) {
    const beatInBar = progress * beatsPerBar
    const currentBeat = Math.floor(beatInBar) + 1
    infoRight.textContent = `Beat ${currentBeat} / ${beatsPerBar}`
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
