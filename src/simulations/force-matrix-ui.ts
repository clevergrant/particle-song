import type { RandomDots } from "./basic-particles"
import {
	emptyMatrix,
	matrixToJSON,
	randomizeMatrix,
} from "./basic-particles"
import { applyStepDelta } from "../ui/number-scroll"

export function buildMatrixUI(
	dots: RandomDots,
	container: HTMLElement,
	rootContainer?: HTMLElement,
) {
	const types = dots.getTypeIds()
	if (types.length === 0) return

	const wrapper = document.createElement("div")
	wrapper.className = "force-matrix-container"

	const headerRow = document.createElement("div")
	headerRow.className = "force-matrix-header"

	const matrixLabel = document.createElement("label")
	matrixLabel.style.fontSize = "12px"
	matrixLabel.style.color = "#999"
	matrixLabel.textContent = "Force Matrix"
	headerRow.appendChild(matrixLabel)

	const randomBtn = document.createElement("button")
	randomBtn.className = "force-matrix-randomize"
	randomBtn.textContent = "Randomize"
	randomBtn.title = "Randomize force matrix and organelle counts"
	randomBtn.addEventListener("click", () => {
		dots.forceMatrix = randomizeMatrix(types)
		dots.forceMatrixDirty = true
		dots.speciesPresence.clear()
		dots.speciesBrightness.clear()
		dots.randomizeCounts()
		syncMatrixUI(dots, wrapper, types)
		syncMatrixHidden(dots, container)

		const queryRoot = rootContainer ?? container
		queryRoot.dispatchEvent(new Event("change", { bubbles: true }))
	})
	headerRow.appendChild(randomBtn)

	const clearBtn = document.createElement("button")
	clearBtn.className = "force-matrix-clear"
	clearBtn.textContent = "Clear"
	clearBtn.title = "Clear force matrix"
	clearBtn.addEventListener("click", () => {
		dots.forceMatrix = emptyMatrix(types)
		dots.forceMatrixDirty = true
		syncMatrixUI(dots, wrapper, types)
		syncMatrixHidden(dots, container)
		container.dispatchEvent(new Event("change", { bubbles: true }))
	})
	headerRow.appendChild(clearBtn)

	wrapper.appendChild(headerRow)

	const grid = document.createElement("div")
	grid.className = "force-matrix-grid"
	grid.style.gridTemplateColumns = `repeat(${types.length + 1}, 16px)`
	grid.style.gridAutoRows = "16px"

	// Top-left empty corner
	grid.appendChild(document.createElement("div"))

	// Column headers
	for (const tgt of types) {
		const hdr = document.createElement("div")
		hdr.className = "force-matrix-header-cell"
		const [r, g, b] = dots.getTypeColor(tgt)
		hdr.innerHTML = `<span class="force-matrix-swatch" style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})"></span>`
		hdr.title = dots.groupNames.get(tgt) || tgt
		grid.appendChild(hdr)
	}

	// Rows
	for (const src of types) {
		const rowHdr = document.createElement("div")
		rowHdr.className = "force-matrix-header-cell"
		const [r, g, b] = dots.getTypeColor(src)
		rowHdr.innerHTML = `<span class="force-matrix-swatch" style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})"></span>`
		rowHdr.title = dots.groupNames.get(src) || src
		grid.appendChild(rowHdr)

		for (const tgt of types) {
			const input = document.createElement("input")
			input.type = "number"
			input.className = "force-matrix-cell"
			input.dataset.src = src
			input.dataset.tgt = tgt
			input.min = "-1"
			input.max = "1"
			input.step = "0.05"
			input.value = String(dots.forceMatrix[src]?.[tgt] ?? 0)
			colorizeCell(input)

			input.addEventListener("wheel", (e) => {
				e.preventDefault()
				e.stopPropagation()
				const direction = e.deltaY > 0 ? -1 : e.deltaY < 0 ? 1 : 0
				if (direction === 0) return
				applyStepDelta(input, direction)
			})

			input.addEventListener("input", () => {
				const val = Math.max(-1, Math.min(1, Number(input.value) || 0))
				const updated: Record<string, Record<string, number>> = {}
				for (const s of types) {
					updated[s] = { ...dots.forceMatrix[s] }
				}
				updated[src][tgt] = val
				dots.forceMatrix = updated
				dots.forceMatrixDirty = true
				colorizeCell(input)
				syncMatrixHidden(dots, container)
			})

			grid.appendChild(input)
		}
	}

	wrapper.appendChild(grid)

	dots._matrixWrapper = wrapper
	dots._matrixContainer = container
	dots._matrixRootContainer = rootContainer ?? container

	container.appendChild(wrapper)
}

export function colorizeCell(input: HTMLInputElement) {
	const val = Math.max(-1, Math.min(1, Number(input.value) || 0))
	const bg = [0x1a, 0x1a, 0x1a] // neutral base (#1a1a1a)
	if (val < 0) {
		const t = Math.abs(val)
		bg[0] = Math.round(bg[0] + (255 - bg[0]) * t)
		bg[1] = Math.round(bg[1] + (0 - bg[1]) * t)
		bg[2] = Math.round(bg[2] + (0 - bg[2]) * t)
	} else if (val > 0) {
		const t = val
		bg[0] = Math.round(bg[0] + (0 - bg[0]) * t)
		bg[1] = Math.round(bg[1] + (255 - bg[1]) * t)
		bg[2] = Math.round(bg[2] + (0 - bg[2]) * t)
	}
	input.style.background = `rgb(${bg[0]}, ${bg[1]}, ${bg[2]})`
	input.style.color = "transparent"
	input.style.caretColor = "transparent"
}

export function syncMatrixUI(
	dots: RandomDots,
	wrapper: HTMLElement,
	types: readonly string[],
) {
	for (const src of types) {
		for (const tgt of types) {
			const input = wrapper.querySelector<HTMLInputElement>(
				`input[data-src="${src}"][data-tgt="${tgt}"]`,
			)
			if (input) {
				input.value = String(dots.forceMatrix[src]?.[tgt] ?? 0)
				colorizeCell(input)
			}
		}
	}
}

export function syncMatrixHeaders(dots: RandomDots, container: HTMLElement) {
	const matrixWrapper = container.querySelector(".force-matrix-container")
	if (!matrixWrapper) return
	const headers = matrixWrapper.querySelectorAll<HTMLElement>(
		".force-matrix-header-cell",
	)
	const types = dots.getTypeIds()
	for (let i = 0; i < types.length; i++) {
		const type = types[i]
		const [r, g, b] = dots.getTypeColor(type)
		const colorStr = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`
		const name = dots.groupNames.get(type) || type
		const colHdr = headers[i]
		if (colHdr) {
			const sw = colHdr.querySelector<HTMLElement>(".force-matrix-swatch")
			if (sw) sw.style.background = colorStr
			colHdr.title = name
		}
		const rowHdr = headers[types.length + i]
		if (rowHdr) {
			const sw = rowHdr.querySelector<HTMLElement>(".force-matrix-swatch")
			if (sw) sw.style.background = colorStr
			rowHdr.title = name
		}
	}
}

export function syncMatrixHidden(dots: RandomDots, container: HTMLElement) {
	const hidden = container.querySelector<HTMLInputElement>(
		'input[data-setting="forceMatrix"]',
	)
	if (hidden) {
		hidden.value = matrixToJSON(dots.forceMatrix)
	}
}
