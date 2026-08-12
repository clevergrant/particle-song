import type { RandomDots } from "./basic-particles"
import { MAX_PARTICLES } from "../constants"
import { resizeMatrix } from "./basic-particles"
import { CustomParticle } from "../particles"
import { ColorPicker } from "../ui/color-picker"
import { applyStepDelta } from "../ui/number-scroll"
import { buildMatrixUI, syncMatrixHeaders } from "./force-matrix-ui"
import { getOpenSections, restoreOpenSections } from "./windows/section"

export function initLedger(dots: RandomDots) {
	dots.ledgerToggle = document.getElementById("ledger-toggle")
	dots.ledgerPanels = document.getElementById("ledger-panels")
	dots.ledgerBackdrop = document.getElementById("ledger-backdrop")
	dots.ledgerOrganellesEl = document.getElementById("ledger-organelles")
	dots.ledgerOrganismsEl = document.getElementById("ledger-organisms")
	dots.ledgerPredictionsEl = document.getElementById("ledger-predictions")

	dots.ledgerToggle?.classList.remove("hidden")

	dots.ledgerToggle?.addEventListener("click", () => {
		const open = dots.ledgerPanels?.classList.toggle("open")
		dots.ledgerBackdrop?.classList.toggle("open", open)
	})
	dots.ledgerBackdrop?.addEventListener("click", () => closeLedger(dots))
	document.addEventListener("keydown", (e) => {
		if (e.key === "Escape" && dots.ledgerPanels?.classList.contains("open"))
			closeLedger(dots)
	})

	// Restore muted organisms from persisted settings
	const savedMuted = dots.ledgerOrganismsEl?.dataset.mutedOrganisms
	if (savedMuted) {
		for (const sig of savedMuted.split(",").filter(Boolean))
			dots.mutedOrganisms.add(sig)
	}

	// Mute button delegation on organism panel
	dots.ledgerOrganismsEl?.addEventListener("click", (e) => {
		const btn = (e.target as HTMLElement).closest<HTMLElement>(".ledger-mute")
		if (!btn) return
		const sig = btn.dataset.sig
		if (!sig) return
		const wasMuted = dots.mutedOrganisms.has(sig)
		if (wasMuted) dots.mutedOrganisms.delete(sig)
		else dots.mutedOrganisms.add(sig)
		syncMutedAttribute(dots)
		updateLedgerUI(dots)
	})
}

export function syncMutedAttribute(dots: RandomDots) {
	const muted = dots.mutedOrganisms
	if (dots.ledgerOrganismsEl) {
		if (muted.size > 0) {
			dots.ledgerOrganismsEl.dataset.mutedOrganisms = [...muted].join(",")
		} else {
			delete dots.ledgerOrganismsEl.dataset.mutedOrganisms
		}
	}
}

export function closeLedger(dots: RandomDots) {
	dots.ledgerPanels?.classList.remove("open")
	dots.ledgerBackdrop?.classList.remove("open")
}

export function updateLedgerUI(dots: RandomDots) {
	const orgEl = dots.ledgerOrganellesEl
	const osmEl = dots.ledgerOrganismsEl
	if (!orgEl || !osmEl) return

	const frame = dots.detectionState
	if (!frame) {
		orgEl.innerHTML = ""
		osmEl.innerHTML = ""
		dots.organelleRows.clear()
		dots.organismRows.clear()
		dots.organelleHeading = null
		dots.organismHeading = null
		return
	}

	const types = dots.getTypeIds()

	// ── Organelles ──────────────────────────────────────────────────
	const activeTypeIds = new Set<number>()
	if (frame.ledger.organellesByType.size > 0) {
		if (!dots.organelleHeading) {
			dots.organelleHeading = document.createElement("div")
			dots.organelleHeading.className = "ledger-heading"
			dots.organelleHeading.textContent = "Organelles"
			orgEl.prepend(dots.organelleHeading)
		}

		const sorted = [...frame.ledger.organellesByType.entries()].sort((a, b) => {
			const sa = frame.ledger.organelleStability.get(a[0]) ?? 0
			const sb = frame.ledger.organelleStability.get(b[0]) ?? 0
			return sb - sa || b[1] - a[1]
		})

		for (const [typeId, count] of sorted) {
			activeTypeIds.add(typeId)
			let entry = dots.organelleRows.get(typeId)
			if (!entry) {
				const row = document.createElement("div")
				row.className = "ledger-row"
				const dot = document.createElement("span")
				dot.className = "ledger-dot"
				const countEl = document.createElement("span")
				countEl.className = "ledger-count"
				row.append(dot, countEl)
				entry = { row, countEl }
				dots.organelleRows.set(typeId, entry)
			}
			const groupId = types[typeId] ?? `t${typeId}`
			const rgb = dots.groupColors.get(groupId) ?? [1, 1, 1]
			const hex = dots.rgbToHex(rgb)
			const dot = entry.row.firstElementChild as HTMLElement
			if (dot.style.background !== hex) {
				dot.style.background = hex
				dot.style.color = hex
			}
			const countStr = String(count)
			if (entry.countEl.textContent !== countStr)
				entry.countEl.textContent = countStr
			if (!entry.row.parentNode) orgEl.append(entry.row)
		}
	} else if (dots.organelleHeading) {
		dots.organelleHeading.remove()
		dots.organelleHeading = null
	}

	for (const [typeId, entry] of dots.organelleRows) {
		if (!activeTypeIds.has(typeId)) {
			entry.row.remove()
			dots.organelleRows.delete(typeId)
		}
	}

	// ── Organisms ───────────────────────────────────────────────────
	const activeSigs = new Set<string>()
	if (frame.ledger.organismsBySignature.size > 0) {
		if (!dots.organismHeading) {
			dots.organismHeading = document.createElement("div")
			dots.organismHeading.className = "ledger-heading"
			const title = document.createElement("span")
			title.textContent = "Organisms"
			dots.unmuteAllBtn = document.createElement("button")
			dots.unmuteAllBtn.className = "ledger-unmute-all"
			dots.unmuteAllBtn.textContent = "🔊"
			dots.unmuteAllBtn.title = "Unmute all"
			dots.unmuteAllBtn.addEventListener("click", () => {
				dots.mutedOrganisms.clear()
				syncMutedAttribute(dots)
				updateLedgerUI(dots)
			})
			dots.organismHeading.append(title, dots.unmuteAllBtn)
			osmEl.prepend(dots.organismHeading)
		}

		const sorted = [...frame.ledger.organismsBySignature.entries()].sort(
			(a, b) => {
				const sa = frame.ledger.organismStability.get(a[0]) ?? 0
				const sb = frame.ledger.organismStability.get(b[0]) ?? 0
				return sb - sa || b[1] - a[1]
			},
		)

		for (const [sig, count] of sorted) {
			activeSigs.add(sig)
			let entry = dots.organismRows.get(sig)
			if (!entry) {
				const row = document.createElement("div")
				row.className = "ledger-row"
				const typeIds = sig.split("+").map(Number)
				for (const tid of typeIds) {
					const dot = document.createElement("span")
					dot.className = "ledger-dot"
					const gid = types[tid] ?? `t${tid}`
					const rgb = dots.groupColors.get(gid) ?? [1, 1, 1]
					const hex = dots.rgbToHex(rgb)
					dot.style.background = hex
					dot.style.color = hex
					row.append(dot)
				}
				const countEl = document.createElement("span")
				countEl.className = "ledger-count"
				const muteBtn = document.createElement("button")
				muteBtn.className = "ledger-mute"
				muteBtn.dataset.sig = sig
				row.append(countEl, muteBtn)
				entry = { row, countEl, muteBtn }
				dots.organismRows.set(sig, entry)
			}
			const countStr = String(count)
			if (entry.countEl.textContent !== countStr)
				entry.countEl.textContent = countStr
			const muted = dots.mutedOrganisms.has(sig)
			const wantClass = muted ? "ledger-mute muted" : "ledger-mute"
			if (entry.muteBtn.className !== wantClass)
				entry.muteBtn.className = wantClass
			const wantIcon = muted ? "🔇" : "🔉"
			if (entry.muteBtn.textContent !== wantIcon)
				entry.muteBtn.textContent = wantIcon
			const wantTitle = muted ? "Unmute" : "Mute"
			if (entry.muteBtn.title !== wantTitle) entry.muteBtn.title = wantTitle
			if (!entry.row.parentNode) osmEl.append(entry.row)
		}
	} else if (dots.organismHeading) {
		dots.organismHeading.remove()
		dots.organismHeading = null
	}

	for (const [sig, entry] of dots.organismRows) {
		if (!activeSigs.has(sig)) {
			entry.row.remove()
			dots.organismRows.delete(sig)
		}
	}

	if (dots.unmuteAllBtn) {
		const anyMuted = dots.mutedOrganisms.size > 0
		const wantDisplay = anyMuted ? "" : "none"
		if (dots.unmuteAllBtn.style.display !== wantDisplay) {
			dots.unmuteAllBtn.style.display = wantDisplay
		}
	}

	updatePredictionLedger(dots)
}

function updatePredictionLedger(dots: RandomDots) {
	const predEl = dots.ledgerPredictionsEl
	if (!predEl) return

	const prediction = dots.organismPrediction
	if (!prediction || prediction.organisms.length === 0) {
		if (dots.predictionHeading) {
			dots.predictionHeading.remove()
			dots.predictionHeading = null
		}
		if (dots.speciesDecaySlider) {
			dots.speciesDecaySlider.parentElement?.remove()
			dots.speciesDecaySlider = null
		}
		for (const [, entry] of dots.predictionRows) entry.row.remove()
		dots.predictionRows.clear()
		dots.speciesPresence.clear()
		dots.speciesBrightness.clear()
		dots.lastPredictionTime = 0
		return
	}

	const now = performance.now()
	const dt =
		dots.lastPredictionTime > 0
			? Math.min((now - dots.lastPredictionTime) / 1000, 0.2)
			: 0
	dots.lastPredictionTime = now

	const decayRate = 0.15

	if (!dots.predictionHeading) {
		dots.predictionHeading = document.createElement("div")
		dots.predictionHeading.className = "ledger-heading"
		dots.predictionHeading.textContent = "Predicted Species"
		predEl.prepend(dots.predictionHeading)
	}

	if (!dots.speciesDecaySlider) {
		const sliderRow = document.createElement("div")
		sliderRow.className = "ledger-row"
		sliderRow.style.gap = "6px"
		sliderRow.style.fontSize = "10px"
		sliderRow.style.color = "#999"
		const label = document.createElement("span")
		label.textContent = "decay"
		const slider = document.createElement("input")
		slider.type = "range"
		slider.min = "0"
		slider.max = "0.5"
		slider.step = "0.01"
		slider.value = String(dots.speciesDecayThreshold)
		slider.style.flex = "1"
		slider.style.height = "12px"
		slider.style.accentColor = "#888"
		slider.addEventListener("input", () => {
			dots.speciesDecayThreshold = parseFloat(slider.value)
		})
		sliderRow.append(label, slider)
		dots.speciesDecaySlider = slider
		predEl.prepend(sliderRow)
		predEl.prepend(dots.predictionHeading)
	}

	const observedSigs =
		dots.detectionState?.ledger.organismsBySignature ?? new Map()

	const allTracked = new Set<string>()
	for (const org of prediction.organisms) allTracked.add(org.signature)
	for (const [sig] of dots.speciesPresence) allTracked.add(sig)
	for (const [sig] of observedSigs) allTracked.add(sig)

	for (const sig of allTracked) {
		const prev = dots.speciesPresence.get(sig) ?? 0
		const prevBright = dots.speciesBrightness.get(sig) ?? 0
		const isObserved = observedSigs.has(sig)
		const next = isObserved ? 1 : prev - dt * decayRate
		const nextBright = isObserved
			? Math.min(1, prevBright + 0.15)
			: prevBright - dt * decayRate * 3
		if (next <= 0) {
			dots.speciesPresence.delete(sig)
			dots.speciesBrightness.delete(sig)
		} else {
			dots.speciesPresence.set(sig, next)
			dots.speciesBrightness.set(sig, Math.max(0, nextBright))
		}
	}

	const predictedBysig = new Map(
		prediction.organisms.map((o) => [o.signature, o] as const),
	)
	const types = dots.getTypeIds()

	const allEntries: {
		sig: string
		presence: number
		predicted: boolean
		typeKeys: ReadonlyArray<string>
	}[] = []

	for (const [sig, presence] of dots.speciesPresence) {
		if (presence < dots.speciesDecayThreshold && !observedSigs.has(sig)) continue
		const pred = predictedBysig.get(sig)
		const typeKeys = pred
			? pred.typeKeys
			: sig
					.split("+")
					.map(Number)
					.map((i) => types[i])
					.filter(Boolean)
		allEntries.push({ sig, presence, predicted: !!pred, typeKeys })
	}
	for (const org of prediction.organisms) {
		if (dots.speciesPresence.has(org.signature)) continue
		allEntries.push({
			sig: org.signature,
			presence: 0,
			predicted: true,
			typeKeys: org.typeKeys,
		})
	}

	allEntries.sort((a, b) => b.presence - a.presence)

	const scores = prediction.organisms.map((o) => o.stabilityScore)
	const minScore = Math.min(...scores)
	const maxScore = Math.max(...scores)
	const scoreRange = maxScore - minScore

	const activeSigs = new Set<string>()
	for (const { sig, predicted, typeKeys } of allEntries) {
		activeSigs.add(sig)
		const entry = ensurePredictionRow(dots, sig, typeKeys)

		if (predicted) {
			const org = predictedBysig.get(sig)!
			const t =
				scoreRange > 0 ? (org.stabilityScore - minScore) / scoreRange : 1
			const [arrow, color] =
				t >= 0.75
					? ["⇈", "#4caf50"]
					: t >= 0.4
						? ["↑", "#81c784"]
						: t >= 0.15
							? ["↓", "#e57373"]
							: ["⇊", "#f44336"]
			if (entry.scoreEl.textContent !== arrow) entry.scoreEl.textContent = arrow
			if (entry.scoreEl.style.color !== color) entry.scoreEl.style.color = color
		} else {
			if (entry.scoreEl.textContent !== "?") entry.scoreEl.textContent = "?"
			if (entry.scoreEl.style.color !== "#ffb74d")
				entry.scoreEl.style.color = "#ffb74d"
		}

		const brightness = dots.speciesBrightness.get(sig) ?? 0
		const wantOpacity = String(Math.max(0.2, brightness))
		if (entry.row.style.opacity !== wantOpacity)
			entry.row.style.opacity = wantOpacity
		predEl.append(entry.row)
	}

	for (const [sig, entry] of dots.predictionRows) {
		if (!activeSigs.has(sig)) {
			entry.row.remove()
			dots.predictionRows.delete(sig)
		}
	}
}

function ensurePredictionRow(
	dots: RandomDots,
	sig: string,
	typeKeys: ReadonlyArray<string>,
): { row: HTMLElement; scoreEl: HTMLElement } {
	let entry = dots.predictionRows.get(sig)
	if (!entry) {
		const row = document.createElement("div")
		row.className = "ledger-row"
		for (const tk of typeKeys) {
			const dot = document.createElement("span")
			dot.className = "ledger-dot"
			const rgb = dots.groupColors.get(tk) ?? [1, 1, 1]
			const hex = dots.rgbToHex(rgb)
			dot.style.background = hex
			dot.style.color = hex
			row.append(dot)
		}
		const scoreEl = document.createElement("span")
		scoreEl.className = "ledger-likelihood"
		row.append(scoreEl)
		entry = { row, scoreEl }
		dots.predictionRows.set(sig, entry)
	}
	return entry
}

export function buildTypeRow(
	dots: RandomDots,
	section: HTMLElement,
	container: HTMLElement,
	type: string,
	members: CustomParticle[],
	rebuildFn: (container: HTMLElement) => void,
) {
	const representative = members[0] ?? null
	const color: [number, number, number] = representative
		? [
				representative.color[0],
				representative.color[1],
				representative.color[2],
			]
		: (dots.groupColors.get(type) ?? [1, 1, 1])

	const row = document.createElement("div")
	row.className = "particle-type-row"

	const deleteBtn = document.createElement("button")
	deleteBtn.className = "particle-card-delete"
	deleteBtn.textContent = "×"
	deleteBtn.title = "Delete organelle"
	deleteBtn.addEventListener("click", (e) => {
		e.stopPropagation()
		const removeIndices: number[] = []
		for (let i = 0; i < dots.particles.length; i++) {
			if (dots.particles[i].groupId === type) removeIndices.push(i)
		}
		dots.removeParticlesByIndices(removeIndices)
		dots.groupNames.delete(type)
		dots.groupColors.delete(type)
		dots.forceMatrixDirty = true

		const types = dots.getTypeIds()
		dots.forceMatrix = resizeMatrix(dots.forceMatrix, types)

		const openSections = getOpenSections(container)
		container.innerHTML = ""
		rebuildFn(container)
		restoreOpenSections(container, openSections)
		container.dispatchEvent(new Event("change", { bubbles: true }))
	})
	row.appendChild(deleteBtn)

	const picker = new ColorPicker(dots.rgbToHex(color))
	picker.element.classList.add("particle-type-swatch")
	picker.input.dataset.setting = `particle:${type}:color`
	picker.onChange((hex) => {
		const rgb = dots.hexToRgb(hex)
		dots.groupColors.set(type, rgb)
		for (const p of dots.particles) {
			if (p.groupId === type) {
				p.color[0] = rgb[0]
				p.color[1] = rgb[1]
				p.color[2] = rgb[2]
			}
		}
		dots.uploadParticleColors()
		syncMatrixHeaders(dots, container)
	})
	row.appendChild(picker.element)

	const nameInput = document.createElement("input")
	nameInput.type = "text"
	nameInput.className = "particle-type-name"
	nameInput.value = dots.groupNames.get(type) || type
	nameInput.dataset.setting = `particle:${type}:name`
	nameInput.addEventListener("input", () => {
		dots.groupNames.set(type, nameInput.value)
		syncMatrixHeaders(dots, container)
	})
	row.appendChild(nameInput)

	const countInput = document.createElement("input")
	countInput.type = "number"
	countInput.className = "particle-type-count"
	countInput.value = String(members.length)
	countInput.min = "0"
	countInput.step = "10"
	countInput.dataset.setting = `particle:${type}:count`
	countInput.addEventListener("wheel", (e) => {
		e.preventDefault()
		e.stopPropagation()
		const direction = e.deltaY > 0 ? -1 : e.deltaY < 0 ? 1 : 0
		if (direction === 0) return
		applyStepDelta(countInput, direction)
	})
	countInput.addEventListener("input", () => {
		const desired = Math.max(0, Number(countInput.value) || 0)
		const currentOfType = dots.particles.filter((p) => p.groupId === type)
		const currentCount = currentOfType.length

		if (desired === currentCount) return

		if (desired > currentCount) {
			const startIdx = dots.particles.length
			const toAdd = Math.min(
				desired - currentCount,
				MAX_PARTICLES - dots.particles.length,
			)
			if (toAdd <= 0) return
			const liveColor = dots.groupColors.get(type) ?? color
			for (let i = 0; i < toAdd; i++) {
				dots.particles.push(
					new CustomParticle(
						Math.random() * dots.width,
						Math.random() * dots.height,
						type,
						[liveColor[0], liveColor[1], liveColor[2]],
					),
				)
			}
			dots.count = dots.particles.length
			dots.uploadParticleRange(startIdx, toAdd)
		} else {
			const removeIndices: number[] = []
			for (
				let i = dots.particles.length - 1;
				i >= 0 && removeIndices.length < currentCount - desired;
				i--
			) {
				if (dots.particles[i].groupId === type) removeIndices.push(i)
			}
			dots.removeParticlesByIndices(removeIndices)
		}

		members.length = 0
		for (const p of dots.particles) {
			if (p.groupId === type) members.push(p)
		}
	})
	row.appendChild(countInput)

	section.appendChild(row)
}
