import type { RandomDots } from "../basic-particles"
import { MAX_PARTICLES } from "../../constants"
import { matrixFromJSON, matrixToJSON, resizeMatrix } from "../basic-particles"
import { CustomParticle } from "../../particles"
import { buildMatrixUI } from "../force-matrix-ui"
import { buildTypeRow } from "../ledger"
import { getOpenSections, makeSection, restoreOpenSections } from "./section"

export function buildParticlesWindow(dots: RandomDots, container: HTMLElement) {
	dots._particlesContainer = container
	// --- Organelles section ---
	const typesAccordion = makeSection("Organelles", false)

	const typeMap = new Map<string, CustomParticle[]>()
	for (const type of dots.groupNames.keys()) {
		typeMap.set(type, [])
	}
	for (const p of dots.particles) {
		let list = typeMap.get(p.groupId)
		if (!list) {
			list = []
			typeMap.set(p.groupId, list)
		}
		list.push(p)
	}

	const typesSection = document.createElement("div")
	typesSection.className = "particle-types-section"

	const rebuildParticles = (c: HTMLElement) => buildParticlesWindow(dots, c)
	for (const [type, members] of typeMap) {
		buildTypeRow(dots, typesSection, container, type, members, rebuildParticles)
	}

	// Add Particle button
	const addParticleBtn = document.createElement("button")
	addParticleBtn.className = "add-particle-btn"
	addParticleBtn.textContent = "+"
	addParticleBtn.title = "Add organelle"
	addParticleBtn.addEventListener("click", () => {
		const newType = dots.generateGroupId()
		const newName = dots.generateName()
		dots.groupNames.set(newType, newName)
		// Generate a random saturated color (HSL with S=0.7, L=0.6)
		const hue = Math.random()
		const newColor: [number, number, number] = (() => {
			const s = 0.7,
				l = 0.6
			const q = l + s - l * s
			const p = 2 * l - q
			const hue2rgb = (t: number) => {
				if (t < 0) t += 1
				if (t > 1) t -= 1
				if (t < 1 / 6) return p + (q - p) * 6 * t
				if (t < 1 / 2) return q
				if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
				return p
			}
			return [hue2rgb(hue + 1 / 3), hue2rgb(hue), hue2rgb(hue - 1 / 3)]
		})()
		dots.groupColors.set(newType, newColor)

		const newCount = Math.min(200, MAX_PARTICLES - dots.particles.length)
		if (newCount <= 0) return
		const startIdx = dots.particles.length
		for (let i = 0; i < newCount; i++) {
			dots.particles.push(
				new CustomParticle(
					Math.random() * dots.width,
					Math.random() * dots.height,
					newType,
					[newColor[0], newColor[1], newColor[2]],
				),
			)
		}
		dots.count = dots.particles.length
		dots.uploadParticleRange(startIdx, newCount)
		dots.forceMatrixDirty = true

		const types = dots.getTypeIds()
		dots.forceMatrix = resizeMatrix(dots.forceMatrix, types)

		// Rebuild this window, preserving accordion states
		const openSections = getOpenSections(container)
		container.innerHTML = ""
		buildParticlesWindow(dots, container)
		restoreOpenSections(container, openSections)
		container.dispatchEvent(new Event("change", { bubbles: true }))
	})
	typesSection.appendChild(addParticleBtn)

	typesAccordion.body.appendChild(typesSection)
	container.appendChild(typesAccordion.section)

	// --- Interaction Matrix section ---
	const matrixAccordion = makeSection("Interaction Matrix", false)
	buildMatrixUI(dots, matrixAccordion.body, container)

	const hiddenMatrixInput = document.createElement("input")
	hiddenMatrixInput.type = "hidden"
	hiddenMatrixInput.dataset.setting = "forceMatrix"
	hiddenMatrixInput.value = matrixToJSON(dots.forceMatrix)
	hiddenMatrixInput.addEventListener("input", () => {
		const types = dots.getTypeIds()
		const restored = matrixFromJSON(hiddenMatrixInput.value, types)
		if (matrixToJSON(restored) !== matrixToJSON(dots.forceMatrix)) {
			dots.forceMatrix = restored
			dots.forceMatrixDirty = true
			const matrixContainer = matrixAccordion.body.querySelector(
				".force-matrix-container",
			)
			if (matrixContainer) {
				matrixContainer.remove()
				buildMatrixUI(dots, matrixAccordion.body)
			}
		}
	})
	matrixAccordion.body.appendChild(hiddenMatrixInput)

	container.appendChild(matrixAccordion.section)
}
