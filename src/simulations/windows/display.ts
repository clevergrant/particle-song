import type { RandomDots } from "../basic-particles"
import { createNumberGroup, createToggleGroup } from "../../ui/ui-helpers"
import { makeSection } from "./section"

export function buildDisplayWindow(dots: RandomDots, container: HTMLElement) {
	// Viewport
	const viewport = makeSection("Viewport", true)

	viewport.body.appendChild(
		createNumberGroup({
			label: "Scale",
			value: dots.scale,
			setting: "scale",
			min: 0.1,
			max: 5,
			step: 0.1,
			suffix: "x",
			onInput: (v) => {
				dots.scale = v
			},
		}),
	)

	viewport.body.appendChild(
		createToggleGroup({
			label: "Show Particles",
			checked: dots.showCircleOverlay,
			setting: "showParticles",
			onChange: (v) => {
				dots.showCircleOverlay = v
			},
		}).group,
	)

	container.appendChild(viewport.section)

	// Particle Appearance
	const appearance = makeSection("Particle Appearance", true)

	appearance.body.appendChild(
		createNumberGroup({
			label: "Radius",
			value: dots.pointSize,
			setting: "radius",
			min: 2,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				dots.pointSize = v
				dots.uploadRenderParams()
			},
		}),
	)

	appearance.body.appendChild(
		createNumberGroup({
			label: "Pulse Scale",
			value: dots.pulseScale,
			setting: "pulseScale",
			min: 0.1,
			step: 0.1,
			suffix: "x",
			onInput: (v) => {
				dots.pulseScale = v
			},
		}),
	)

	container.appendChild(appearance.section)
}
