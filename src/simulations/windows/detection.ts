import type { RandomDots } from "../basic-particles"
import { createNumberGroup, createToggleGroup } from "../../ui/ui-helpers"
import { initLedger } from "../ledger"
import { makeSection } from "./section"

export function buildDetectionWindow(dots: RandomDots, container: HTMLElement) {
	// Organelle Detection (algorithm parameters only)
	const organelleDet = makeSection("Organelle Detection", true)

	organelleDet.body.appendChild(
		createNumberGroup({
			label: "Search Radius",
			value: dots.detectionConfig.proximityRadius,
			setting: "detProximityRadius",
			min: 5,
			max: 200,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				dots.detectionConfig = { ...dots.detectionConfig, proximityRadius: v }
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)
	organelleDet.body.appendChild(
		createNumberGroup({
			label: "Coherence",
			value: dots.detectionConfig.coherenceThreshold,
			setting: "detCoherenceThreshold",
			min: 1,
			max: 200,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				dots.detectionConfig = { ...dots.detectionConfig, coherenceThreshold: v }
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)
	organelleDet.body.appendChild(
		createNumberGroup({
			label: "Min Size",
			value: dots.detectionConfig.minOrganelleSize,
			setting: "detMinOrganelleSize",
			min: 2,
			max: 50,
			step: 1,
			suffix: "particles",
			onInput: (v) => {
				dots.detectionConfig = { ...dots.detectionConfig, minOrganelleSize: v }
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)
	organelleDet.body.appendChild(
		createNumberGroup({
			label: "Latch Duration",
			value: dots.detectionConfig.organelleLatchSeconds,
			setting: "detOrganelleLatchSeconds",
			min: 0.05,
			max: 4,
			step: 0.05,
			suffix: "s",
			onInput: (v) => {
				dots.detectionConfig = { ...dots.detectionConfig, organelleLatchSeconds: v }
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)

	container.appendChild(organelleDet.section)

	// Organism Detection (algorithm parameters only)
	const organismDet = makeSection("Organism Detection", true)

	organismDet.body.appendChild(
		createNumberGroup({
			label: "Search Radius",
			value: dots.detectionConfig.organismProximityRadius,
			setting: "detOrganismProximityRadius",
			min: 20,
			max: 400,
			step: 10,
			suffix: "px",
			onInput: (v) => {
				dots.detectionConfig = {
					...dots.detectionConfig,
					organismProximityRadius: v,
				}
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)
	organismDet.body.appendChild(
		createNumberGroup({
			label: "Coherence",
			value: dots.detectionConfig.organismCoherenceThreshold,
			setting: "detOrganismCoherenceThreshold",
			min: 10,
			max: 200,
			step: 10,
			suffix: "px",
			onInput: (v) => {
				dots.detectionConfig = {
					...dots.detectionConfig,
					organismCoherenceThreshold: v,
				}
				dots.detectionState = null
				dots.prevFrameWire = null
			},
		}),
	)

	container.appendChild(organismDet.section)

	// Detection Overlays (all visual toggles and rendering params)
	const overlays = makeSection("Detection Overlays", true)

	overlays.body.appendChild(
		createToggleGroup({
			label: "Organelle Overlay",
			checked: dots.showOrganelleOverlay,
			setting: "detShowOrganelleOverlay",
			onChange: (v) => {
				dots.showOrganelleOverlay = v
				if (!v && dots.detectionBuffer && dots.device) {
					dots.device.queue.writeBuffer(
						dots.detectionBuffer,
						0,
						new Uint32Array(dots.count),
					)
				}
			},
		}).group,
	)

	overlays.body.appendChild(
		createToggleGroup({
			label: "Organism Outline",
			checked: dots.showOrganismOverlay,
			setting: "detShowOrganismOverlay",
			onChange: (v) => {
				dots.showOrganismOverlay = v
			},
		}).group,
	)

	overlays.body.appendChild(
		createToggleGroup({
			label: "Centroids & Connectors",
			checked: dots.showOrganismCentroids,
			setting: "detShowOrganismCentroids",
			onChange: (v) => {
				dots.showOrganismCentroids = v
				if (!v) {
					dots.organismCentroidCount = 0
					dots.organismLineCount = 0
					dots.organismLineEdges = []
				}
			},
		}).group,
	)

	// Bubble boundary rendering controls
	overlays.body.appendChild(
		createNumberGroup({
			label: "Bubble Threshold",
			value: dots.bubbleThreshold,
			setting: "detBubbleThreshold",
			min: 10,
			max: 150,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				dots.bubbleThreshold = v
				dots.uploadBubbleParams()
			},
		}),
	)
	overlays.body.appendChild(
		createNumberGroup({
			label: "Edge Width",
			value: dots.bubbleEdgeWidth,
			setting: "detBubbleEdgeWidth",
			min: 1,
			max: 10,
			step: 0.5,
			suffix: "px",
			onInput: (v) => {
				dots.bubbleEdgeWidth = v
				dots.uploadBubbleParams()
			},
		}),
	)

	container.appendChild(overlays.section)

	// Ledger
	initLedger(dots)
}
