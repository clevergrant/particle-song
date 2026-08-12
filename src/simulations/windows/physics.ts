import type { RandomDots } from "../basic-particles"
import type { VuMeter } from "../../ui/widgets/vu-meter"
import type { MiniGauge } from "../../ui/widgets/mini-gauge"
import { createNumberGroup, createToggleGroup } from "../../ui/ui-helpers"
import { CurveEditor } from "../../ui/curve-editor"
import { makeSection } from "./section"

export function buildPhysicsWindow(dots: RandomDots, container: HTMLElement) {
	// Pause toggle
	const pause = createToggleGroup({
		label: "Pause Simulation",
		checked: false,
		setting: "simPaused",
		onChange: (v) => {
			window.dispatchEvent(new CustomEvent("sim-pause", { detail: { paused: v } }))
		},
	})
	window.addEventListener("sim-pause", ((e: CustomEvent<{ paused: boolean }>) => {
		pause.checkbox.checked = e.detail.paused
	}) as EventListener)
	container.appendChild(pause.group)

	// Auto Balance toggle (top-level, like Enable Sound in music)
	const autoBalance = createToggleGroup({
		label: "Auto Balance",
		checked: dots.autoBalanceEnabled,
		setting: "autoBalance",
		onChange: () => {}, // real handler wired below (needs updateForceVisibility)
	})
	container.appendChild(autoBalance.group)

	// Read-only summary shown when auto-balance is ON
	const autoBalanceSummary = document.createElement("div")
	autoBalanceSummary.className = "auto-balance-summary"
	dots._autoBalanceSummary = autoBalanceSummary
	dots.renderAutoBalanceSummary()
	container.appendChild(autoBalanceSummary)

	// Container for all manual force sections (hidden when auto-balance ON)
	const forceSliders = document.createElement("div")

	// Force Reach
	const forceReach = makeSection("Force Reach", false)

	const affectRadiusEl = createNumberGroup({
		label: "Affect Radius",
		value: dots.affectRadius,
		setting: "affectRadius",
		min: 1,
		step: 1,
		suffix: "px",
		onInput: (v) => {
			dots.affectRadius = v
		},
	})
	dots._affectRadiusInput = affectRadiusEl
	forceReach.body.appendChild(affectRadiusEl)

	const forceRepelDistanceEl = createNumberGroup({
		label: "Force/Repel Distance",
		value: dots.forceRepelDistance,
		setting: "forceRepelDistance",
		min: 0,
		step: 1,
		suffix: "px",
		onInput: (v) => {
			dots.forceRepelDistance = v
		},
	})
	dots._forceRepelDistanceInput = forceRepelDistanceEl
	forceReach.body.appendChild(forceRepelDistanceEl)

	// Falloff curve editor (belongs with reach/distance)
	const falloffGroup = document.createElement("div")
	falloffGroup.className = "control-group control-group-column"
	const falloffLabel = document.createElement("label")
	falloffLabel.textContent = "Falloff Curve"
	falloffGroup.appendChild(falloffLabel)
	const falloffHint = document.createElement("div")
	falloffHint.className = "control-hint"
	falloffHint.textContent =
		"Dbl-click: add/remove • Right-click: remove • Drag handles"
	falloffGroup.appendChild(falloffHint)

	dots.curveEditor = new CurveEditor(falloffGroup)
	const hiddenCurveInput = document.createElement("input")
	hiddenCurveInput.type = "hidden"
	hiddenCurveInput.dataset.setting = "falloffCurve"
	hiddenCurveInput.value = dots.curveEditor.toJSON()
	dots.curveEditor.onChange((lut) => {
		dots.uploadFalloffLUT(lut)
		hiddenCurveInput.value = dots.curveEditor!.toJSON()
		hiddenCurveInput.dispatchEvent(new Event("input", { bubbles: true }))
	})
	hiddenCurveInput.addEventListener("input", () => {
		if (dots.curveEditor && hiddenCurveInput.value !== dots.curveEditor.toJSON()) {
			dots.curveEditor.fromJSON(hiddenCurveInput.value)
			dots.uploadFalloffLUT()
		}
	})
	falloffGroup.appendChild(hiddenCurveInput)
	forceReach.body.appendChild(falloffGroup)

	forceSliders.appendChild(forceReach.section)

	// Force Strength
	const forceStrength = makeSection("Force Strength", false)

	const forceStrengthVu = document.createElement("vu-meter") as VuMeter
	const baseStrengthEl = createNumberGroup({
		label: "Force Strength",
		value: dots.baseStrength,
		setting: "baseStrength",
		min: 1,
		step: 1,
		onInput: (v) => {
			dots.baseStrength = v
			forceStrengthVu.value = Math.min(1, v / 500)
		},
	})
	baseStrengthEl.appendChild(forceStrengthVu)
	forceStrengthVu.value = Math.min(1, dots.baseStrength / 500)
	dots._forceStrengthVu = forceStrengthVu
	dots._baseStrengthInput = baseStrengthEl
	forceStrength.body.appendChild(baseStrengthEl)

	const repelStrengthVu = document.createElement("vu-meter") as VuMeter
	const repelStrengthEl = createNumberGroup({
		label: "Repel Strength",
		value: dots.repelStrength,
		setting: "repelStrength",
		min: 0,
		step: 1,
		onInput: (v) => {
			dots.repelStrength = v
			repelStrengthVu.value = Math.min(1, v / 500)
		},
	})
	repelStrengthEl.appendChild(repelStrengthVu)
	repelStrengthVu.value = Math.min(1, dots.repelStrength / 500)
	dots._repelStrengthVu = repelStrengthVu
	dots._repelStrengthInput = repelStrengthEl
	forceStrength.body.appendChild(repelStrengthEl)

	forceSliders.appendChild(forceStrength.section)

	// Crowd Density
	const crowdDensity = makeSection("Crowd Density", false)

	const crowdLimitEl = createNumberGroup({
		label: "Crowd Limit",
		value: dots.crowdLimit,
		setting: "crowdLimit",
		min: 1,
		step: 1,
		onInput: (v) => {
			dots.crowdLimit = v
		},
	})
	dots._crowdLimitInput = crowdLimitEl
	crowdDensity.body.appendChild(crowdLimitEl)

	const spreadGauge = document.createElement("mini-gauge") as MiniGauge
	const spreadEl = createNumberGroup({
		label: "Spread",
		value: dots.spread,
		setting: "spread",
		min: 0,
		max: 100,
		step: 1,
		suffix: "%",
		onInput: (v) => {
			dots.spread = v
			spreadGauge.value = v / 100
		},
	})
	spreadEl.appendChild(spreadGauge)
	spreadGauge.value = dots.spread / 100
	dots._spreadGauge = spreadGauge
	dots._spreadInput = spreadEl
	crowdDensity.body.appendChild(spreadEl)

	forceSliders.appendChild(crowdDensity.section)

	// Toggle visibility based on auto-balance state
	const updateForceVisibility = () => {
		forceSliders.style.display = dots.autoBalanceEnabled ? "none" : ""
		autoBalanceSummary.style.display = dots.autoBalanceEnabled ? "" : "none"
	}
	updateForceVisibility()

	autoBalance.checkbox.addEventListener("change", () => {
		dots.autoBalanceEnabled = autoBalance.checkbox.checked
		updateForceVisibility()
		if (dots.autoBalanceEnabled) {
			dots.predictionDirty = true
		}
	})

	container.appendChild(forceSliders)

	// Speed Limiter — always visible, independent of auto-balance
	const speedLimiter = makeSection("Speed Limiter", false)

	speedLimiter.body.appendChild(
		createNumberGroup({
			label: "Max Speed",
			value: dots.maxSpeedPct,
			setting: "maxSpeedPct",
			min: 1,
			max: 100,
			step: 0.1,
			suffix: "%",
			onInput: (v) => {
				dots.maxSpeedPct = v
			},
		}),
	)

	container.appendChild(speedLimiter.section)
}
