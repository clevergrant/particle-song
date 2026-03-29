import { autoBalance } from "../auto-balance"
import { ColorPicker } from "../color-picker"
import { CurveEditor } from "../curve-editor"
import {
	DEFAULT_DETECTION_CONFIG,
	runDetection,
	updateRegistry,
	type DetectionConfig,
	type DetectionFrame,
	type OrganelleState,
	type OrganelleTreeNode,
	type OrganismRegistry,
	type ReadbackData,
} from "../detection"
import { EnvelopeEditor, buildGateAwareLUT } from "../envelope-editor"
import { applyStepDelta } from "../number-scroll"
import {
	predictOrganisms,
	type OrganismPrediction,
} from "../organism-prediction"
import { CustomParticle, type ForceMatrix } from "../particles"
import type { GpuContext, Simulation, WindowDefinition } from "../types"
import { createNumberGroup } from "../ui-helpers"
import type { MiniClock } from "../ui/widgets/mini-clock"
import type { MiniGauge } from "../ui/widgets/mini-gauge"
import type { StabilityBars } from "../ui/widgets/stability-bars"
import type { VuMeter } from "../ui/widgets/vu-meter"

// New music pipeline
import { AudioGraph } from "../music/audio-graph"
import {
	hideBarVisualizer,
	onPhraseChange,
	setPhraseStripCells,
	showBarVisualizer,
	updateBarVisualizer,
} from "../music/bar-visualizer"
import {
	BassLayer,
	DEFAULT_PHRASE_CELLS,
	expandPhrase,
	type BassDensity,
} from "../music/bass-layer"
import { computeGlobalMetrics } from "../music/global-metrics"
import { scheduleBar, type ScheduleConfig } from "../music/hit-scheduler"
import { applyVoiceBudget } from "../music/voice-budget"
import {
	barDuration,
	barStartTime as barStartTimeFn,
	checkBarBoundary,
} from "../music/timing"
import type {
	BarSnapshot,
	GlobalMetrics,
	MusicState,
	ScheduledBar,
	SnapshotOrganelle,
	SnapshotOrganism,
} from "../music/types"

import {
	PARTICLE_PREFIX,
	buildParticleShader,
	buildQuadShader,
	effectDefaults,
	findParticleEffect,
	findPostEffect,
	particleEffects,
	postEffects,
	type ShaderEffect,
} from "../shader-registry"
import jfaComputeSrc from "../shaders/jfa.compute.wgsl?raw"
import computeShaderSrc from "../shaders/particles.compute.wgsl?raw"
import stainUpdateSrc from "../shaders/stain-update.wgsl?raw"

/* ------------------------------------------------------------------ */
/*  Pure helpers for the force matrix                                  */
/* ------------------------------------------------------------------ */

function emptyMatrix(types: readonly string[]): ForceMatrix {
	const m: Record<string, Record<string, number>> = {}
	for (const src of types) {
		const row: Record<string, number> = {}
		for (const tgt of types) row[tgt] = 0
		m[src] = row
	}
	return m
}

function randomizeMatrix(types: readonly string[]): ForceMatrix {
	const m: Record<string, Record<string, number>> = {}
	for (const src of types) {
		const row: Record<string, number> = {}
		for (const tgt of types)
			row[tgt] = Math.round((Math.random() * 2 - 1) * 100) / 100
		m[src] = row
	}
	return m
}

function resizeMatrix(
	prev: ForceMatrix,
	types: readonly string[],
): ForceMatrix {
	const m: Record<string, Record<string, number>> = {}
	for (const src of types) {
		const row: Record<string, number> = {}
		for (const tgt of types) row[tgt] = prev[src]?.[tgt] ?? 0
		m[src] = row
	}
	return m
}

function matrixToJSON(matrix: ForceMatrix): string {
	return JSON.stringify(matrix)
}

function matrixFromJSON(json: string, types: readonly string[]): ForceMatrix {
	try {
		const raw = JSON.parse(json) as Record<string, Record<string, number>>
		return resizeMatrix(raw, types)
	} catch {
		return emptyMatrix(types)
	}
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const MAX_PARTICLES = 10000
const MAX_TYPES = 32
const PARTICLE_STRIDE = 48 // bytes per Particle struct (must match WGSL)
const WORKGROUP_SIZE = 64

/* ------------------------------------------------------------------ */
/*  Simulation                                                         */
/* ------------------------------------------------------------------ */

export class RandomDots implements Simulation {
	name = "Random Dots"
	settingsVersion = "2026-03-28r"

	// GPU resources
	private device: GPUDevice | null = null
	private canvas: HTMLCanvasElement | null = null
	private canvasContext: GPUCanvasContext | null = null
	private canvasFormat: GPUTextureFormat = "bgra8unorm"

	// Compute pipeline
	private computePipeline: GPUComputePipeline | null = null
	private computeBindGroupLayout: GPUBindGroupLayout | null = null
	private computeBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	private particleBuffers: [GPUBuffer | null, GPUBuffer | null] = [null, null]
	private particleStagingBuffer: GPUBuffer | null = null
	private simParamsBuffer: GPUBuffer | null = null
	private forceMatrixBuffer: GPUBuffer | null = null
	private stressBuffer: GPUBuffer | null = null
	private peakUpdatePipeline: GPUComputePipeline | null = null
	private pingPong = 0

	// Render pipeline (particles)
	private particleRenderPipeline: GPURenderPipeline | null = null
	private particleRenderBindGroupLayout: GPUBindGroupLayout | null = null
	private particleRenderBindGroups: [GPUBindGroup | null, GPUBindGroup | null] =
		[null, null]
	private renderParamsBuffer: GPUBuffer | null = null
	private falloffTexture: GPUTexture | null = null
	private falloffSampler: GPUSampler | null = null

	// Render pipeline (circle overlay) — same shaders, different blend + mode
	private circleRenderPipeline: GPURenderPipeline | null = null
	private circleRenderParamsBuffer: GPUBuffer | null = null
	private circleRenderBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]

	// Detection pipelines: fill organelle IDs to R8Uint + color to RGBA8, edge-detect outlines
	private detectionIdTexture: GPUTexture | null = null
	private detectionColorTexture: GPUTexture | null = null
	private detectionFillPipeline: GPURenderPipeline | null = null
	private detectionEdgePipeline: GPURenderPipeline | null = null
	private detectionFillParamsBuffer: GPUBuffer | null = null
	private detectionFillBindGroups: [GPUBindGroup | null, GPUBindGroup | null] =
		[null, null]
	private detectionEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	private detectionEdgeBindGroup: GPUBindGroup | null = null

	// Organism outline pipelines: fill organism IDs to R8Uint (inflated particles), edge-detect white outlines
	private organismIdTexture: GPUTexture | null = null
	private organismFillPipeline: GPURenderPipeline | null = null
	private organismFillParamsBuffer: GPUBuffer | null = null
	private organismFillBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	private organismEdgePipeline: GPURenderPipeline | null = null
	private organismEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	private organismEdgeBindGroup: GPUBindGroup | null = null

	// JFA bubble boundary system
	private jfaOrganelleTextures: [GPUTexture | null, GPUTexture | null] = [
		null,
		null,
	]
	private jfaOrganismTextures: [GPUTexture | null, GPUTexture | null] = [
		null,
		null,
	]
	private jfaComputePipeline: GPUComputePipeline | null = null
	private jfaComputeBindGroupLayout: GPUBindGroupLayout | null = null
	private jfaParamsBuffers: GPUBuffer[] = []
	private jfaOrganelleBindGroups: [GPUBindGroup[], GPUBindGroup[]] = [[], []]
	private jfaOrganismBindGroups: [GPUBindGroup[], GPUBindGroup[]] = [[], []]
	private jfaPassCount = 0
	private jfaOrganelleEdgeBindGroups: [
		GPUBindGroup | null,
		GPUBindGroup | null,
	] = [null, null]
	private jfaOrganismEdgeBindGroups: [
		GPUBindGroup | null,
		GPUBindGroup | null,
	] = [null, null]
	private jfaEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	private jfaOrganelleEdgePipeline: GPURenderPipeline | null = null
	private jfaOrganismEdgePipeline: GPURenderPipeline | null = null
	private jfaOrganelleSeedPipeline: GPURenderPipeline | null = null
	private jfaOrganismSeedPipeline: GPURenderPipeline | null = null
	private jfaOrganismCentroidSeedPipeline: GPURenderPipeline | null = null
	private jfaOrganismLineSeedPipeline: GPURenderPipeline | null = null
	private bubbleParamsBuffer: GPUBuffer | null = null
	private bubbleThreshold = 5
	private bubbleEdgeWidth = 3

	// Organism centroid circle overlay
	private organismCentroidPipeline: GPURenderPipeline | null = null
	private organismCentroidBindGroupLayout: GPUBindGroupLayout | null = null
	private organismCentroidBindGroup: GPUBindGroup | null = null
	private organismCentroidBuffer: GPUBuffer | null = null
	private organismCentroidParamsBuffer: GPUBuffer | null = null
	private organismCentroidCount = 0
	private organismCentroidSnapshot: {
		cx: number
		cy: number
		vx: number
		vy: number
		id: number
	}[] = []
	private organismCentroidSnapshotTime = 0

	// Organism connection lines (edges between organelle centroids in same organism)
	private organismLinePipeline: GPURenderPipeline | null = null
	private organismLineFillPipeline: GPURenderPipeline | null = null
	private organismLineBuffer: GPUBuffer | null = null
	private organismLineBindGroup: GPUBindGroup | null = null
	private organismLineCount = 0
	private organismLineEdges: [number, number][] = [] // pairs of indices into centroid snapshot

	// Fill-variant pipeline for centroid circles (writes organism ID to r8uint)
	private organismCentroidFillPipeline: GPURenderPipeline | null = null

	// Organism-level centroid circles (larger, wrapping organelle centroids)
	private osmLevelCentroidPipeline: GPURenderPipeline | null = null
	private osmLevelCentroidBindGroup: GPUBindGroup | null = null
	private osmLevelCentroidBuffer: GPUBuffer | null = null
	private osmLevelCentroidCount = 0
	private osmLevelCentroidSnapshot: {
		cx: number
		cy: number
		vx: number
		vy: number
		id: number
	}[] = []
	private osmLevelCentroidSnapshotTime = 0

	// Quad (fullscreen normalization) pipeline
	private quadPipeline: GPURenderPipeline | null = null
	private quadBindGroupLayout: GPUBindGroupLayout | null = null
	private quadBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	private quadParamsBuffer: GPUBuffer | null = null
	private offscreenTexture: GPUTexture | null = null
	private offscreenView: GPUTextureView | null = null
	private offscreenSampler: GPUSampler | null = null

	// Stain (phosphor persistence) — ping-pong pair
	private stainTextures: [GPUTexture | null, GPUTexture | null] = [null, null]
	private stainViews: [GPUTextureView | null, GPUTextureView | null] = [null, null]
	private stainPingPong = 0
	private stainPipeline: GPURenderPipeline | null = null
	private stainBindGroupLayout: GPUBindGroupLayout | null = null
	private stainBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [null, null]
	private stainParamsBuffer: GPUBuffer | null = null

	// Simulation state (CPU-side, for UI and buffer uploads)
	private count = 500
	private width = 0
	private height = 0
	private particles: CustomParticle[] = []
	private nextGroupId = 0
	private groupNames = new Map<string, string>()
	private groupColors = new Map<string, [number, number, number]>()
	private showCircleOverlay = false
	private prevVelX: Float32Array | null = null // previous-frame per-particle vx
	private prevVelY: Float32Array | null = null // previous-frame per-particle vy
	private pointSize = 27.0
	private pulseScale = 6
	private curveEditor: CurveEditor | null = null
	private staccatoCurveEditor: CurveEditor | null = null
	private lpfCurveEditor: CurveEditor | null = null
	private envelopeEditor: EnvelopeEditor | null = null
	private bassEnvelopeEditor: EnvelopeEditor | null = null

	// Per-effect shader params (effectId → [param0, param1, ...])
	private particleEffectParams: Record<string, number[]> = {
		gradient: [1.1, 0, 0, 0],
		solid: [1, 0.4, 0, 0],
		"speed-color": [0.66, 0.5, 0, 0],
		"stress-color": [1, 0.65, 0, 0],
	}
	private postEffectParams: Record<string, number[]> = {
		normalize: [0.6, 0.8, 0, 0],
		chromatic: [0.6, 0.8, 0.009, 0],
		crt: [0.6, 0.8, 0.15, 800],
		palette: [0.6, 0, 0, 0],
		metaball: [0.45, 1.3, 0.8, 0],
		duotone: [0.6, 0, 0, 0],
		"edge-glow": [0.6, 0.8, 3, 0.3],
		stain: [0.6, 0.8, 0.012, 0],
	}

	// Force matrix (single source of truth for inter-type forces)
	private forceMatrix: ForceMatrix = {}
	private autoRandomizeMatrixEnabled = true
	private autoRandomizeCountsEnabled = true
	private affectRadius = 61.1
	private forceRepelDistance = 40.72
	private baseStrength = 207.94

	// Density regulation (user-facing controls)
	private crowdLimit = 29.06
	private spread = 26 // 0–100%
	private maxSpeedPct = 100 // 1–100%, soft speed limiter

	// Universal repulsion between all particles
	private repelStrength = 147.61

	// Scale: spatial zoom multiplier (1 = default, 2 = everything twice as large)
	private scale = 0.5

	// Auto-balance: derive physics params from force matrix + particle counts
	private autoBalanceEnabled = true

	// Accumulated time for animated shaders
	private time = 0

	// Active shader effects (one per category)
	private activeParticleEffect: ShaderEffect = particleEffects[0]
	private activePostEffect: ShaderEffect = findPostEffect("metaball")

	// Callbacks for shader menu sync
	onParticleEffectChanged: ((id: string) => void) | null = null
	onPostEffectChanged: ((id: string) => void) | null = null

	// Hidden inputs for settings persistence
	private _hiddenParticleEffect: HTMLInputElement | null = null
	private _hiddenPostEffect: HTMLInputElement | null = null
	private _hiddenParticleParams: HTMLInputElement | null = null
	private _hiddenPostParams: HTMLInputElement | null = null

	// Mouse interaction state
	private mouseX = 0
	private mouseY = 0
	private mouseLeft = false
	private mouseRight = false
	private mouseForceRadius = 200
	private mouseForceStrength = 5000
	private boundMouseMove: ((e: MouseEvent) => void) | null = null
	private boundMouseDown: ((e: MouseEvent) => void) | null = null
	private boundMouseUp: ((e: MouseEvent) => void) | null = null
	private boundContextMenu: ((e: MouseEvent) => void) | null = null

	// Slider references for auto-balance sync
	private _affectRadiusInput: HTMLElement | null = null
	private _forceRepelDistanceInput: HTMLElement | null = null
	private _baseStrengthInput: HTMLElement | null = null
	private _repelStrengthInput: HTMLElement | null = null
	private _crowdLimitInput: HTMLElement | null = null
	private _spreadInput: HTMLElement | null = null
	private _autoBalanceSummary: HTMLElement | null = null

	// Skeuomorphic widget references (updated in update loop)
	private _volumeVu: VuMeter | null = null
	private _forceStrengthVu: VuMeter | null = null
	private _repelStrengthVu: VuMeter | null = null
	private _bpmGauge: MiniGauge | null = null
	private _spreadGauge: MiniGauge | null = null
	private _stabilityBars: StabilityBars | null = null
	private _latchClock: MiniClock | null = null
	private _phaseClock: MiniClock | null = null

	// DOM refs for auto-randomize UI sync from update loop
	private _matrixWrapper: HTMLElement | null = null
	private _matrixContainer: HTMLElement | null = null
	private _matrixRootContainer: HTMLElement | null = null
	private _particlesContainer: HTMLElement | null = null
	private _autoRandomizeMatrixClock: MiniClock | null = null
	private _autoRandomizeCountsClock: MiniClock | null = null

	// Dirty flags — avoid re-uploading every frame
	private forceMatrixDirty = true
	private particleBufferDirty = true

	// Music engine (bar-boundary architecture)
	private audioGraph = new AudioGraph()
	private bassLayer = new BassLayer()
	private musicState: MusicState | null = null
	private currentScheduledBar: ScheduledBar | null = null
	private musicBarNumber = -1
	private tSoundStart = 0
	private musicBpm = 90
	private musicTimeMultiplier = 1
	private musicBeatsPerBar = 4
	private overtonePhaseRate = 1
	private qualificationFraction = 0.5
	private preferNiceModes = false
	private phrasePattern: BassDensity[] = [...DEFAULT_PHRASE_CELLS]
	private phraseMirror = false
	private voiceBudget = 32
	private latestGlobalMetrics: GlobalMetrics | null = null
	private mutedOrganisms = new Set<string>()
	private readbackBuffer: GPUBuffer | null = null
	private readbackPending = false
	private frameCounter = 0

	// Detection
	private detectionState: DetectionFrame | null = null
	private detectionConfig: DetectionConfig = { ...DEFAULT_DETECTION_CONFIG }

	// Organism registry (stable identity across frames)
	private organismRegistry: OrganismRegistry | null = null
	private lastReadbackTime = 0
	private ledgerToggle: HTMLElement | null = null
	private ledgerPanels: HTMLElement | null = null
	private ledgerBackdrop: HTMLElement | null = null
	private ledgerOrganellesEl: HTMLElement | null = null
	private ledgerOrganismsEl: HTMLElement | null = null
	private organelleRows = new Map<
		number,
		{ row: HTMLElement; countEl: HTMLElement }
	>()
	private organismRows = new Map<
		string,
		{ row: HTMLElement; countEl: HTMLElement; muteBtn: HTMLElement }
	>()
	private organelleHeading: HTMLElement | null = null
	private organismHeading: HTMLElement | null = null
	private unmuteAllBtn: HTMLElement | null = null
	private showOrganelleOverlay = false
	private showOrganismOverlay = false
	private showOrganismCentroids = false
	private organismDepthRanks = new Map<number, number>() // osmId (1-based) → depth rank
	private organismDepthTexture: GPUTexture | null = null
	private organismPrediction: OrganismPrediction | null = null
	private predictionDirty = true
	private ledgerPredictionsEl: HTMLElement | null = null
	private predictionHeading: HTMLElement | null = null
	private predictionRows = new Map<
		string,
		{ row: HTMLElement; scoreEl: HTMLElement }
	>()
	private speciesPresence = new Map<string, number>() // sig → presence score [0,1]
	private speciesBrightness = new Map<string, number>() // sig → visual brightness [0,1]
	private speciesDecayThreshold = 0.05
	private speciesDecaySlider: HTMLInputElement | null = null
	private lastPredictionTime = 0
	private detectionBuffer: GPUBuffer | null = null
	private radiusScaleBuffer: GPUBuffer | null = null

	// Active organelle pulses: key = "organismId:typeId:beatIndex"
	// particleIndices snapshotted at trigger time so detection changes don't reset the pulse
	private activePulses = new Map<
		string,
		{
			startTime: number
			duration: number
			particleIndices: Uint32Array
			attackFrac: number // fraction of duration for attack
			decayFrac: number // fraction of duration for decay
			sustainFrac: number // fraction of duration for sustain hold
			peakLevel: number
			sustainLevel: number
			envelopeLut: Float32Array | null // manual envelope LUT, overrides ADSR
		}
	>()

	/* ================================================================ */
	/*  effective params — base values × scale                           */
	/* ================================================================ */

	private getEffectiveParams() {
		const s = this.scale
		return {
			affectRadius: this.affectRadius * s,
			forceRepelDistance: this.forceRepelDistance * s,
			baseStrength: this.baseStrength * s,
			repelStrength: this.repelStrength * s,
			crowdLimit: this.crowdLimit, // count-based, not spatial
			spread: this.spread, // percentage, not spatial
			pointSize: this.pointSize * s,
			pulseScale: this.pulseScale * s,
			mouseForceRadius: this.mouseForceRadius * s,
		}
	}

	private syncAutoBalanceSliders() {
		const pairs: [HTMLElement | null, number][] = [
			[this._affectRadiusInput, this.affectRadius],
			[this._forceRepelDistanceInput, this.forceRepelDistance],
			[this._baseStrengthInput, this.baseStrength],
			[this._repelStrengthInput, this.repelStrength],
			[this._crowdLimitInput, this.crowdLimit],
			[this._spreadInput, this.spread],
		]
		for (const [el, value] of pairs) {
			if (!el) continue
			const input = (el as any).input as HTMLInputElement | undefined
			if (input) input.value = String(Math.round(value * 100) / 100)
		}

		// Update read-only summary
		this.renderAutoBalanceSummary()
	}

	private renderAutoBalanceSummary() {
		const el = this._autoBalanceSummary
		if (!el) return
		const items: [string, number][] = [
			["rad", Math.round(this.affectRadius)],
			["repD", Math.round(this.forceRepelDistance)],
			["str", Math.round(this.baseStrength)],
			["rep", Math.round(this.repelStrength)],
			["cwd", Math.round(this.crowdLimit)],
			["spr", this.spread],
		]
		el.innerHTML = items
			.map(
				([label, value]) =>
					`<div class="ab-item"><span class="ab-label">${label}</span><span class="ab-value">${value}</span></div>`,
			)
			.join("")
	}

	/* ================================================================ */
	/*  setup                                                            */
	/* ================================================================ */

	setup(gpu: GpuContext, width: number, height: number) {
		this.cleanup()
		this.initEffectParams()
		this.audioGraph = new AudioGraph()
		this.bassLayer = new BassLayer()
		this.device = gpu.device
		this.canvas = gpu.canvas
		this.canvasContext = gpu.canvasContext
		this.canvasFormat = gpu.format
		this.width = width
		this.height = height

		// --- Create particles from saved config or defaults ---
		this.particles = []
		this.nextGroupId = 0
		this.groupNames.clear()
		this.groupColors.clear()
		const savedConfig = this.readSavedParticleConfig()
		if (savedConfig) {
			for (const cfg of savedConfig.types) {
				this.groupNames.set(cfg.type, cfg.name)
				this.groupColors.set(cfg.type, cfg.color)
				const idNum = parseInt(cfg.type.slice(1))
				if (!isNaN(idNum) && idNum >= this.nextGroupId) {
					this.nextGroupId = idNum + 1
				}
				const cappedCount = Math.min(
					cfg.count,
					MAX_PARTICLES - this.particles.length,
				)
				for (let i = 0; i < cappedCount; i++) {
					this.particles.push(
						new CustomParticle(
							Math.random() * width,
							Math.random() * height,
							cfg.type,
							[cfg.color[0], cfg.color[1], cfg.color[2]],
						),
					)
				}
			}
			const types = savedConfig.types.map((c) => c.type)
			this.forceMatrix = savedConfig.matrix
				? matrixFromJSON(savedConfig.matrix, types)
				: emptyMatrix(types)
		} else {
			const defaults: {
				name: string
				color: [number, number, number]
				count: number
			}[] = [
				{ name: "R", color: this.hexToRgb("#ff4c2e"), count: 0 },
				{ name: "O", color: this.hexToRgb("#fe792e"), count: 0 },
				{ name: "Y", color: this.hexToRgb("#fdd42c"), count: 497 },
				{ name: "L", color: this.hexToRgb("#baff15"), count: 795 },
				{ name: "G", color: this.hexToRgb("#64b53c"), count: 0 },
				{ name: "B", color: this.hexToRgb("#89d6e8"), count: 425 },
				{ name: "I", color: this.hexToRgb("#3f5d93"), count: 0 },
				{ name: "V", color: this.hexToRgb("#ff63a8"), count: 1537 },
				{ name: "P", color: this.hexToRgb("#6464ff"), count: 1158 },
			]
			const typeIds: string[] = []
			for (const def of defaults) {
				const typeId = this.generateGroupId()
				typeIds.push(typeId)
				this.groupNames.set(typeId, def.name)
				this.groupColors.set(typeId, def.color)
				for (let i = 0; i < def.count; i++) {
					this.particles.push(
						new CustomParticle(
							Math.random() * width,
							Math.random() * height,
							typeId,
							[def.color[0], def.color[1], def.color[2]],
						),
					)
				}
			}
			this.forceMatrix = matrixFromJSON(
				JSON.stringify({
					p0: {
						p0: 0.27,
						p1: 0.15,
						p2: -0.18,
						p3: 0.88,
						p4: 0.84,
						p5: -0.83,
						p6: 0.22,
						p7: -0.95,
						p8: 0.16,
					},
					p1: {
						p0: 0.46,
						p1: -0.35,
						p2: 0.26,
						p3: 0.52,
						p4: 0.55,
						p5: -0.28,
						p6: -0.27,
						p7: -0.46,
						p8: -0.74,
					},
					p2: {
						p0: 0.69,
						p1: 0.44,
						p2: -0.99,
						p3: -0.33,
						p4: 0.65,
						p5: 0.38,
						p6: -0.76,
						p7: -0.4,
						p8: 0.17,
					},
					p3: {
						p0: 0.27,
						p1: -0.67,
						p2: 0.69,
						p3: -0.72,
						p4: 0.48,
						p5: 0.26,
						p6: 0.22,
						p7: 0.89,
						p8: 0.12,
					},
					p4: {
						p0: 0.6,
						p1: -0.44,
						p2: -0.41,
						p3: -0.37,
						p4: 0.75,
						p5: 0.46,
						p6: -0.65,
						p7: 0.83,
						p8: 0.94,
					},
					p5: {
						p0: 0.07,
						p1: 0.6,
						p2: -0.13,
						p3: -0.08,
						p4: 0.38,
						p5: 0.55,
						p6: 0.53,
						p7: 0.94,
						p8: 0.24,
					},
					p6: {
						p0: -0.13,
						p1: -0.75,
						p2: 0.75,
						p3: 0.02,
						p4: -0.41,
						p5: -0.82,
						p6: 0.95,
						p7: 0.58,
						p8: 0.85,
					},
					p7: {
						p0: 0.39,
						p1: -0.23,
						p2: 0.89,
						p3: -0.59,
						p4: -0.48,
						p5: 0.38,
						p6: -0.13,
						p7: -0.38,
						p8: 0.34,
					},
					p8: {
						p0: 0.9,
						p1: 0.99,
						p2: -0.22,
						p3: 0.03,
						p4: -0.58,
						p5: -0.43,
						p6: -0.23,
						p7: -0.23,
						p8: 0.69,
					},
				}),
				typeIds,
			)
		}
		this.count = this.particles.length

		// --- GPU resource creation ---
		this.createBuffers()
		this.createComputePipeline()
		this.createRenderPipelines()
		this.createOffscreenTexture()
		this.createQuadPipeline()
		this.createStainPipeline()
		this.rebuildAllBindGroups()

		// Upload initial data
		this.particleBufferDirty = true
		this.forceMatrixDirty = true
		this.uploadParticleData()
		this.uploadForceMatrix()
		this.uploadRenderParams()
		this.uploadQuadParams()

		// Mouse interaction listeners
		this.boundMouseMove = (e: MouseEvent) => {
			const rect = this.canvas!.getBoundingClientRect()
			this.mouseX = e.clientX - rect.left
			this.mouseY = e.clientY - rect.top
		}
		this.boundMouseDown = (e: MouseEvent) => {
			if (e.button === 0) this.mouseLeft = true
			if (e.button === 2) this.mouseRight = true
		}
		this.boundMouseUp = (e: MouseEvent) => {
			if (e.button === 0) this.mouseLeft = false
			if (e.button === 2) this.mouseRight = false
		}
		this.boundContextMenu = (e: MouseEvent) => e.preventDefault()

		this.canvas.addEventListener("mousemove", this.boundMouseMove)
		this.canvas.addEventListener("mousedown", this.boundMouseDown)
		window.addEventListener("mouseup", this.boundMouseUp)
		this.canvas.addEventListener("contextmenu", this.boundContextMenu)
	}

	/* ================================================================ */
	/*  resize                                                           */
	/* ================================================================ */

	resize(gpu: GpuContext, width: number, height: number) {
		const oldW = this.width
		const oldH = this.height
		this.width = width
		this.height = height

		// Remap particle positions proportionally
		if (oldW > 0 && oldH > 0) {
			const sx = width / oldW
			const sy = height / oldH
			for (const p of this.particles) {
				p.x *= sx
				p.y *= sy
				p.wrapPosition(width, height)
			}
			this.particleBufferDirty = true
			this.uploadParticleData()
		}

		// Recreate offscreen texture at new size
		this.createOffscreenTexture()
		// Rebuild bind groups since texture views changed
		this.rebuildQuadBindGroup()
		this.rebuildStainBindGroups()
		this.rebuildCircleRenderBindGroups()
	}

	/* ================================================================ */
	/*  update — just upload uniforms, compute shader does the physics   */
	/* ================================================================ */

	update(dt: number) {
		const device = this.device!
		this.time += dt

		// Upload render params every frame (time changes)
		this.uploadRenderParams()
		this.uploadQuadParams()

		// Upload sim params every frame (dt, mouse state change)
		const mouseActive = this.mouseLeft ? 1 : this.mouseRight ? 2 : 0
		const types = this.getTypeIds()
		const eff = this.getEffectiveParams()
		const paramData = new Float32Array(20) // 80 bytes = 20 f32s
		paramData[0] = this.width
		paramData[1] = this.height
		const halfOffset = eff.forceRepelDistance / 2
		const interactionRadius = eff.affectRadius + halfOffset
		const repelRadius = Math.max(1, eff.affectRadius - halfOffset)
		paramData[2] = interactionRadius
		paramData[3] = eff.baseStrength
		paramData[4] = 0.97 // damping
		paramData[5] = dt
		// u32 fields written as float bits
		new Uint32Array(paramData.buffer)[6] = this.count
		new Uint32Array(paramData.buffer)[7] = types.length
		paramData[8] = this.mouseX
		paramData[9] = this.mouseY
		paramData[10] = eff.mouseForceRadius
		paramData[11] = this.mouseForceStrength
		new Uint32Array(paramData.buffer)[12] = mouseActive
		// Derive GPU params from user-facing controls
		paramData[13] = interactionRadius * 0.5 // densityRadius: half of interaction radius
		paramData[14] = eff.crowdLimit // densityThreshold
		paramData[15] = (eff.spread / 100) * 4.0 // densityRepulsion: 0–100% → 0–4 strength
		paramData[16] = eff.repelStrength // repelStrength
		paramData[17] = repelRadius // repelRadius
		// Soft speed cap: exponential mapping so 100% ≈ unlimited, 1% ≈ very slow
		paramData[18] = 2 * Math.pow(500, this.maxSpeedPct / 100)
		device.queue.writeBuffer(this.simParamsBuffer!, 0, paramData)

		// Upload force matrix if changed
		if (this.forceMatrixDirty) {
			this.uploadForceMatrix()
			this.forceMatrixDirty = false
			this.predictionDirty = true
		}

		// Upload particle data if changed via UI
		if (this.particleBufferDirty) {
			this.uploadParticleData()
			this.particleBufferDirty = false
			this.predictionDirty = true
		}

		// Recompute organism predictions when force matrix or particle counts change
		if (this.predictionDirty) {
			this.predictionDirty = false
			const types = this.getTypeIds()
			const counts: Record<string, number> = {}
			for (const p of this.particles) {
				counts[p.groupId] = (counts[p.groupId] ?? 0) + 1
			}
			this.organismPrediction = predictOrganisms(
				this.forceMatrix,
				types,
				counts,
			)

			// Auto-balance: derive physics params from force matrix + density
			if (this.autoBalanceEnabled) {
				const result = autoBalance({
					forceMatrix: this.forceMatrix,
					typeKeys: types,
					typeCounts: counts,
					totalParticles: this.count,
					worldWidth: this.width,
					worldHeight: this.height,
				})
				this.affectRadius = result.affectRadius
				this.forceRepelDistance = result.forceRepelDistance
				this.baseStrength = result.baseStrength
				this.repelStrength = result.repelStrength
				this.crowdLimit = result.crowdLimit
				this.spread = result.spread
				this.syncAutoBalanceSliders()
			}
		}

		// Dispatch compute shader + peak stress update
		const encoder = device.createCommandEncoder()
		const pass = encoder.beginComputePass()
		pass.setPipeline(this.computePipeline!)
		pass.setBindGroup(0, this.computeBindGroups[this.pingPong]!)
		pass.dispatchWorkgroups(Math.ceil(this.count / WORKGROUP_SIZE))
		// Single-invocation pass to decay running peak and reset frame max
		pass.setPipeline(this.peakUpdatePipeline!)
		pass.dispatchWorkgroups(1)
		pass.end()
		device.queue.submit([encoder.finish()])

		// Swap ping-pong
		this.pingPong = 1 - this.pingPong

		// Bar-boundary music scheduling (§3.1)
		if (this.audioGraph.isEnabled) {
			const barDur = barDuration(
				this.musicBpm,
				this.musicBeatsPerBar,
				this.musicTimeMultiplier,
			)
			const now = this.audioGraph.currentTime
			const newBar = checkBarBoundary(
				this.tSoundStart,
				this.musicBarNumber,
				now,
				barDur,
			)

			if (newBar !== null && this.detectionState && this.latestGlobalMetrics) {
				this.musicBarNumber = newBar
				const barStart = barStartTimeFn(this.tSoundStart, newBar, barDur)

				// Build snapshot from detection state
				const snapshot = this.buildBarSnapshot()

				const config: ScheduleConfig = {
					barsPerPhase: this.overtonePhaseRate,
					qualificationFraction: this.qualificationFraction,
					preferNiceModes: this.preferNiceModes,
					beatsPerBar: this.musicBeatsPerBar,
				}

				const scheduled = scheduleBar(
					snapshot,
					newBar,
					barStart,
					barDur,
					this.musicState,
					config,
				)

				// Apply voice budget — cull excess melody voices
				const culled = applyVoiceBudget(scheduled, this.voiceBudget)

				// Play the bar and trigger visual pulses
				const hitTimings = this.audioGraph.playScheduledBar(culled)
				this.triggerVisualPulses(culled, hitTimings)

				// Update bass layer — schedule bass plucks for this bar
				const phraseSeq = expandPhrase(this.phrasePattern, this.phraseMirror)
				this.bassLayer.applyUpdate(
					scheduled.bassUpdate,
					barStart,
					barDur,
					this.musicBeatsPerBar,
					scheduled.barNumber,
					phraseSeq,
				)

				// Auto-randomize one bar before phrase cycle restarts, so the
				// transition bar resolves before the bass sequence begins anew.
				const { idx: phraseIdx, len: phraseLen } = this.phrasePosition(newBar)
				if (phraseLen > 0 && phraseIdx === phraseLen - 2) {
					if (this.autoRandomizeMatrixEnabled) {
						const arTypes = this.getTypeIds()
						this.forceMatrix = randomizeMatrix(arTypes)
						this.forceMatrixDirty = true
						this.speciesPresence.clear()
						this.speciesBrightness.clear()
						if (this._matrixWrapper)
							this.syncMatrixUI(this._matrixWrapper, arTypes)
						if (this._matrixContainer)
							this.syncMatrixHidden(this._matrixContainer)
						if (this._matrixRootContainer) {
							this._matrixRootContainer.dispatchEvent(
								new Event("change", { bubbles: true }),
							)
						}
					}
					if (this.autoRandomizeCountsEnabled) {
						this.randomizeCounts()
					}
				}

				// Update music state for next bar
				this.musicState = {
					currentBarNumber: newBar,
					currentMode: scheduled.mode,
					currentRootMidi: scheduled.rootMidi,
					netStability: scheduled.netStability,
					prevScheduledBar: scheduled,
					isBufferBar: scheduled.isBufferBar,
					bufferPitchClasses: scheduled.bufferPitchClasses,
					envelopeRanges: scheduled.envelopeRanges,
				}
				this.currentScheduledBar = culled
			}

			// Continuous bass layer volume updates (not bar-quantized, §9.4)
			if (this.latestGlobalMetrics) {
				this.bassLayer.updateFreeParticleVolumes(
					this.latestGlobalMetrics.freeParticlePercentByType,
				)
			}
		}

		// Bar visualizer (runs every frame for smooth scrubber — only when sound is on)
		if (this.audioGraph.isEnabled) {
			const bvBarDur = barDuration(
				this.musicBpm,
				this.musicBeatsPerBar,
				this.musicTimeMultiplier,
			)
			const bvNow = this.audioGraph.currentTime
			const bvBarStart = barStartTimeFn(
				this.tSoundStart,
				this.musicBarNumber,
				bvBarDur,
			)
			const phraseSeqExpanded = expandPhrase(
				this.phrasePattern,
				this.phraseMirror,
			)
			const seqLen = phraseSeqExpanded.length
			const cycleOrigin = this.bassLayer.cycleOrigin
			const phraseBarInCycle =
				cycleOrigin != null && seqLen > 0
					? (((this.musicBarNumber - cycleOrigin) % seqLen) + seqLen) % seqLen
					: -1
			updateBarVisualizer({
				scheduledBar: this.currentScheduledBar,
				barStartTime: bvBarStart,
				barDuration: bvBarDur,
				beatsPerBar: this.musicBeatsPerBar,
				now: bvNow,
				groupColors: this.groupColors,
				typeKeys: this.getTypeIds(),
				phraseBarInCycle,
				phraseSequenceLength: seqLen,
			})
		}

		// Upload radius scales after rhythm tick so new pulses render same-frame
		this.uploadRadiusScales()

		// Update skeuomorphic widget clocks
		if (
			this._phaseClock &&
			this.musicBarNumber >= 0 &&
			this.overtonePhaseRate > 0
		) {
			this._phaseClock.value =
				(this.musicBarNumber % this.overtonePhaseRate) / this.overtonePhaseRate
		}
		if ((this._latchClock || this.lpfCurveEditor) && this.audioGraph.isEnabled) {
			const barDur = barDuration(
				this.musicBpm,
				this.musicBeatsPerBar,
				this.musicTimeMultiplier,
			)
			const now = this.audioGraph.currentTime
			const barStart = barStartTimeFn(
				this.tSoundStart,
				this.musicBarNumber,
				barDur,
			)
			if (this._latchClock) {
				const beatDur = barDur / this.musicBeatsPerBar
				const latchDur = beatDur * this.detectionConfig.organelleLatchBeats
				if (latchDur > 0) {
					const elapsed = now - barStart
					this._latchClock.value = (elapsed % latchDur) / latchDur
				}
			}
			// Update master bus LPF cutoff from curve position in bar
			if (this.lpfCurveEditor && barDur > 0) {
				const elapsed = now - barStart
				const barFraction = Math.max(0, Math.min(1, elapsed / barDur))
				this.audioGraph.updateLpfFromBarPosition(barFraction)
			}
		}

		if (this._autoRandomizeMatrixClock || this._autoRandomizeCountsClock) {
			const { idx, len } = this.phrasePosition(this.musicBarNumber)
			const progress =
				this.audioGraph.isEnabled && len > 0 && idx >= 0 ? idx / len : 0
			if (this._autoRandomizeMatrixClock) {
				this._autoRandomizeMatrixClock.value = this.autoRandomizeMatrixEnabled
					? progress
					: 0
			}
			if (this._autoRandomizeCountsClock) {
				this._autoRandomizeCountsClock.value = this.autoRandomizeCountsEnabled
					? progress
					: 0
			}
		}

		// Readback particle data every ~6 frames (~10Hz at 60fps)
		// Used for detection pipeline and audio spatial metrics
		this.frameCounter++
		if (!this.readbackPending && this.frameCounter % 6 === 0) {
			this.readbackPending = true
			const outputBuf = this.particleBuffers[this.pingPong]!
			const readback = this.readbackBuffer!
			const copySize = this.count * PARTICLE_STRIDE

			const copyEncoder = device.createCommandEncoder()
			copyEncoder.copyBufferToBuffer(outputBuf, 0, readback, 0, copySize)
			device.queue.submit([copyEncoder.finish()])

			const n = this.count
			const types = this.getTypeIds()
			const numTypes = types.length
			const w = this.width
			const h = this.height
			const eff = this.getEffectiveParams()
			const cellSize = Math.max(
				1,
				eff.affectRadius + eff.forceRepelDistance * 0.5,
			)

			readback
				.mapAsync(GPUMapMode.READ, 0, copySize)
				.then(() => {
					const f32 = new Float32Array(readback.getMappedRange(0, copySize))
					const stride = PARTICLE_STRIDE / 4

					// Grid dimensions for spatial hash
					const cols = Math.max(1, Math.ceil(w / cellSize))
					const rows = Math.max(1, Math.ceil(h / cellSize))
					const gridSize = cols * rows

					// Per-particle data for spatial hash and detection
					const particleCells = new Uint32Array(n) // cell index
					const particleTypes = new Uint32Array(n) // type index
					const posX = new Float32Array(n)
					const posY = new Float32Array(n)
					const velX = new Float32Array(n)
					const velY = new Float32Array(n)

					// Per-particle acceleration (current vel - previous vel)
					const particleAccelX = new Float32Array(n)
					const particleAccelY = new Float32Array(n)
					const prevVX = this.prevVelX
					const prevVY = this.prevVelY
					const hasPrev =
						prevVX !== null && prevVY !== null && prevVX.length === n

					// Allocate next-frame velocity storage
					const nextVelX = new Float32Array(n)
					const nextVelY = new Float32Array(n)

					// Cell → particle index lists (for distance-based neighbor queries)
					const cellHeads = new Int32Array(gridSize).fill(-1)
					const cellNext = new Int32Array(n).fill(-1)

					// Pass 1: bin particles and accumulate basic stats
					for (let i = 0; i < n; i++) {
						const base = i * stride
						const px = f32[base + 0]
						const py = f32[base + 1]
						const vx = f32[base + 2]
						const vy = f32[base + 3]
						const typeIdx = new Uint32Array(
							f32.buffer,
							f32.byteOffset + (base + 8) * 4,
							1,
						)[0]
						const ti = typeIdx < numTypes ? typeIdx : 0

						// Store position and velocity for detection pipeline
						posX[i] = px
						posY[i] = py
						velX[i] = vx
						velY[i] = vy

						// Store current velocity for next frame's acceleration calc
						nextVelX[i] = vx
						nextVelY[i] = vy

						// Compute per-particle acceleration
						const ax = hasPrev ? vx - prevVX![i] : 0
						const ay = hasPrev ? vy - prevVY![i] : 0
						particleAccelX[i] = ax
						particleAccelY[i] = ay

						const col = Math.min(
							cols - 1,
							Math.max(0, Math.floor(px / cellSize)),
						)
						const row = Math.min(
							rows - 1,
							Math.max(0, Math.floor(py / cellSize)),
						)
						const cellIdx = row * cols + col
						particleCells[i] = cellIdx
						particleTypes[i] = ti
						// Linked-list insertion (prepend)
						cellNext[i] = cellHeads[cellIdx]
						cellHeads[cellIdx] = i
					}

					// Save velocities for next frame
					this.prevVelX = nextVelX
					this.prevVelY = nextVelY

					// Run detection pipeline
					const now = performance.now() / 1000
					const dt =
						this.lastReadbackTime > 0 ? now - this.lastReadbackTime : 0.1
					this.lastReadbackTime = now

					const readbackData: ReadbackData = {
						n,
						posX,
						posY,
						velX,
						velY,
						particleTypes,
						particleCells,
						cellHeads,
						cellNext,
						cols,
						rows,
						cellSize,
						width: w,
						height: h,
					}
					const scaledDetConfig: DetectionConfig = {
						...this.detectionConfig,
						proximityRadius: this.detectionConfig.proximityRadius * this.scale,
						organismProximityRadius:
							this.detectionConfig.organismProximityRadius * this.scale,
					}
					this.detectionState = runDetection(
						readbackData,
						this.detectionState,
						scaledDetConfig,
						dt,
						this.musicBpm,
						this.forceMatrix,
						types,
					)

					// Compute global metrics for music pipeline
					const typeCounts = new Map<number, number>()
					for (let i = 0; i < n; i++) {
						const ti = particleTypes[i]
						typeCounts.set(ti, (typeCounts.get(ti) ?? 0) + 1)
					}
					this.latestGlobalMetrics = computeGlobalMetrics(
						readbackData,
						this.detectionState,
						typeCounts,
						this.detectionState,
						this.organismPrediction,
						this.getTypeIds(),
					)
					if (
						this.showOrganelleOverlay ||
						this.showOrganismOverlay ||
						this.showOrganismCentroids
					) {
						this.uploadDetectionIds(this.detectionState, n)
					}
					if (this.showOrganismOverlay || this.showOrganismCentroids) {
						this.uploadOrganismCentroids(this.detectionState)
						this.uploadOsmLevelCentroids(this.detectionState)
					}
					this.updateLedgerUI()

					// Update organism registry for stable identity tracking
					if (this.detectionState) {
						const organelleMap = new Map<number, OrganelleState>()
						for (const org of this.detectionState.organelles) {
							organelleMap.set(org.id, org)
						}
						this.organismRegistry = updateRegistry(
							this.organismRegistry,
							this.detectionState,
							organelleMap,
							dt,
							80,
							this.audioGraph.currentTime,
						)
					}

					readback.unmap()
					this.readbackPending = false
				})
				.catch(() => {
					this.readbackPending = false
				})
		}
	}

	/* ================================================================ */
	/*  draw                                                             */
	/* ================================================================ */

	draw(gpu: GpuContext) {
		const device = this.device!

		const encoder = device.createCommandEncoder()

		// The "current" buffer (output of last compute) is the one pingPong now points to
		// (we swapped after dispatch, so pingPong indexes the freshly-written buffer)
		const readIndex = this.pingPong

		// --- Pass 1: Render particles additively to offscreen RGBA16F ---
		{
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: this.offscreenView!,
						clearValue: { r: 0, g: 0, b: 0, a: 0 },
						loadOp: "clear",
						storeOp: "store",
					},
				],
			})
			pass.setPipeline(this.particleRenderPipeline!)
			pass.setBindGroup(0, this.particleRenderBindGroups[readIndex]!)
			pass.draw(6, this.count)
			pass.end()
		}

		// --- Stain update pass (only when stain effect is active) ---
		if (this.activePostEffect.id === "stain" && this.stainPipeline && this.stainBindGroups[this.stainPingPong]) {
			const postParams = this.getActivePostParams()
			this.device!.queue.writeBuffer(
				this.stainParamsBuffer!,
				0,
				new Float32Array([0, 0, 0, postParams[2] ?? 0.012]),
			)

			const writeIdx = 1 - this.stainPingPong
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: this.stainViews[writeIdx]!,
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: "clear",
						storeOp: "store",
					},
				],
			})
			pass.setPipeline(this.stainPipeline)
			pass.setBindGroup(0, this.stainBindGroups[this.stainPingPong]!)
			pass.draw(4)
			pass.end()
			this.stainPingPong = writeIdx
		}

		// --- Pass 2: Fullscreen quad — post-process ---
		const canvasView = gpu.canvasContext.getCurrentTexture().createView()
		{
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: canvasView,
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: "clear",
						storeOp: "store",
					},
				],
			})
			pass.setPipeline(this.quadPipeline!)
			const quadIdx = this.activePostEffect.id === "stain" ? (1 - this.stainPingPong) : 0
			pass.setBindGroup(0, this.quadBindGroups[quadIdx]!)
			pass.draw(4)
			pass.end()
		}

		// --- Pass 3: Circle outline overlay (optional) ---
		if (this.showCircleOverlay && !this.showOrganelleOverlay) {
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: canvasView,
						loadOp: "load",
						storeOp: "store",
					},
				],
			})
			pass.setPipeline(this.circleRenderPipeline!)
			pass.setBindGroup(0, this.circleRenderBindGroups[readIndex]!)
			pass.draw(6, this.count)
			pass.end()
		}

		// --- Pass 3b: Organism overlays ---
		if (this.showOrganismOverlay || this.showOrganismCentroids) {
			this.extrapolateOrganismCentroids()
			this.extrapolateOsmLevelCentroids()

			if (this.showOrganismOverlay) {
				const osmIdView = this.organismIdTexture!.createView()
				const jfaSeedView = this.jfaOrganismTextures[0]!.createView()

				// Original fill: organism IDs into r8uint (particle-hugging outlines)
				const osmDepthView = this.organismDepthTexture!.createView()

				const osmFillPass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: osmIdView,
							loadOp: "clear",
							storeOp: "store",
							clearValue: [0, 0, 0, 0],
						},
					],
					depthStencilAttachment: {
						view: osmDepthView,
						depthLoadOp: "clear",
						depthStoreOp: "store",
						depthClearValue: 1.0,
					},
				})
				osmFillPass.setPipeline(this.organismFillPipeline!)
				osmFillPass.setBindGroup(0, this.organismFillBindGroups[readIndex]!)
				osmFillPass.draw(6, this.count)
				osmFillPass.end()

				// Centroid circle fill → organism ID texture (only when centroids visible)
				if (
					this.showOrganismCentroids &&
					this.organismCentroidCount > 0 &&
					this.organismCentroidFillPipeline
				) {
					const circFillPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: osmIdView,
								loadOp: "load",
								storeOp: "store",
							},
						],
						depthStencilAttachment: {
							view: osmDepthView,
							depthLoadOp: "load",
							depthStoreOp: "store",
						},
					})
					circFillPass.setPipeline(this.organismCentroidFillPipeline)
					circFillPass.setBindGroup(0, this.organismCentroidBindGroup!)
					circFillPass.draw(6, this.organismCentroidCount)
					circFillPass.end()
				}

				// Line fill → organism ID texture (only when centroids visible)
				if (
					this.showOrganismCentroids &&
					this.organismLineCount > 0 &&
					this.organismLineFillPipeline &&
					this.organismLineBindGroup
				) {
					const lineFillPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: osmIdView,
								loadOp: "load",
								storeOp: "store",
							},
						],
						depthStencilAttachment: {
							view: osmDepthView,
							depthLoadOp: "load",
							depthStoreOp: "store",
						},
					})
					lineFillPass.setPipeline(this.organismLineFillPipeline)
					lineFillPass.setBindGroup(0, this.organismLineBindGroup)
					lineFillPass.draw(6, this.organismLineCount)
					lineFillPass.end()
				}

				// JFA seed pass (in parallel with the ID fill — same data, different format)
				const osmSeedPass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: jfaSeedView,
							loadOp: "clear",
							storeOp: "store",
							clearValue: [4294967295, 0, 0, 0],
						},
					],
					depthStencilAttachment: {
						view: osmDepthView,
						depthLoadOp: "clear",
						depthStoreOp: "store",
						depthClearValue: 1.0,
					},
				})
				osmSeedPass.setPipeline(this.jfaOrganismSeedPipeline!)
				osmSeedPass.setBindGroup(0, this.organismFillBindGroups[readIndex]!)
				osmSeedPass.draw(6, this.count)
				osmSeedPass.end()

				// JFA seed: centroid circles (only when centroids visible)
				if (
					this.showOrganismCentroids &&
					this.organismCentroidCount > 0 &&
					this.jfaOrganismCentroidSeedPipeline
				) {
					const circSeedPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: jfaSeedView,
								loadOp: "load",
								storeOp: "store",
							},
						],
						depthStencilAttachment: {
							view: osmDepthView,
							depthLoadOp: "load",
							depthStoreOp: "store",
						},
					})
					circSeedPass.setPipeline(this.jfaOrganismCentroidSeedPipeline)
					circSeedPass.setBindGroup(0, this.organismCentroidBindGroup!)
					circSeedPass.draw(6, this.organismCentroidCount)
					circSeedPass.end()
				}

				// JFA seed: connection lines (only when centroids visible)
				if (
					this.showOrganismCentroids &&
					this.organismLineCount > 0 &&
					this.jfaOrganismLineSeedPipeline &&
					this.organismLineBindGroup
				) {
					const lineSeedPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: jfaSeedView,
								loadOp: "load",
								storeOp: "store",
							},
						],
						depthStencilAttachment: {
							view: osmDepthView,
							depthLoadOp: "load",
							depthStoreOp: "store",
						},
					})
					lineSeedPass.setPipeline(this.jfaOrganismLineSeedPipeline)
					lineSeedPass.setBindGroup(0, this.organismLineBindGroup)
					lineSeedPass.draw(6, this.organismLineCount)
					lineSeedPass.end()
				}

				// JFA flood passes
				let osmReadIdx = 0
				for (let i = 0; i < this.jfaPassCount; i++) {
					const cp = encoder.beginComputePass()
					cp.setPipeline(this.jfaComputePipeline!)
					cp.setBindGroup(0, this.jfaOrganismBindGroups[osmReadIdx][i]!)
					cp.dispatchWorkgroups(
						Math.ceil(this.width / 8),
						Math.ceil(this.height / 8),
					)
					cp.end()
					osmReadIdx = 1 - osmReadIdx
				}

				// Edge detect: ID texture (outer boundary) + JFA (Voronoi inter-group)
				const osmEdgePass = encoder.beginRenderPass({
					colorAttachments: [
						{
							view: canvasView,
							loadOp: "load",
							storeOp: "store",
						},
					],
				})
				osmEdgePass.setPipeline(this.jfaOrganismEdgePipeline!)
				osmEdgePass.setBindGroup(0, this.jfaOrganismEdgeBindGroups[osmReadIdx]!)
				osmEdgePass.draw(3)
				osmEdgePass.end()
			}

			// Visual: centroids + connector lines onto canvas
			if (this.showOrganismCentroids) {
				// Connection lines
				if (
					this.organismLineCount > 0 &&
					this.organismLinePipeline &&
					this.organismLineBindGroup
				) {
					const linePass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: canvasView,
								loadOp: "load",
								storeOp: "store",
							},
						],
					})
					linePass.setPipeline(this.organismLinePipeline)
					linePass.setBindGroup(0, this.organismLineBindGroup)
					linePass.draw(6, this.organismLineCount)
					linePass.end()
				}

				// Organelle centroid circles
				if (this.organismCentroidCount > 0) {
					const centroidPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: canvasView,
								loadOp: "load",
								storeOp: "store",
							},
						],
					})
					centroidPass.setPipeline(this.organismCentroidPipeline!)
					centroidPass.setBindGroup(0, this.organismCentroidBindGroup!)
					centroidPass.draw(6, this.organismCentroidCount)
					centroidPass.end()
				}

				// Organism-level centroid circles (larger, wrapping organelle circles)
				if (
					this.osmLevelCentroidCount > 0 &&
					this.osmLevelCentroidBindGroup &&
					this.osmLevelCentroidPipeline
				) {
					const osmCentroidPass = encoder.beginRenderPass({
						colorAttachments: [
							{
								view: canvasView,
								loadOp: "load",
								storeOp: "store",
							},
						],
					})
					osmCentroidPass.setPipeline(this.osmLevelCentroidPipeline)
					osmCentroidPass.setBindGroup(0, this.osmLevelCentroidBindGroup!)
					osmCentroidPass.draw(6, this.osmLevelCentroidCount)
					osmCentroidPass.end()
				}
			}
		}

		// --- Pass 3c: Organelle overlay — colored outlines + Voronoi inter-group ---
		if (this.showOrganelleOverlay) {
			const jfaSeedView = this.jfaOrganelleTextures[0]!.createView()

			// Original fill: organelle IDs + colors (particle-hugging shape)
			const fillPass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: this.detectionIdTexture!.createView(),
						loadOp: "clear",
						storeOp: "store",
						clearValue: [0, 0, 0, 0],
					},
					{
						view: this.detectionColorTexture!.createView(),
						loadOp: "clear",
						storeOp: "store",
						clearValue: { r: 0, g: 0, b: 0, a: 0 },
					},
				],
			})
			fillPass.setPipeline(this.detectionFillPipeline!)
			fillPass.setBindGroup(0, this.detectionFillBindGroups[readIndex]!)
			fillPass.draw(6, this.count)
			fillPass.end()

			// JFA seed pass (packed coords + organelleId + color)
			const seedPass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: jfaSeedView,
						loadOp: "clear",
						storeOp: "store",
						clearValue: [4294967295, 0, 0, 0],
					},
					{
						// Also render colors (re-use same color texture — already filled above,
						// but JFA seed uses same particles so result is identical)
						view: this.detectionColorTexture!.createView(),
						loadOp: "load",
						storeOp: "store",
					},
				],
			})
			seedPass.setPipeline(this.jfaOrganelleSeedPipeline!)
			seedPass.setBindGroup(0, this.detectionFillBindGroups[readIndex]!)
			seedPass.draw(6, this.count)
			seedPass.end()

			// JFA flood passes
			let orgReadIdx = 0
			for (let i = 0; i < this.jfaPassCount; i++) {
				const cp = encoder.beginComputePass()
				cp.setPipeline(this.jfaComputePipeline!)
				cp.setBindGroup(0, this.jfaOrganelleBindGroups[orgReadIdx][i]!)
				cp.dispatchWorkgroups(
					Math.ceil(this.width / 8),
					Math.ceil(this.height / 8),
				)
				cp.end()
				orgReadIdx = 1 - orgReadIdx
			}

			// Edge detect: ID texture (outer) + JFA (Voronoi inter-group)
			const edgePass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: canvasView,
						loadOp: "load",
						storeOp: "store",
					},
				],
			})
			edgePass.setPipeline(this.jfaOrganelleEdgePipeline!)
			edgePass.setBindGroup(0, this.jfaOrganelleEdgeBindGroups[orgReadIdx]!)
			edgePass.draw(3)
			edgePass.end()
		}

		device.queue.submit([encoder.finish()])
	}

	/* ================================================================ */
	/*  Shader switching (called by shader menu)                         */
	/* ================================================================ */

	switchParticleShader(effectId: string) {
		const effect = findParticleEffect(effectId)
		if (effect.id === this.activeParticleEffect.id) return
		this.activeParticleEffect = effect
		if (this.device) {
			this.rebuildParticleRenderPipeline()
		}
		// Sync hidden input for settings persistence
		if (
			this._hiddenParticleEffect &&
			this._hiddenParticleEffect.value !== effectId
		) {
			this._hiddenParticleEffect.value = effectId
			this._hiddenParticleEffect.dispatchEvent(
				new Event("input", { bubbles: true }),
			)
		}
	}

	switchPostShader(effectId: string) {
		const effect = findPostEffect(effectId)
		if (effect.id === this.activePostEffect.id) return
		this.activePostEffect = effect
		if (this.device) {
			this.rebuildQuadPipeline()
			// Clear stain textures when switching to stain so old trails don't linger
			if (effect.id === "stain") {
				this.clearStainTextures()
			}
		}
		// Sync hidden input for settings persistence
		if (this._hiddenPostEffect && this._hiddenPostEffect.value !== effectId) {
			this._hiddenPostEffect.value = effectId
			this._hiddenPostEffect.dispatchEvent(
				new Event("input", { bubbles: true }),
			)
		}
	}

	getActiveParticleEffectId(): string {
		return this.activeParticleEffect.id
	}
	getActivePostEffectId(): string {
		return this.activePostEffect.id
	}

	/** Returns param definitions for the active particle shader */
	getParticleShaderParams(): import("../shader-menu").ShaderParamDef[] {
		const effect = this.activeParticleEffect
		const vals = this.getActiveParticleParams()
		return (effect.params ?? []).map((p) => ({
			label: p.label,
			setting: `fx:p:${effect.id}:${p.slot}`,
			value: vals[p.slot] ?? p.default,
			min: p.min,
			max: p.max,
			step: p.step,
			onChange: (v: number) => {
				const arr =
					this.particleEffectParams[effect.id] ?? effectDefaults(effect)
				arr[p.slot] = v
				this.particleEffectParams[effect.id] = arr
				this.uploadRenderParams()
				this.syncParamHiddenInputs()
			},
		}))
	}

	/** Returns param definitions for the active post-process shader */
	getPostShaderParams(): import("../shader-menu").ShaderParamDef[] {
		const effect = this.activePostEffect
		const vals = this.getActivePostParams()
		return (effect.params ?? []).map((p) => ({
			label: p.label,
			setting: `fx:q:${effect.id}:${p.slot}`,
			value: vals[p.slot] ?? p.default,
			min: p.min,
			max: p.max,
			step: p.step,
			onChange: (v: number) => {
				const arr = this.postEffectParams[effect.id] ?? effectDefaults(effect)
				arr[p.slot] = v
				this.postEffectParams[effect.id] = arr
				this.uploadQuadParams()
				this.syncParamHiddenInputs()
			},
		}))
	}

	/* ================================================================ */
	/*  GPU resource creation helpers                                    */
	/* ================================================================ */

	private createBuffers() {
		const device = this.device!
		const bufSize = MAX_PARTICLES * PARTICLE_STRIDE

		this.particleBuffers[0] = device.createBuffer({
			size: bufSize,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC,
		})
		this.particleBuffers[1] = device.createBuffer({
			size: bufSize,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC,
		})

		// Small staging buffer for swap-and-shrink particle removal
		this.particleStagingBuffer = device.createBuffer({
			size: PARTICLE_STRIDE,
			usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		})

		this.simParamsBuffer = device.createBuffer({
			size: 80, // SimParams: 20 x f32
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.forceMatrixBuffer = device.createBuffer({
			size: MAX_TYPES * MAX_TYPES * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		// Peak stats: [stressFrameMax, stressPeak, speedFrameMax, speedPeak]
		this.stressBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})
		// Initialize peaks to 1.0 (avoids div-by-zero before first frame)
		const initPeaks = new Float32Array([0, 1.0, 0, 1.0])
		device.queue.writeBuffer(this.stressBuffer, 0, initPeaks)

		this.renderParamsBuffer = device.createBuffer({
			size: 32, // RenderParams: vec2 + f32 + u32 + f32 (padded to 32)
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.circleRenderParamsBuffer = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.detectionFillParamsBuffer = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.organismFillParamsBuffer = device.createBuffer({
			size: 32,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		// Bubble params for JFA edge detection (threshold, edgeWidth, pad, pad)
		this.bubbleParamsBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})
		this.uploadBubbleParams()

		// Organelle centroid buffer: up to 256 organelles, each 16 bytes (vec2f pos, f32 radius, u32 id)
		this.organismCentroidBuffer = device.createBuffer({
			size: 256 * 16,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		// Organism-level centroid buffer: up to 128 organisms, each 16 bytes
		this.osmLevelCentroidBuffer = device.createBuffer({
			size: 128 * 16,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		this.organismCentroidParamsBuffer = device.createBuffer({
			size: 16, // vec2f resolution + padding
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		// Organism connection lines: up to 1024 segments, each 32 bytes (2 × vec2f + u32 osmId + padding)
		this.organismLineBuffer = device.createBuffer({
			size: 1024 * 32,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		this.quadParamsBuffer = device.createBuffer({
			size: 32, // QuadParams: 5 x f32 (padded to 32)
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		// Readback buffer for audio velocity sampling
		this.readbackBuffer = device.createBuffer({
			size: bufSize,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		})

		// Detection buffer: 1 u32 per particle (packed organelleId + organismId)
		this.detectionBuffer = device.createBuffer({
			size: MAX_PARTICLES * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		// Radius scale buffer: 1 f32 per particle (multiplier for point size, default 1.0)
		this.radiusScaleBuffer = device.createBuffer({
			size: MAX_PARTICLES * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})
		const initScales = new Float32Array(MAX_PARTICLES)
		initScales.fill(1.0)
		device.queue.writeBuffer(this.radiusScaleBuffer, 0, initScales)

		// Falloff LUT texture (256x1, r8unorm)
		this.falloffTexture = device.createTexture({
			size: [256, 1],
			format: "r8unorm",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		})

		this.falloffSampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
			addressModeU: "clamp-to-edge",
			addressModeV: "clamp-to-edge",
		})

		// Upload default linear LUT (1→0: bright at center, dark at edge)
		const defaultLUT = new Uint8Array(256)
		for (let i = 0; i < 256; i++) defaultLUT[i] = 255 - i
		device.queue.writeTexture(
			{ texture: this.falloffTexture },
			defaultLUT,
			{ bytesPerRow: 256 },
			{ width: 256, height: 1 },
		)

		this.offscreenSampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		})
	}

	private createComputePipeline() {
		const device = this.device!

		this.computeBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" },
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" },
				},
				{
					binding: 3,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 4,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "storage" },
				},
			],
		})

		const computeLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.computeBindGroupLayout],
		})
		const computeModule = device.createShaderModule({ code: computeShaderSrc })

		this.computePipeline = device.createComputePipeline({
			layout: computeLayout,
			compute: { module: computeModule, entryPoint: "main" },
		})

		this.peakUpdatePipeline = device.createComputePipeline({
			layout: computeLayout,
			compute: { module: computeModule, entryPoint: "updatePeaks" },
		})

		// JFA flood compute pipeline
		this.jfaComputeBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.COMPUTE,
					texture: { sampleType: "uint" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: "rg32uint",
					},
				},
				{
					binding: 2,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: "uniform" },
				},
			],
		})
		const jfaLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.jfaComputeBindGroupLayout],
		})
		const jfaModule = device.createShaderModule({ code: jfaComputeSrc })
		this.jfaComputePipeline = device.createComputePipeline({
			layout: jfaLayout,
			compute: { module: jfaModule, entryPoint: "main" },
		})
	}

	private createRenderPipelines() {
		const device = this.device!

		this.particleRenderBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: "uniform" },
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "float" },
				},
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
				{
					binding: 4,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 5,
					visibility: GPUShaderStage.VERTEX,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 6,
					visibility: GPUShaderStage.VERTEX,
					buffer: { type: "read-only-storage" },
				},
			],
		})

		this.rebuildParticleRenderPipeline()
		this.rebuildCircleOverlayPipeline()
	}

	private rebuildParticleRenderPipeline() {
		const device = this.device!
		const shaderSrc = buildParticleShader(this.activeParticleEffect)
		const particleModule = device.createShaderModule({ code: shaderSrc })

		const particlePipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.particleRenderBindGroupLayout!],
		})

		// Pass 1 pipeline: additive blending, render to rgba16float
		this.particleRenderPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: particleModule, entryPoint: "vs_main" },
			fragment: {
				module: particleModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: "rgba16float",
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		this.rebuildParticleRenderBindGroups()
	}

	private rebuildCircleOverlayPipeline() {
		const device = this.device!
		// Circle overlay always uses the "solid" effect
		const solidEffect = findParticleEffect("solid")
		const shaderSrc = buildParticleShader(solidEffect)
		const circleModule = device.createShaderModule({ code: shaderSrc })

		const particlePipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.particleRenderBindGroupLayout!],
		})

		// Pass 3 pipeline: circle overlay, alpha blending, render to canvas format
		this.circleRenderPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: circleModule, entryPoint: "vs_main" },
			fragment: {
				module: circleModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// Detection fill pipeline: writes organelle ID to R8Uint + color to RGBA8
		const fillShaderCode =
			PARTICLE_PREFIX +
			`
struct FillOutput {
  @location(0) id: u32,
  @location(1) color: vec4<f32>,
};
@fragment
fn fs_main(in: VertexOutput) -> FillOutput {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.x < 0.5) { discard; }
  var out: FillOutput;
  out.id = u32(in.organelleId);
  out.color = vec4<f32>(in.color, 1.0);
  return out;
}`
		const fillModule = device.createShaderModule({ code: fillShaderCode })
		this.detectionFillPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: fillModule, entryPoint: "vs_main" },
			fragment: {
				module: fillModule,
				entryPoint: "fs_main",
				targets: [{ format: "r8uint" }, { format: "rgba8unorm" }],
			},
			primitive: { topology: "triangle-list" },
		})

		// Detection edge pipeline: fullscreen quad that edge-detects on the ID texture
		this.detectionEdgeBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "uint" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "float" },
				},
			],
		})
		const edgePipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.detectionEdgeBindGroupLayout],
		})

		const edgeShaderCode = /* wgsl */ `
@group(0) @binding(0) var detIdTex: texture_2d<u32>;
@group(0) @binding(1) var detColorTex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  // Fullscreen triangle (3 vertices, covers entire screen)
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));
  let center = textureLoad(detIdTex, coord, 0).r;
  let up    = textureLoad(detIdTex, coord + vec2(0, -1), 0).r;
  let down  = textureLoad(detIdTex, coord + vec2(0,  1), 0).r;
  let left  = textureLoad(detIdTex, coord + vec2(-1, 0), 0).r;
  let right = textureLoad(detIdTex, coord + vec2( 1, 0), 0).r;

  let isEdge = (up != center) || (down != center) || (left != center) || (right != center);
  if (!isEdge) { discard; }

  let color = textureLoad(detColorTex, coord, 0).rgb;
  return vec4<f32>(color, 1.0);
}`
		const edgeModule = device.createShaderModule({ code: edgeShaderCode })
		this.detectionEdgePipeline = device.createRenderPipeline({
			layout: edgePipelineLayout,
			vertex: { module: edgeModule, entryPoint: "vs_main" },
			fragment: {
				module: edgeModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// Organism fill pipeline: writes organism ID to R8Uint (inflated particles, 2.5x point size)
		// Use modified prefix with depth from organism size rank
		const osmFillPrefix = PARTICLE_PREFIX.replace(
			"out.position = vec4<f32>(clip, 0.0, 1.0);",
			`let osmDepthRank = (det >> 24u) & 0xFFu;
  out.position = vec4<f32>(clip, f32(osmDepthRank) / 255.0, 1.0);`,
		)
		const osmFillShaderCode =
			osmFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.y < 0.5) { discard; }
  return u32(in.organismId);
}`
		const osmFillModule = device.createShaderModule({ code: osmFillShaderCode })
		this.organismFillPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: osmFillModule, entryPoint: "vs_main" },
			fragment: {
				module: osmFillModule,
				entryPoint: "fs_main",
				targets: [{ format: "r8uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// Organism centroid circle pipeline: draws circles at organism centroids onto canvas
		this.organismCentroidBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: "read-only-storage" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.VERTEX,
					buffer: { type: "uniform" },
				},
			],
		})
		const centroidPipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.organismCentroidBindGroupLayout],
		})

		// Shared WGSL prefix for centroid circle shaders
		const centroidPrefix = /* wgsl */ `
struct OrganismCentroid {
  pos: vec2<f32>,
  radius: f32,
  osmId: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> centroids: array<OrganismCentroid>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let c = centroids[instanceIndex];

  var out: VertexOutput;
  out.uv = corner;
  out.osmId = c.osmId;

  let pos = c.pos + corner * c.radius;
  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  out.position = vec4<f32>(clip, 0.0, 1.0);

  return out;
}
`

		// Visual variant: white ring onto canvas
		const centroidVisualCode =
			centroidPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let ring = smoothstep(0.55, 0.65, dist) * smoothstep(1.0, 0.9, dist);
  if (ring < 0.01) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, ring);
}`

		// Thin-ring variant for organism-level centroid circles
		const centroidThinRingCode =
			centroidPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let ring = smoothstep(0.82, 0.88, dist) * smoothstep(1.0, 0.94, dist);
  if (ring < 0.01) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, ring);
}`
		const centroidThinRingModule = device.createShaderModule({
			code: centroidThinRingCode,
		})
		this.osmLevelCentroidPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidThinRingModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidThinRingModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// Inflated centroid prefix: 2.5x radius for fill/seed passes (matches particle inflation)
		// osmId encodes: bits 0-7 = organism ID, bits 8-15 = depth rank
		const centroidFillPrefix = /* wgsl */ `
struct OrganismCentroid {
  pos: vec2<f32>,
  radius: f32,
  osmId: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> centroids: array<OrganismCentroid>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let c = centroids[instanceIndex];

  var out: VertexOutput;
  out.uv = corner;
  out.osmId = c.osmId & 0xFFu;

  let depthRank = (c.osmId >> 8u) & 0xFFu;
  let pos = c.pos + corner * c.radius * 2.5;
  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  out.position = vec4<f32>(clip, f32(depthRank) / 255.0, 1.0);

  return out;
}
`

		// Fill variant: writes organism ID to r8uint (inflated radius)
		const centroidFillCode =
			centroidFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  return in.osmId;
}`
		const centroidModule = device.createShaderModule({
			code: centroidVisualCode,
		})
		const centroidFillModule = device.createShaderModule({
			code: centroidFillCode,
		})
		this.organismCentroidPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// Centroid circle fill pipeline: writes organism ID to r8uint
		this.organismCentroidFillPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidFillModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidFillModule,
				entryPoint: "fs_main",
				targets: [{ format: "r8uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// Organism connection line pipeline: draws white lines between linked organelle centroids
		// Reuses the same bind group layout as centroid circles (storage + uniform)
		// Shared WGSL prefix for line shaders
		const linePrefix = /* wgsl */ `
struct LineSegment {
  startPos: vec2<f32>,
  endPos: vec2<f32>,
  osmId: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) along: f32,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> lines: array<LineSegment>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0),
  vec2(0.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let seg = lines[instanceIndex];
  let dir = seg.endPos - seg.startPos;
  let len = length(dir);
  let tangent = dir / max(len, 0.001);
  let normal = vec2<f32>(-tangent.y, tangent.x);

  let halfWidth = 1.5; // pixels
  let pos = seg.startPos + tangent * corner.x * len + normal * corner.y * halfWidth;

  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;

  var out: VertexOutput;
  out.position = vec4<f32>(clip, 0.0, 1.0);
  out.along = corner.x;
  out.osmId = seg.osmId;
  return out;
}
`

		// Visual variant: white lines onto canvas
		const lineVisualCode =
			linePrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let fade = smoothstep(0.0, 0.05, in.along) * smoothstep(1.0, 0.95, in.along);
  return vec4<f32>(1.0, 1.0, 1.0, 0.6 * fade);
}`

		// Inflated line prefix: wider fill to create smooth outline envelope
		const lineFillPrefix = /* wgsl */ `
struct LineSegment {
  startPos: vec2<f32>,
  endPos: vec2<f32>,
  osmId: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

struct CentroidParams {
  resolution: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) along: f32,
  @location(1) @interpolate(flat) osmId: u32,
};

@group(0) @binding(0) var<storage, read> lines: array<LineSegment>;
@group(0) @binding(1) var<uniform> params: CentroidParams;

const CORNERS = array<vec2<f32>, 6>(
  vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0),
  vec2(0.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let seg = lines[instanceIndex];
  let dir = seg.endPos - seg.startPos;
  let len = length(dir);
  let tangent = dir / max(len, 0.001);
  let normal = vec2<f32>(-tangent.y, tangent.x);

  let halfWidth = 20.0; // inflated to match particle fill inflation
  // Extend endpoints by halfWidth for rounded caps
  let pos = seg.startPos - tangent * halfWidth + tangent * corner.x * (len + halfWidth * 2.0) + normal * corner.y * halfWidth;

  var clip = (pos / params.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;

  let depthRank = (seg.osmId >> 8u) & 0xFFu;
  var out: VertexOutput;
  out.position = vec4<f32>(clip, f32(depthRank) / 255.0, 1.0);
  out.along = corner.x;
  out.osmId = seg.osmId & 0xFFu;
  return out;
}
`

		// Fill variant: writes organism ID to r8uint (inflated width)
		const lineFillCode =
			lineFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  return in.osmId;
}`

		const lineModule = device.createShaderModule({ code: lineVisualCode })
		const lineFillModule = device.createShaderModule({ code: lineFillCode })
		this.organismLinePipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: lineModule, entryPoint: "vs_main" },
			fragment: {
				module: lineModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})
		this.organismLineFillPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: lineFillModule, entryPoint: "vs_main" },
			fragment: {
				module: lineFillModule,
				entryPoint: "fs_main",
				targets: [{ format: "r8uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// Organism edge pipeline: fullscreen edge detection, white outlines
		this.organismEdgeBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "uint" },
				},
			],
		})
		const osmEdgePipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.organismEdgeBindGroupLayout],
		})

		const osmEdgeShaderCode = /* wgsl */ `
@group(0) @binding(0) var osmIdTex: texture_2d<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));
  let center = textureLoad(osmIdTex, coord, 0).r;
  let up    = textureLoad(osmIdTex, coord + vec2(0, -1), 0).r;
  let down  = textureLoad(osmIdTex, coord + vec2(0,  1), 0).r;
  let left  = textureLoad(osmIdTex, coord + vec2(-1, 0), 0).r;
  let right = textureLoad(osmIdTex, coord + vec2( 1, 0), 0).r;
  let isEdge = (up != center) || (down != center) || (left != center) || (right != center);
  if (!isEdge) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}`
		const osmEdgeModule = device.createShaderModule({ code: osmEdgeShaderCode })
		this.organismEdgePipeline = device.createRenderPipeline({
			layout: osmEdgePipelineLayout,
			vertex: { module: osmEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: osmEdgeModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// --- JFA seed pipelines ---
		// Organelle JFA seed: writes packed (x,y) + organelleId to rg32uint
		const jfaOrganelleSeedCode =
			PARTICLE_PREFIX +
			`
struct SeedOutput {
  @location(0) seed: vec2<u32>,
  @location(1) color: vec4<f32>,
};
@fragment
fn fs_main(in: VertexOutput) -> SeedOutput {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.x < 0.5) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  var out: SeedOutput;
  out.seed = vec2<u32>(packed, u32(in.organelleId));
  out.color = vec4<f32>(in.color, 1.0);
  return out;
}`
		const jfaOrganelleSeedModule = device.createShaderModule({
			code: jfaOrganelleSeedCode,
		})
		this.jfaOrganelleSeedPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: jfaOrganelleSeedModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOrganelleSeedModule,
				entryPoint: "fs_main",
				targets: [{ format: "rg32uint" }, { format: "rgba8unorm" }],
			},
			primitive: { topology: "triangle-list" },
		})

		// Organism JFA seed: writes packed (x,y) + organismId to rg32uint
		const jfaOrganismSeedCode =
			osmFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.y < 0.5) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  return vec2<u32>(packed, u32(in.organismId));
}`
		const jfaOrganismSeedModule = device.createShaderModule({
			code: jfaOrganismSeedCode,
		})
		this.jfaOrganismSeedPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: jfaOrganismSeedModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOrganismSeedModule,
				entryPoint: "fs_main",
				targets: [{ format: "rg32uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// Organism centroid seed: writes packed coords + osmId to rg32uint (inflated radius)
		const centroidSeedCode =
			centroidFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  return vec2<u32>(packed, in.osmId);
}`
		const centroidSeedModule = device.createShaderModule({
			code: centroidSeedCode,
		})
		this.jfaOrganismCentroidSeedPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidSeedModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidSeedModule,
				entryPoint: "fs_main",
				targets: [{ format: "rg32uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// Organism line seed: writes packed coords + osmId to rg32uint (inflated width)
		const lineSeedCode =
			lineFillPrefix +
			`
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32> {
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  return vec2<u32>(packed, in.osmId);
}`
		const lineSeedModule = device.createShaderModule({ code: lineSeedCode })
		this.jfaOrganismLineSeedPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: lineSeedModule, entryPoint: "vs_main" },
			fragment: {
				module: lineSeedModule,
				entryPoint: "fs_main",
				targets: [{ format: "rg32uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: {
				format: "depth24plus",
				depthWriteEnabled: true,
				depthCompare: "less-equal",
			},
		})

		// --- JFA edge detection pipelines ---
		this.jfaEdgeBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "uint" },
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "float" },
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: "uniform" },
				},
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "uint" },
				},
			],
		})
		const jfaEdgePipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [this.jfaEdgeBindGroupLayout],
		})

		const jfaEdgeShaderPrefix = /* wgsl */ `
@group(0) @binding(0) var jfaTex: texture_2d<u32>;
@group(0) @binding(1) var detColorTex: texture_2d<f32>;

struct BubbleParams {
  threshold: f32,
  edgeWidth: f32,
  organelleThreshold: f32,
  _pad1: f32,
};
@group(0) @binding(2) var<uniform> bubbleParams: BubbleParams;
@group(0) @binding(3) var idTex: texture_2d<u32>;

const SENTINEL = 0xFFFFFFFFu;

fn unpackXY(packed: u32) -> vec2<f32> {
  return vec2<f32>(f32(packed >> 16u), f32(packed & 0xFFFFu));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}
`
		// Organelle JFA edge: smooth distance-based bubble + Voronoi where bubbles touch
		const jfaOrganelleEdgeCode =
			jfaEdgeShaderPrefix +
			`
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));

  let jfaCenter = textureLoad(jfaTex, coord, 0).rg;
  if (jfaCenter.x == SENTINEL) { discard; }

  let seedPos = unpackXY(jfaCenter.x);
  let dist = distance(vec2<f32>(coord), seedPos);
  let groupId = jfaCenter.y;

  // Smooth distance-based bubble edge around each organelle (half organism threshold)
  let orgThreshold = bubbleParams.organelleThreshold;
  let halfEdge = bubbleParams.edgeWidth * 0.5;
  let bubble = dist > (orgThreshold - halfEdge) && dist < (orgThreshold + halfEdge);

  // Voronoi boundary where organelle bubbles overlap
  var voronoi = false;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nc = coord + vec2<i32>(dx, dy);
      let n = textureLoad(jfaTex, nc, 0).rg;
      if (n.x != SENTINEL && n.y != groupId && dist < orgThreshold) {
        voronoi = true;
      }
    }
  }

  if (!bubble && !voronoi) { discard; }
  // Sample color from nearest seed position (current pixel may be far from any particle)
  let seedCoord = vec2<i32>(unpackXY(jfaCenter.x));
  let color = textureLoad(detColorTex, seedCoord, 0).rgb;
  return vec4<f32>(color, 1.0);
}`
		const jfaOrgEdgeModule = device.createShaderModule({
			code: jfaOrganelleEdgeCode,
		})
		this.jfaOrganelleEdgePipeline = device.createRenderPipeline({
			layout: jfaEdgePipelineLayout,
			vertex: { module: jfaOrgEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOrgEdgeModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		// Organism JFA edge: smooth distance-based bubble + Voronoi inter-group boundaries (white)
		const jfaOrganismEdgeCode =
			jfaEdgeShaderPrefix +
			`
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));
  let centerId = textureLoad(idTex, coord, 0).r;

  let jfaCenter = textureLoad(jfaTex, coord, 0).rg;
  if (jfaCenter.x == SENTINEL) { discard; }

  let seedPos = unpackXY(jfaCenter.x);
  let dist = distance(vec2<f32>(coord), seedPos);
  let groupId = jfaCenter.y;

  // Smooth distance-based bubble edge around each organism
  let halfEdge = bubbleParams.edgeWidth * 0.5;
  let bubble = dist > (bubbleParams.threshold - halfEdge) && dist < (bubbleParams.threshold + halfEdge);

  // JFA Voronoi inter-group boundary (organism vs organism)
  var voronoi = false;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nc = coord + vec2<i32>(dx, dy);
      let n = textureLoad(jfaTex, nc, 0).rg;
      let nId = textureLoad(idTex, nc, 0).r;
      if (n.x != SENTINEL && n.y != groupId && dist < bubbleParams.threshold) {
        voronoi = true;
      }
    }
  }

  if (!bubble && !voronoi) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}`
		const jfaOsmEdgeModule = device.createShaderModule({
			code: jfaOrganismEdgeCode,
		})
		this.jfaOrganismEdgePipeline = device.createRenderPipeline({
			layout: jfaEdgePipelineLayout,
			vertex: { module: jfaOsmEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOsmEdgeModule,
				entryPoint: "fs_main",
				targets: [
					{
						format: this.canvasFormat,
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
				],
			},
			primitive: { topology: "triangle-list" },
		})

		this.rebuildCircleRenderBindGroups()
	}

	private createOffscreenTexture() {
		const device = this.device!
		if (this.offscreenTexture) this.offscreenTexture.destroy()
		for (const t of this.stainTextures) t?.destroy()
		if (this.detectionIdTexture) this.detectionIdTexture.destroy()
		if (this.detectionColorTexture) this.detectionColorTexture.destroy()
		if (this.organismIdTexture) this.organismIdTexture.destroy()
		for (const t of this.jfaOrganelleTextures) t?.destroy()
		for (const t of this.jfaOrganismTextures) t?.destroy()

		this.offscreenTexture = device.createTexture({
			size: [this.width, this.height],
			format: "rgba16float",
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		})
		this.offscreenView = this.offscreenTexture.createView()

		// Stain ping-pong textures (phosphor persistence)
		for (let i = 0; i < 2; i++) {
			this.stainTextures[i] = device.createTexture({
				size: [this.width, this.height],
				format: "rgba8unorm",
				usage:
					GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
			})
			this.stainViews[i] = this.stainTextures[i]!.createView()
		}

		this.detectionIdTexture = device.createTexture({
			size: [this.width, this.height],
			format: "r8uint",
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		})

		this.detectionColorTexture = device.createTexture({
			size: [this.width, this.height],
			format: "rgba8unorm",
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		})

		this.organismIdTexture = device.createTexture({
			size: [this.width, this.height],
			format: "r8uint",
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		})

		// Depth texture for organism z-ordering (smaller organisms render on top)
		if (this.organismDepthTexture) this.organismDepthTexture.destroy()
		this.organismDepthTexture = device.createTexture({
			size: [this.width, this.height],
			format: "depth24plus",
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
		})

		// JFA ping-pong textures (rg32uint: packed seed xy + group ID)
		const jfaUsage =
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING
		for (let i = 0; i < 2; i++) {
			this.jfaOrganelleTextures[i] = device.createTexture({
				size: [this.width, this.height],
				format: "rg32uint",
				usage: jfaUsage,
			})
			this.jfaOrganismTextures[i] = device.createTexture({
				size: [this.width, this.height],
				format: "rg32uint",
				usage: jfaUsage,
			})
		}

		// Pre-compute JFA pass count and rebuild param buffers
		this.jfaPassCount = Math.ceil(Math.log2(Math.max(this.width, this.height)))
		for (const buf of this.jfaParamsBuffers) buf.destroy()
		this.jfaParamsBuffers = []
		for (let i = 0; i < this.jfaPassCount; i++) {
			const step = 1 << (this.jfaPassCount - 1 - i)
			const buf = device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			})
			device.queue.writeBuffer(
				buf,
				0,
				new Uint32Array([step, this.width, this.height, 0]),
			)
			this.jfaParamsBuffers.push(buf)
		}
	}

	private createQuadPipeline() {
		const device = this.device!

		this.quadBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "float" },
				},
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: "uniform" },
				},
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: "float" },
				},
			],
		})

		this.rebuildQuadPipeline()
	}

	private rebuildQuadPipeline() {
		const device = this.device!
		const shaderSrc = buildQuadShader(this.activePostEffect)
		const quadModule = device.createShaderModule({ code: shaderSrc })

		this.quadPipeline = device.createRenderPipeline({
			layout: device.createPipelineLayout({
				bindGroupLayouts: [this.quadBindGroupLayout!],
			}),
			vertex: { module: quadModule, entryPoint: "vs_main" },
			fragment: {
				module: quadModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat }],
			},
			primitive: { topology: "triangle-strip" },
		})

		this.rebuildQuadBindGroup()
	}

	private createStainPipeline() {
		const device = this.device!

		this.stainParamsBuffer = device.createBuffer({
			size: 16, // bgR, bgG, bgB, decayRate
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.stainBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
			],
		})

		const module = device.createShaderModule({ code: stainUpdateSrc })
		this.stainPipeline = device.createRenderPipeline({
			layout: device.createPipelineLayout({
				bindGroupLayouts: [this.stainBindGroupLayout],
			}),
			vertex: { module, entryPoint: "vs_main" },
			fragment: {
				module,
				entryPoint: "fs_main",
				targets: [{ format: "rgba8unorm" }],
			},
			primitive: { topology: "triangle-strip" },
		})

		this.rebuildStainBindGroups()
	}

	private clearStainTextures() {
		const device = this.device!
		const encoder = device.createCommandEncoder()
		for (let i = 0; i < 2; i++) {
			if (!this.stainViews[i]) continue
			const pass = encoder.beginRenderPass({
				colorAttachments: [{
					view: this.stainViews[i]!,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				}],
			})
			pass.end()
		}
		device.queue.submit([encoder.finish()])
		this.stainPingPong = 0
	}

	private rebuildStainBindGroups() {
		const device = this.device!
		if (!this.stainBindGroupLayout || !this.stainViews[0] || !this.stainViews[1]) return

		// Group 0: read stain[0] + particles → write stain[1]
		// Group 1: read stain[1] + particles → write stain[0]
		for (let i = 0; i < 2; i++) {
			this.stainBindGroups[i] = device.createBindGroup({
				layout: this.stainBindGroupLayout,
				entries: [
					{ binding: 0, resource: this.stainViews[i]! },
					{ binding: 1, resource: this.offscreenSampler! },
					{ binding: 2, resource: this.offscreenView! },
					{ binding: 3, resource: { buffer: this.stainParamsBuffer! } },
				],
			})
		}
	}

	/* ================================================================ */
	/*  Bind group management                                            */
	/* ================================================================ */

	private rebuildAllBindGroups() {
		this.rebuildComputeBindGroups()
		this.rebuildParticleRenderBindGroups()
		this.rebuildCircleRenderBindGroups()
		this.rebuildQuadBindGroup()
		this.rebuildStainBindGroups()
	}

	private rebuildComputeBindGroups() {
		const device = this.device!
		const layout = this.computeBindGroupLayout!

		// Group 0: read buf0, write buf1
		this.computeBindGroups[0] = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.particleBuffers[0]! } },
				{ binding: 1, resource: { buffer: this.particleBuffers[1]! } },
				{ binding: 2, resource: { buffer: this.simParamsBuffer! } },
				{ binding: 3, resource: { buffer: this.forceMatrixBuffer! } },
				{ binding: 4, resource: { buffer: this.stressBuffer! } },
			],
		})

		// Group 1: read buf1, write buf0
		this.computeBindGroups[1] = device.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.particleBuffers[1]! } },
				{ binding: 1, resource: { buffer: this.particleBuffers[0]! } },
				{ binding: 2, resource: { buffer: this.simParamsBuffer! } },
				{ binding: 3, resource: { buffer: this.forceMatrixBuffer! } },
				{ binding: 4, resource: { buffer: this.stressBuffer! } },
			],
		})
	}

	private rebuildParticleRenderBindGroups() {
		const device = this.device!
		const layout = this.particleRenderBindGroupLayout!
		const falloffView = this.falloffTexture!.createView()

		for (let i = 0; i < 2; i++) {
			this.particleRenderBindGroups[i] = device.createBindGroup({
				layout,
				entries: [
					{ binding: 0, resource: { buffer: this.particleBuffers[i]! } },
					{ binding: 1, resource: { buffer: this.renderParamsBuffer! } },
					{ binding: 2, resource: falloffView },
					{ binding: 3, resource: this.falloffSampler! },
					{ binding: 4, resource: { buffer: this.stressBuffer! } },
					{ binding: 5, resource: { buffer: this.detectionBuffer! } },
					{ binding: 6, resource: { buffer: this.radiusScaleBuffer! } },
				],
			})
		}
	}

	private rebuildCircleRenderBindGroups() {
		const device = this.device!
		const layout = this.particleRenderBindGroupLayout!
		const falloffView = this.falloffTexture!.createView()

		for (let i = 0; i < 2; i++) {
			this.circleRenderBindGroups[i] = device.createBindGroup({
				layout,
				entries: [
					{ binding: 0, resource: { buffer: this.particleBuffers[i]! } },
					{ binding: 1, resource: { buffer: this.circleRenderParamsBuffer! } },
					{ binding: 2, resource: falloffView },
					{ binding: 3, resource: this.falloffSampler! },
					{ binding: 4, resource: { buffer: this.stressBuffer! } },
					{ binding: 5, resource: { buffer: this.detectionBuffer! } },
					{ binding: 6, resource: { buffer: this.radiusScaleBuffer! } },
				],
			})
			this.detectionFillBindGroups[i] = device.createBindGroup({
				layout,
				entries: [
					{ binding: 0, resource: { buffer: this.particleBuffers[i]! } },
					{ binding: 1, resource: { buffer: this.detectionFillParamsBuffer! } },
					{ binding: 2, resource: falloffView },
					{ binding: 3, resource: this.falloffSampler! },
					{ binding: 4, resource: { buffer: this.stressBuffer! } },
					{ binding: 5, resource: { buffer: this.detectionBuffer! } },
					{ binding: 6, resource: { buffer: this.radiusScaleBuffer! } },
				],
			})
		}

		// Organism fill bind groups (ping-pong, same layout as particle render)
		for (let i = 0; i < 2; i++) {
			this.organismFillBindGroups[i] = device.createBindGroup({
				layout,
				entries: [
					{ binding: 0, resource: { buffer: this.particleBuffers[i]! } },
					{ binding: 1, resource: { buffer: this.organismFillParamsBuffer! } },
					{ binding: 2, resource: falloffView },
					{ binding: 3, resource: this.falloffSampler! },
					{ binding: 4, resource: { buffer: this.stressBuffer! } },
					{ binding: 5, resource: { buffer: this.detectionBuffer! } },
					{ binding: 6, resource: { buffer: this.radiusScaleBuffer! } },
				],
			})
		}

		// Organism centroid circle bind group
		if (
			this.organismCentroidBindGroupLayout &&
			this.organismCentroidBuffer &&
			this.organismCentroidParamsBuffer
		) {
			this.organismCentroidBindGroup = device.createBindGroup({
				layout: this.organismCentroidBindGroupLayout,
				entries: [
					{ binding: 0, resource: { buffer: this.organismCentroidBuffer } },
					{
						binding: 1,
						resource: { buffer: this.organismCentroidParamsBuffer },
					},
				],
			})
		}

		// Organism-level centroid circle bind group (reuses same layout + params)
		if (
			this.organismCentroidBindGroupLayout &&
			this.osmLevelCentroidBuffer &&
			this.organismCentroidParamsBuffer
		) {
			this.osmLevelCentroidBindGroup = device.createBindGroup({
				layout: this.organismCentroidBindGroupLayout,
				entries: [
					{ binding: 0, resource: { buffer: this.osmLevelCentroidBuffer } },
					{
						binding: 1,
						resource: { buffer: this.organismCentroidParamsBuffer },
					},
				],
			})
		}

		// Organism connection line bind group (reuses centroid layout: storage + uniform)
		if (
			this.organismCentroidBindGroupLayout &&
			this.organismLineBuffer &&
			this.organismCentroidParamsBuffer
		) {
			this.organismLineBindGroup = device.createBindGroup({
				layout: this.organismCentroidBindGroupLayout,
				entries: [
					{ binding: 0, resource: { buffer: this.organismLineBuffer } },
					{
						binding: 1,
						resource: { buffer: this.organismCentroidParamsBuffer },
					},
				],
			})
		}

		// Edge detection bind group: ID texture + color texture
		if (
			this.detectionEdgeBindGroupLayout &&
			this.detectionIdTexture &&
			this.detectionColorTexture
		) {
			this.detectionEdgeBindGroup = device.createBindGroup({
				layout: this.detectionEdgeBindGroupLayout,
				entries: [
					{ binding: 0, resource: this.detectionIdTexture.createView() },
					{ binding: 1, resource: this.detectionColorTexture.createView() },
				],
			})
		}

		// Organism edge bind group: organism ID texture only
		if (this.organismEdgeBindGroupLayout && this.organismIdTexture) {
			this.organismEdgeBindGroup = device.createBindGroup({
				layout: this.organismEdgeBindGroupLayout,
				entries: [
					{ binding: 0, resource: this.organismIdTexture.createView() },
				],
			})
		}

		// JFA compute bind groups (ping-pong for each pass)
		this.rebuildJfaBindGroups()
	}

	private rebuildJfaBindGroups() {
		const device = this.device!
		if (
			!this.jfaComputeBindGroupLayout ||
			!this.jfaOrganelleTextures[0] ||
			!this.jfaOrganelleTextures[1] ||
			!this.jfaOrganismTextures[0] ||
			!this.jfaOrganismTextures[1] ||
			!this.jfaEdgeBindGroupLayout ||
			!this.bubbleParamsBuffer
		)
			return

		// For each pass, create bind groups for both ping-pong directions
		// readIdx=0: read tex[0], write tex[1]; readIdx=1: read tex[1], write tex[0]
		this.jfaOrganelleBindGroups = [[], []]
		this.jfaOrganismBindGroups = [[], []]

		for (let i = 0; i < this.jfaPassCount; i++) {
			for (let readIdx = 0; readIdx < 2; readIdx++) {
				const writeIdx = 1 - readIdx
				this.jfaOrganelleBindGroups[readIdx].push(
					device.createBindGroup({
						layout: this.jfaComputeBindGroupLayout,
						entries: [
							{
								binding: 0,
								resource: this.jfaOrganelleTextures[readIdx]!.createView(),
							},
							{
								binding: 1,
								resource: this.jfaOrganelleTextures[writeIdx]!.createView(),
							},
							{
								binding: 2,
								resource: { buffer: this.jfaParamsBuffers[i]! },
							},
						],
					}),
				)
				this.jfaOrganismBindGroups[readIdx].push(
					device.createBindGroup({
						layout: this.jfaComputeBindGroupLayout,
						entries: [
							{
								binding: 0,
								resource: this.jfaOrganismTextures[readIdx]!.createView(),
							},
							{
								binding: 1,
								resource: this.jfaOrganismTextures[writeIdx]!.createView(),
							},
							{
								binding: 2,
								resource: { buffer: this.jfaParamsBuffers[i]! },
							},
						],
					}),
				)
			}
		}

		// JFA edge bind groups: JFA result + color + bubble params + ID texture
		const colorView = this.detectionColorTexture
			? this.detectionColorTexture.createView()
			: this.jfaOrganelleTextures[0]!.createView()
		const organelleIdView = this.detectionIdTexture
			? this.detectionIdTexture.createView()
			: this.jfaOrganelleTextures[0]!.createView()
		const organismIdView = this.organismIdTexture
			? this.organismIdTexture.createView()
			: this.jfaOrganismTextures[0]!.createView()
		for (let i = 0; i < 2; i++) {
			this.jfaOrganelleEdgeBindGroups[i] = device.createBindGroup({
				layout: this.jfaEdgeBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: this.jfaOrganelleTextures[i]!.createView(),
					},
					{ binding: 1, resource: colorView },
					{ binding: 2, resource: { buffer: this.bubbleParamsBuffer } },
					{ binding: 3, resource: organelleIdView },
				],
			})
			this.jfaOrganismEdgeBindGroups[i] = device.createBindGroup({
				layout: this.jfaEdgeBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: this.jfaOrganismTextures[i]!.createView(),
					},
					{ binding: 1, resource: colorView },
					{ binding: 2, resource: { buffer: this.bubbleParamsBuffer } },
					{ binding: 3, resource: organismIdView },
				],
			})
		}
	}

	private rebuildQuadBindGroup() {
		const device = this.device!
		// Two bind groups: [0] reads stain[1], [1] reads stain[0]
		// After stain pass writes to stain[1-pingPong], quad reads stain[1-pingPong]
		for (let i = 0; i < 2; i++) {
			const stainReadIdx = 1 - i // stain pass reading [i] writes [1-i]
			this.quadBindGroups[i] = device.createBindGroup({
				layout: this.quadBindGroupLayout!,
				entries: [
					{ binding: 0, resource: this.offscreenView! },
					{ binding: 1, resource: this.offscreenSampler! },
					{ binding: 2, resource: { buffer: this.quadParamsBuffer! } },
					{ binding: 3, resource: this.stainViews[stainReadIdx] ?? this.offscreenView! },
				],
			})
		}
	}

	/* ================================================================ */
	/*  Data upload helpers                                               */
	/* ================================================================ */

	private uploadBubbleParams() {
		if (!this.device || !this.bubbleParamsBuffer) return
		this.device.queue.writeBuffer(
			this.bubbleParamsBuffer,
			0,
			new Float32Array([
				this.bubbleThreshold,
				this.bubbleEdgeWidth,
				this.bubbleThreshold * 0.5,
				0,
			]),
		)
	}

	private uploadParticleData() {
		const device = this.device!
		const n = this.count
		const types = this.getTypeIds()
		const typeIdMap = new Map<string, number>()
		types.forEach((t, i) => typeIdMap.set(t, i))

		// Build flat buffer matching Particle struct layout
		const data = new ArrayBuffer(n * PARTICLE_STRIDE)
		const f32 = new Float32Array(data)
		const u32 = new Uint32Array(data)

		for (let i = 0; i < n; i++) {
			const p = this.particles[i]
			const base = i * (PARTICLE_STRIDE / 4) // index in f32/u32 terms
			f32[base + 0] = p.x // pos.x
			f32[base + 1] = p.y // pos.y
			f32[base + 2] = p.vx // vel.x
			f32[base + 3] = p.vy // vel.y
			f32[base + 4] = p.color[0] // color.r
			f32[base + 5] = p.color[1] // color.g
			f32[base + 6] = p.color[2] // color.b
			f32[base + 7] = 1.0 // color.a (unused)
			u32[base + 8] = typeIdMap.get(p.groupId) ?? 0 // typeId
			u32[base + 9] = 0 // pad
			u32[base + 10] = 0 // pad
			u32[base + 11] = 0 // pad
		}

		// Write to both ping-pong buffers so initial state is correct
		device.queue.writeBuffer(this.particleBuffers[0]!, 0, data)
		device.queue.writeBuffer(this.particleBuffers[1]!, 0, data)
	}

	private uploadForceMatrix() {
		const device = this.device!
		const types = this.getTypeIds()
		const n = types.length
		const flat = new Float32Array(MAX_TYPES * MAX_TYPES)

		for (let si = 0; si < n; si++) {
			const row = this.forceMatrix[types[si]]
			if (!row) continue
			for (let ti = 0; ti < n; ti++) {
				flat[si * MAX_TYPES + ti] = row[types[ti]] ?? 0
			}
		}

		device.queue.writeBuffer(this.forceMatrixBuffer!, 0, flat)
	}

	/**
	 * Upload per-particle detection IDs into the detection buffer.
	 * This is a separate GPU buffer (1 u32 per particle), avoiding strided writes
	 * into the particle buffer and keeping detection data cleanly separated.
	 */
	private uploadDetectionIds(frame: DetectionFrame, n: number) {
		const device = this.device!
		const buf = this.detectionBuffer
		if (!buf) return

		const data = new Uint32Array(n)

		// Build organelle → organism lookup
		const orgToOsm = new Map<number, number>()
		for (const osm of frame.organisms) {
			for (const oid of osm.organelleIds) {
				orgToOsm.set(oid, osm.id + 1)
			}
		}

		// Compute organism sizes (particle count) for depth ranking
		const osmSizes = new Map<number, number>()
		for (const org of frame.organelles) {
			const osmId = orgToOsm.get(org.id) ?? 0
			if (osmId > 0) {
				osmSizes.set(
					osmId,
					(osmSizes.get(osmId) ?? 0) + org.particleIndices.length,
				)
			}
		}
		// Sort by size descending: largest → highest rank (furthest back)
		const sorted = [...osmSizes.entries()].sort((a, b) => b[1] - a[1])
		this.organismDepthRanks.clear()
		for (let i = 0; i < sorted.length; i++) {
			this.organismDepthRanks.set(sorted[i][0], i + 1)
		}

		// Pack: bits 0-15 = organelleId, bits 16-23 = organismId, bits 24-31 = depthRank
		const inOrganelle = new Set<number>()
		for (const org of frame.organelles) {
			const orgId = (org.id + 1) & 0xffff
			const osmId = (orgToOsm.get(org.id) ?? 0) & 0xff
			const depthRank =
				(this.organismDepthRanks.get(orgToOsm.get(org.id) ?? 0) ?? 0) & 0xff
			const packed = orgId | (osmId << 16) | (depthRank << 24)
			for (let k = 0; k < org.particleIndices.length; k++) {
				const pi = org.particleIndices[k]
				if (pi < n) {
					data[pi] = packed
					inOrganelle.add(pi)
				}
			}
		}

		// Held particles (timer still active but not in current organelles) stay marked
		for (const [pi] of frame.holdTimers) {
			if (pi < n && !inOrganelle.has(pi) && data[pi] === 0) {
				data[pi] = 1 // minimal flag: organelleId=1, organismId=0
			}
		}

		device.queue.writeBuffer(buf, 0, data)
	}

	/** Get current param values for the active particle effect */
	private getActiveParticleParams(): number[] {
		const id = this.activeParticleEffect.id
		return (
			this.particleEffectParams[id] ?? effectDefaults(this.activeParticleEffect)
		)
	}

	/** Get current param values for the active post effect */
	private getActivePostParams(): number[] {
		const id = this.activePostEffect.id
		return this.postEffectParams[id] ?? effectDefaults(this.activePostEffect)
	}

	/** Ensure defaults exist for all effects */
	private initEffectParams() {
		for (const e of particleEffects) {
			if (!this.particleEffectParams[e.id])
				this.particleEffectParams[e.id] = effectDefaults(e)
		}
		for (const e of postEffects) {
			if (!this.postEffectParams[e.id])
				this.postEffectParams[e.id] = effectDefaults(e)
		}
	}

	/** Sync hidden inputs with current param maps (triggers save) */
	private syncParamHiddenInputs() {
		if (this._hiddenParticleParams) {
			this._hiddenParticleParams.value = JSON.stringify(
				this.particleEffectParams,
			)
			this._hiddenParticleParams.dispatchEvent(
				new Event("input", { bubbles: true }),
			)
		}
		if (this._hiddenPostParams) {
			this._hiddenPostParams.value = JSON.stringify(this.postEffectParams)
			this._hiddenPostParams.dispatchEvent(
				new Event("input", { bubbles: true }),
			)
		}
	}

	private uploadRenderParams() {
		const device = this.device!
		const eff = this.getEffectiveParams()
		const params = this.getActiveParticleParams()
		const data = new ArrayBuffer(32)
		const f32 = new Float32Array(data)
		const u32 = new Uint32Array(data)
		f32[0] = this.width
		f32[1] = this.height
		f32[2] = eff.pointSize
		u32[3] = 0 // mode = gradient
		f32[4] = this.time
		f32[5] = params[0] ?? 0
		f32[6] = params[1] ?? 0
		f32[7] = params[2] ?? 0
		device.queue.writeBuffer(this.renderParamsBuffer!, 0, data)

		// Circle overlay params
		const data2 = new ArrayBuffer(32)
		const f32b = new Float32Array(data2)
		const u32b = new Uint32Array(data2)
		f32b[0] = this.width
		f32b[1] = this.height
		f32b[2] = 12.0
		u32b[3] = 1 // mode=1: circle overlay
		f32b[4] = this.time
		f32b[5] = 1.0
		f32b[6] = 0.8 // solid core fill so particle albedo shows through
		f32b[7] = 0
		device.queue.writeBuffer(this.circleRenderParamsBuffer!, 0, data2)

		// Detection ID-fill params: 1.5x point size, mode=2 (discard undetected)
		const data3 = new ArrayBuffer(32)
		const f32c = new Float32Array(data3)
		const u32c = new Uint32Array(data3)
		f32c[0] = this.width
		f32c[1] = this.height
		f32c[2] = eff.pointSize * 1.5 // 1.5x to close inter-particle gaps in clusters
		u32c[3] = 2 // mode=2: discard non-detected
		f32c[4] = this.time
		f32c[5] = 1.0
		f32c[6] = 0
		f32c[7] = 0
		device.queue.writeBuffer(this.detectionFillParamsBuffer!, 0, data3)

		// Organism fill params: 2.5x point size so outlines are bigger than organelle outlines
		const data4 = new ArrayBuffer(32)
		const f32d = new Float32Array(data4)
		const u32d = new Uint32Array(data4)
		f32d[0] = this.width
		f32d[1] = this.height
		f32d[2] = eff.pointSize * 2.5
		u32d[3] = 2
		f32d[4] = this.time
		f32d[5] = 1.0
		f32d[6] = 0
		f32d[7] = 0
		device.queue.writeBuffer(this.organismFillParamsBuffer!, 0, data4)

		// Organism centroid circle params: just resolution
		const data5 = new Float32Array([this.width, this.height, 0, 0])
		device.queue.writeBuffer(this.organismCentroidParamsBuffer!, 0, data5)
	}

	/**
	 * Snapshot the particle indices for a specific organelle beat.
	 * Returns null if the organelle can't be resolved from the current detection state.
	 */

	/**
	 * Re-anchor the bar grid so the next bar boundary fires immediately.
	 * Called when BPM, time multiplier, or beats-per-bar change mid-playback.
	 */
	private resetBarGrid(): void {
		if (this.audioGraph.isEnabled) {
			this.tSoundStart = this.audioGraph.currentTime
			this.musicBarNumber = -1
			this.currentScheduledBar = null
		}
	}

	/**
	 * Build a BarSnapshot from current detection state for the hit scheduler.
	 */
	private buildBarSnapshot(): BarSnapshot {
		const typeKeys = this.getTypeIds()
		const organisms: SnapshotOrganism[] = []

		if (this.organismRegistry && this.detectionState) {
			const organelleMap = new Map<number, OrganelleState>()
			for (const org of this.detectionState.organelles) {
				organelleMap.set(org.id, org)
			}

			for (const ro of this.organismRegistry.organisms) {
				const composition = new Map<number, number>()

				// BFS the organism tree to get organelle IDs in traversal order.
				// Most-connected organelles (structural core) come first.
				const bfsOrder: number[] = []
				const queue: OrganelleTreeNode[] = [ro.tree]
				while (queue.length > 0) {
					const node = queue.shift()!
					bfsOrder.push(node.organelleId)
					for (const child of node.children) queue.push(child)
				}

				const snapshotOrganelles: SnapshotOrganelle[] = []

				for (const orgId of bfsOrder) {
					const org = organelleMap.get(orgId)
					if (!org) continue

					composition.set(org.typeId, (composition.get(org.typeId) ?? 0) + 1)

					const speed = Math.sqrt(
						org.avgVelX * org.avgVelX + org.avgVelY * org.avgVelY,
					)
					const w =
						(org.maxCol - org.minCol + 1) *
						(this.detectionConfig.proximityRadius || 18)
					const h =
						(org.maxRow - org.minRow + 1) *
						(this.detectionConfig.proximityRadius || 18)
					const area = w * h
					const density = area > 0 ? org.particleIndices.length / area : 0
					const spatialRadius = Math.max(w, h) / 2

					// Angular offset from organism velocity vector (visual only)
					const dx = org.centroidX - ro.centroidX
					const dy = org.centroidY - ro.centroidY
					const orgAngle = Math.atan2(dy, dx)
					const velAngle = Math.atan2(ro.velY, ro.velX)
					const angularOffset = orgAngle - velAngle

					snapshotOrganelles.push({
						id: org.id,
						typeId: org.typeId,
						particleCount: org.particleIndices.length,
						centroidX: org.centroidX,
						centroidY: org.centroidY,
						centroidSpeed: speed,
						density,
						spatialRadius,
						angularOffset,
						crossTypeLinks: ro.crossTypeLinks.get(org.id) ?? 0,
					})
				}

				organisms.push({
					registryId: ro.registryId,
					colorSignature: ro.colorSignature,
					centroidX: ro.centroidX,
					centroidY: ro.centroidY,
					velX: ro.velX,
					velY: ro.velY,
					creationTime: ro.creationTime,
					organelles: snapshotOrganelles,
					composition,
				})
			}
		}

		return {
			organisms,
			globalMetrics: this.latestGlobalMetrics!,
			forceMatrix: this.forceMatrix,
			typeKeys,
			canvasWidth: this.width,
			beatsPerBar: this.musicBeatsPerBar,
		}
	}

	private snapshotPulseParticles(
		registryId: number,
		typeId: number,
		beatIndex: number,
	): Uint32Array | null {
		const frame = this.detectionState
		const registry = this.organismRegistry
		if (!frame || !registry) return null

		// Find the registered organism by registryId
		const ro = registry.organisms.find((o) => o.registryId === registryId)
		if (!ro) return null

		const organelleById = new Map<number, OrganelleState>()
		for (const org of frame.organelles) {
			organelleById.set(org.id, org)
		}

		// Pick one organelle of the target typeId within this organism.
		// Visual ordering: alternating left-right from organism velocity vector (§3.4).
		const matching: { id: number; angularOffset: number }[] = []
		const velAngle = Math.atan2(ro.velY, ro.velX)
		for (const id of ro.organelleIds) {
			const org = organelleById.get(id)
			if (org && org.typeId === typeId) {
				const orgAngle = Math.atan2(
					org.centroidY - ro.centroidY,
					org.centroidX - ro.centroidX,
				)
				matching.push({ id, angularOffset: orgAngle - velAngle })
			}
		}
		if (matching.length === 0) return null

		// Sort by |angular offset| ascending, then interleave: right, left, right, left...
		const right = matching
			.filter((m) => m.angularOffset >= 0)
			.sort((a, b) => a.angularOffset - b.angularOffset)
		const left = matching
			.filter((m) => m.angularOffset < 0)
			.sort((a, b) => -a.angularOffset + b.angularOffset)
		const visualOrder: number[] = []
		const maxLen = Math.max(right.length, left.length)
		for (let i = 0; i < maxLen; i++) {
			if (i < right.length) visualOrder.push(right[i].id)
			if (i < left.length) visualOrder.push(left[i].id)
		}

		const targetOrgId = visualOrder[beatIndex % visualOrder.length]
		const org = organelleById.get(targetOrgId)
		return org ? new Uint32Array(org.particleIndices) : null
	}

	/**
	 * Trigger visual pulses for each hit in a scheduled bar (§3.2).
	 * Converts AudioContext time → performance time for the pulse system.
	 */
	private triggerVisualPulses(
		bar: ScheduledBar,
		hitTimings: readonly {
			readonly startTime: number
			readonly endTime: number
		}[],
	): void {
		if (hitTimings.length === 0) return

		// Convert AudioContext time to performance.now() time
		const audioNow = this.audioGraph.currentTime
		const perfNow = performance.now() / 1000

		for (let i = 0; i < bar.hits.length; i++) {
			const hit = bar.hits[i]
			const timing = hitTimings[i]
			if (!timing) continue

			const particles = this.snapshotPulseParticles(
				hit.organismId,
				hit.typeId,
				hit.organelleIndex,
			)
			if (!particles) continue

			// Use the quantized hit.time (from the scheduler grid) so visual
			// pulses fire exactly on the beat, not the audio's adjusted startTime.
			const perfStart = perfNow + (hit.time - audioNow)
			const key = `${hit.organismId}:${hit.typeId}:${hit.organelleIndex}:${bar.barNumber}`

			const manualShape = this.envelopeEditor?.getShape() ?? null

			// Gate-aware timing: sustain fills the gate after attack+decay
			let atkDur = hit.envelope.attackDuration
			let decDur = hit.envelope.decayDuration
			const gateDur = hit.gateDuration
			if (gateDur < atkDur + decDur) {
				const ratio = gateDur / (atkDur + decDur)
				atkDur *= ratio
				decDur *= ratio
			}
			const susDur = Math.max(0, gateDur - atkDur - decDur)
			const relDur = hit.envelope.releaseDuration
			const totalDur = atkDur + decDur + susDur + relDur

			this.activePulses.set(key, {
				startTime: perfStart,
				duration: totalDur,
				particleIndices: particles,
				attackFrac: atkDur / totalDur,
				decayFrac: decDur / totalDur,
				sustainFrac: susDur / totalDur,
				peakLevel: hit.envelope.peakLevel,
				sustainLevel: hit.envelope.sustainLevel,
				envelopeLut: manualShape
					? buildGateAwareLUT(manualShape, atkDur, decDur, susDur, relDur, 128)
					: null,
			})
		}
	}

	/**
	 * Compute per-particle radius scale from active organelle pulses and upload to GPU.
	 * Uses the same pluck envelope shape as the audio: fast attack, quick decay, quadratic release.
	 * Particle indices are snapshotted at trigger time so detection changes don't reset the pulse.
	 */
	private uploadRadiusScales() {
		const device = this.device!
		const buf = this.radiusScaleBuffer
		if (!buf) return

		const n = this.count
		const scales = new Float32Array(n)
		scales.fill(1.0)

		const now = performance.now() / 1000
		if (this.activePulses.size > 0) {
			const expired: string[] = []
			for (const [key, pulse] of this.activePulses) {
				const elapsed = now - pulse.startTime
				if (elapsed >= pulse.duration) {
					expired.push(key)
					continue
				}

				// Envelope matching audio (§3.2): brightness follows envelope curve
				const x = elapsed / pulse.duration
				let envelope: number
				if (pulse.envelopeLut) {
					// Manual envelope: sample from LUT
					const idx = Math.min(
						pulse.envelopeLut.length - 1,
						Math.floor(x * (pulse.envelopeLut.length - 1)),
					)
					envelope = pulse.envelopeLut[idx]
				} else {
					// ADSR envelope
					const aEnd = pulse.attackFrac
					const dEnd = aEnd + pulse.decayFrac
					const sEnd = dEnd + pulse.sustainFrac
					if (x < aEnd) {
						envelope = aEnd > 0 ? pulse.peakLevel * (x / aEnd) : pulse.peakLevel
					} else if (x < dEnd) {
						const t = (x - aEnd) / (dEnd - aEnd)
						envelope =
							pulse.peakLevel - (pulse.peakLevel - pulse.sustainLevel) * t
					} else if (x < sEnd) {
						envelope = pulse.sustainLevel
					} else {
						const t = (x - sEnd) / (1.0 - sEnd)
						envelope = pulse.sustainLevel * (1 - t) * (1 - t)
					}
				}

				// Scale: 1.0 at rest, up to (1.0 + pulseScale) at peak
				const scale = 1.0 + envelope * this.getEffectiveParams().pulseScale

				// Apply to snapshotted particle indices
				for (let pi = 0; pi < pulse.particleIndices.length; pi++) {
					const idx = pulse.particleIndices[pi]
					if (idx < n && scale > scales[idx]) {
						scales[idx] = scale
					}
				}
			}

			for (const key of expired) this.activePulses.delete(key)
		}

		device.queue.writeBuffer(buf, 0, scales.buffer, 0, n * 4)
	}

	private uploadOrganismCentroids(frame: DetectionFrame) {
		const organelles = frame.organelles
		const count = Math.min(organelles.length, 256)
		this.organismCentroidCount = count

		// Build organelle index → organism ID (1-based) lookup
		const osmIdByOrganelle = new Uint8Array(count) // 0 = no organism
		for (const osm of frame.organisms) {
			for (const orgId of osm.organelleIds) {
				if (orgId < count) osmIdByOrganelle[orgId] = osm.id + 1
			}
		}

		// Store snapshot with velocities for per-frame extrapolation
		this.organismCentroidSnapshot = []
		for (let i = 0; i < count; i++) {
			const org = organelles[i]
			this.organismCentroidSnapshot.push({
				cx: org.centroidX,
				cy: org.centroidY,
				vx: org.avgVelX,
				vy: org.avgVelY,
				id: osmIdByOrganelle[i],
			})
		}
		this.organismCentroidSnapshotTime = performance.now() / 1000

		// Compute organism connection edges: pairs of organelles that pass proximity + coherence
		const proxRadSq =
			this.detectionConfig.organismProximityRadius *
			this.detectionConfig.organismProximityRadius
		const cohThreshSq =
			this.detectionConfig.organismCoherenceThreshold *
			this.detectionConfig.organismCoherenceThreshold
		this.organismLineEdges = []
		for (let i = 0; i < count; i++) {
			const a = organelles[i]
			for (let j = i + 1; j < count; j++) {
				const b = organelles[j]
				// Only connect organelles within the SAME organism
				const osmA = osmIdByOrganelle[i]
				const osmB = osmIdByOrganelle[j]
				if (osmA === 0 || osmA !== osmB) continue
				if (a.typeId === b.typeId) continue
				const dx = a.centroidX - b.centroidX
				const dy = a.centroidY - b.centroidY
				if (dx * dx + dy * dy >= proxRadSq) continue
				const dvx = a.avgVelX - b.avgVelX
				const dvy = a.avgVelY - b.avgVelY
				if (dvx * dvx + dvy * dvy >= cohThreshSq) continue
				this.organismLineEdges.push([i, j])
				if (this.organismLineEdges.length >= 1024) break
			}
			if (this.organismLineEdges.length >= 1024) break
		}
		this.organismLineCount = this.organismLineEdges.length
	}

	/** Extrapolate organism centroids forward and upload to GPU — called every frame */
	private extrapolateOrganismCentroids() {
		const device = this.device
		const buf = this.organismCentroidBuffer
		const snap = this.organismCentroidSnapshot
		if (!device || !buf || snap.length === 0) return

		const now = performance.now() / 1000
		const dt = now - this.organismCentroidSnapshotTime
		const radius = 15.0 // fixed size — debug circles should not scale with particle radius

		const data = new ArrayBuffer(snap.length * 16)
		const f32 = new Float32Array(data)
		const u32 = new Uint32Array(data)

		for (let i = 0; i < snap.length; i++) {
			const s = snap[i]
			const off = i * 4
			f32[off + 0] = s.cx + s.vx * dt
			f32[off + 1] = s.cy + s.vy * dt
			f32[off + 2] = radius
			// Pack: bits 0-7 = osmId, bits 8-15 = depth rank
			const depthRank = this.organismDepthRanks.get(s.id) ?? 0
			u32[off + 3] = (s.id & 0xff) | ((depthRank & 0xff) << 8)
		}

		device.queue.writeBuffer(buf, 0, data)

		// Extrapolate line endpoints from the same centroid snapshot (32 bytes per segment)
		const lineBuf = this.organismLineBuffer
		const edges = this.organismLineEdges
		if (lineBuf && edges.length > 0) {
			const lineBuf32 = new ArrayBuffer(edges.length * 32)
			const lineF32 = new Float32Array(lineBuf32)
			const lineU32 = new Uint32Array(lineBuf32)
			for (let i = 0; i < edges.length; i++) {
				const [ai, bi] = edges[i]
				const a = snap[ai]
				const b = snap[bi]
				const off = i * 8 // 32 bytes = 8 floats/u32s
				lineF32[off + 0] = a.cx + a.vx * dt
				lineF32[off + 1] = a.cy + a.vy * dt
				lineF32[off + 2] = b.cx + b.vx * dt
				lineF32[off + 3] = b.cy + b.vy * dt
				// Pack: bits 0-7 = osmId, bits 8-15 = depth rank
				const depthRank = this.organismDepthRanks.get(a.id) ?? 0
				lineU32[off + 4] = (a.id & 0xff) | ((depthRank & 0xff) << 8)
				lineU32[off + 5] = 0
				lineU32[off + 6] = 0
				lineU32[off + 7] = 0
			}
			device.queue.writeBuffer(lineBuf, 0, lineBuf32)
		}
	}

	/** Snapshot organism-level centroids (averaged from constituent organelles) */
	private uploadOsmLevelCentroids(frame: DetectionFrame) {
		const organelleById = new Map<
			number,
			{ centroidX: number; centroidY: number; avgVelX: number; avgVelY: number }
		>()
		for (const org of frame.organelles) {
			organelleById.set(org.id, org)
		}

		const count = Math.min(frame.organisms.length, 128)
		this.osmLevelCentroidCount = count
		this.osmLevelCentroidSnapshot = []

		for (let i = 0; i < count; i++) {
			const osm = frame.organisms[i]
			// Average velocity from constituent organelles
			let vx = 0,
				vy = 0,
				n = 0
			for (const orgId of osm.organelleIds) {
				const org = organelleById.get(orgId)
				if (org) {
					vx += org.avgVelX
					vy += org.avgVelY
					n++
				}
			}
			if (n > 0) {
				vx /= n
				vy /= n
			}

			this.osmLevelCentroidSnapshot.push({
				cx: osm.centroidX,
				cy: osm.centroidY,
				vx,
				vy,
				id: osm.id + 1,
			})
		}
		this.osmLevelCentroidSnapshotTime = performance.now() / 1000
	}

	/** Extrapolate organism-level centroids forward and upload to GPU — called every frame */
	private extrapolateOsmLevelCentroids() {
		const device = this.device
		const buf = this.osmLevelCentroidBuffer
		const snap = this.osmLevelCentroidSnapshot
		if (!device || !buf || snap.length === 0) return

		const now = performance.now() / 1000
		const dt = now - this.osmLevelCentroidSnapshotTime
		const radius = 30.0 // 2x the organelle centroid circle radius (15.0)

		const data = new ArrayBuffer(snap.length * 16)
		const f32 = new Float32Array(data)
		const u32 = new Uint32Array(data)

		for (let i = 0; i < snap.length; i++) {
			const s = snap[i]
			const off = i * 4
			f32[off + 0] = s.cx + s.vx * dt
			f32[off + 1] = s.cy + s.vy * dt
			f32[off + 2] = radius
			u32[off + 3] = s.id
		}

		device.queue.writeBuffer(buf, 0, data)
	}

	private uploadQuadParams() {
		const device = this.device!
		const params = this.getActivePostParams()
		const data = new Float32Array(8) // padded to 32 bytes
		data[0] = this.time
		data[1] = params[0] ?? 0
		data[2] = params[1] ?? 0
		data[3] = params[2] ?? 0
		data[4] = params[3] ?? 0
		device.queue.writeBuffer(this.quadParamsBuffer!, 0, data)
	}



	private uploadFalloffLUT(lut?: Float32Array) {
		const device = this.device
		if (!device || !this.falloffTexture) return
		const data = lut ?? this.curveEditor?.getLUT()
		if (!data) return
		const bytes = new Uint8Array(data.length)
		for (let i = 0; i < data.length; i++) {
			bytes[i] = Math.round(data[i] * 255)
		}
		device.queue.writeTexture(
			{ texture: this.falloffTexture },
			bytes,
			{ bytesPerRow: 256 },
			{ width: 256, height: 1 },
		)
	}

	/* ================================================================ */
	/*  rebuildBuffers — called when particle count/colors change via UI */
	/* ================================================================ */

	private rebuildBuffers() {
		this.count = this.particles.length
		this.particleBufferDirty = true
		this.forceMatrixDirty = true
		this.uploadRenderParams()
	}

	/** Write only color (and type-id) fields into the GPU buffers,
	 *  leaving positions and velocities untouched so the simulation
	 *  continues from its current state. */
	private uploadParticleColors() {
		const device = this.device!
		const n = this.count
		const types = this.getTypeIds()
		const typeIdMap = new Map<string, number>()
		types.forEach((t, i) => typeIdMap.set(t, i))

		// color starts at float offset 4 (16 bytes) within each particle struct
		// layout: [pos.x, pos.y, vel.x, vel.y, r, g, b, a, typeId, pad, pad, pad]
		const colorOffset = 4 * 4 // 16 bytes
		const colorSize = 4 * 4 + 4 * 4 // color (4 floats) + typeId+pad (4 u32s) = 32 bytes

		for (let i = 0; i < n; i++) {
			const p = this.particles[i]
			const buf = new ArrayBuffer(colorSize)
			const f32 = new Float32Array(buf)
			const u32 = new Uint32Array(buf)
			f32[0] = p.color[0]
			f32[1] = p.color[1]
			f32[2] = p.color[2]
			f32[3] = 1.0
			u32[4] = typeIdMap.get(p.groupId) ?? 0
			u32[5] = 0
			u32[6] = 0
			u32[7] = 0

			const byteOffset = i * PARTICLE_STRIDE + colorOffset
			device.queue.writeBuffer(this.particleBuffers[0]!, byteOffset, buf)
			device.queue.writeBuffer(this.particleBuffers[1]!, byteOffset, buf)
		}
	}

	/** Write a contiguous range of CPU particles to both GPU buffers.
	 *  Used when appending new particles — existing GPU data is untouched. */
	private uploadParticleRange(startIdx: number, count: number) {
		if (count <= 0) return
		const device = this.device!
		const types = this.getTypeIds()
		const typeIdMap = new Map<string, number>()
		types.forEach((t, i) => typeIdMap.set(t, i))

		const data = new ArrayBuffer(count * PARTICLE_STRIDE)
		const f32 = new Float32Array(data)
		const u32 = new Uint32Array(data)

		for (let i = 0; i < count; i++) {
			const p = this.particles[startIdx + i]
			const base = i * (PARTICLE_STRIDE / 4)
			f32[base + 0] = p.x
			f32[base + 1] = p.y
			f32[base + 2] = p.vx
			f32[base + 3] = p.vy
			f32[base + 4] = p.color[0]
			f32[base + 5] = p.color[1]
			f32[base + 6] = p.color[2]
			f32[base + 7] = 1.0
			u32[base + 8] = typeIdMap.get(p.groupId) ?? 0
			u32[base + 9] = 0
			u32[base + 10] = 0
			u32[base + 11] = 0
		}

		const byteOffset = startIdx * PARTICLE_STRIDE
		device.queue.writeBuffer(this.particleBuffers[0]!, byteOffset, data)
		device.queue.writeBuffer(this.particleBuffers[1]!, byteOffset, data)
	}

	/** Remove particles at the given indices using swap-and-shrink.
	 *  Preserves GPU-evolved positions for all surviving particles. */
	private removeParticlesByIndices(indices: number[]) {
		if (indices.length === 0) return
		const device = this.device!
		const staging = this.particleStagingBuffer!

		// Sort descending so we shrink from the end
		const sorted = [...indices].sort((a, b) => b - a)
		let lastActive = this.particles.length - 1

		const encoder = device.createCommandEncoder()

		for (const removeIdx of sorted) {
			if (removeIdx > lastActive) continue // already past the active range
			if (removeIdx < lastActive) {
				// GPU: copy lastActive slot → removeIdx slot via staging, for both buffers
				for (const buf of this.particleBuffers) {
					encoder.copyBufferToBuffer(
						buf!,
						lastActive * PARTICLE_STRIDE,
						staging,
						0,
						PARTICLE_STRIDE,
					)
					encoder.copyBufferToBuffer(
						staging,
						0,
						buf!,
						removeIdx * PARTICLE_STRIDE,
						PARTICLE_STRIDE,
					)
				}
				// CPU: swap
				this.particles[removeIdx] = this.particles[lastActive]
			}
			lastActive--
		}

		device.queue.submit([encoder.finish()])

		// Truncate CPU array
		this.particles.length = lastActive + 1
		this.count = this.particles.length
	}

	/* ================================================================ */
	/*  getWindows – floating window definitions                         */
	/* ================================================================ */

	/** Shared helper: create collapsible section */
	private makeSection(
		title: string,
		defaultOpen: boolean,
	): { section: HTMLElement; body: HTMLElement } {
		const section = document.createElement("div")
		section.className = "settings-section" + (defaultOpen ? " open" : "")
		section.dataset.section = title

		const header = document.createElement("div")
		header.className = "settings-section-header"
		header.textContent = title
		header.addEventListener("click", () => section.classList.toggle("open"))
		section.appendChild(header)

		const body = document.createElement("div")
		body.className = "settings-section-body"
		section.appendChild(body)

		return { section, body }
	}

	private getOpenSections(container: HTMLElement): Set<string> {
		const open = new Set<string>()
		for (const el of container.querySelectorAll(".settings-section.open")) {
			const name = (el as HTMLElement).dataset.section
			if (name) open.add(name)
		}
		return open
	}

	private restoreOpenSections(container: HTMLElement, open: Set<string>) {
		for (const el of container.querySelectorAll(".settings-section")) {
			const name = (el as HTMLElement).dataset.section
			if (name && open.has(name)) (el as HTMLElement).classList.add("open")
		}
	}

	getWindows(): WindowDefinition[] {
		return [
			{
				id: "display",
				title: "Display",
				icon: "\uD83D\uDC41\uFE0F",
				category: "simulation",
				defaultVisible: true,
				defaultPosition: { x: 12, y: 60 },
				defaultWidth: 280,
				build: (c) => this.buildDisplayWindow(c),
			},
			{
				id: "physics",
				title: "Physics",
				icon: "\u2699\uFE0F",
				category: "simulation",
				defaultVisible: false,
				defaultPosition: { x: 12, y: 300 },
				defaultWidth: 280,
				build: (c) => this.buildPhysicsWindow(c),
			},
			{
				id: "particles",
				title: "Particles",
				icon: "\uD83D\uDFE2",
				category: "simulation",
				defaultVisible: false,
				defaultPosition: { x: 300, y: 60 },
				defaultWidth: 280,
				build: (c) => this.buildParticlesWindow(c),
			},
			{
				id: "music",
				title: "Music",
				icon: "\uD83C\uDFB5",
				category: "music",
				defaultVisible: false,
				defaultPosition: { x: 300, y: 300 },
				defaultWidth: 280,
				build: (c) => this.buildMusicWindow(c),
			},
			{
				id: "detection",
				title: "Detection",
				icon: "\uD83D\uDD2C",
				category: "detection",
				defaultVisible: false,
				defaultPosition: { x: 600, y: 60 },
				defaultWidth: 280,
				build: (c) => this.buildDetectionWindow(c),
			},
			{
				id: "shaders",
				title: "Shader Effects",
				icon: "\u2728",
				category: "visual",
				defaultVisible: false,
				defaultPosition: { x: 600, y: 300 },
				defaultWidth: 280,
				build: (c) => this.buildShadersWindow(c),
			},
		]
	}

	/* ================================================================ */
	/*  Window builders                                                  */
	/* ================================================================ */

	private buildDisplayWindow(container: HTMLElement) {
		// Viewport
		const viewport = this.makeSection("Viewport", true)

		viewport.body.appendChild(
			createNumberGroup({
				label: "Scale",
				value: this.scale,
				setting: "scale",
				min: 0.1,
				max: 5,
				step: 0.1,
				suffix: "x",
				onInput: (v) => {
					this.scale = v
				},
			}),
		)

		const toggleGroup = document.createElement("div")
		toggleGroup.className = "control-group"
		const toggleLabel = document.createElement("label")
		toggleLabel.textContent = "Show Particles"
		toggleGroup.appendChild(toggleLabel)
		const checkbox = document.createElement("input")
		checkbox.type = "checkbox"
		checkbox.checked = this.showCircleOverlay
		checkbox.dataset.setting = "showParticles"
		checkbox.addEventListener("change", () => {
			this.showCircleOverlay = checkbox.checked
		})
		toggleGroup.appendChild(checkbox)
		viewport.body.appendChild(toggleGroup)

		container.appendChild(viewport.section)

		// Particle Appearance
		const appearance = this.makeSection("Particle Appearance", true)

		appearance.body.appendChild(
			createNumberGroup({
				label: "Radius",
				value: this.pointSize,
				setting: "radius",
				min: 2,
				step: 1,
				suffix: "px",
				onInput: (v) => {
					this.pointSize = v
					this.uploadRenderParams()
				},
			}),
		)

		appearance.body.appendChild(
			createNumberGroup({
				label: "Pulse Scale",
				value: this.pulseScale,
				setting: "pulseScale",
				min: 0.1,
				step: 0.1,
				suffix: "x",
				onInput: (v) => {
					this.pulseScale = v
				},
			}),
		)

		container.appendChild(appearance.section)
	}

	private buildPhysicsWindow(container: HTMLElement) {
		// Pause toggle
		const pauseGroup = document.createElement("div")
		pauseGroup.className = "control-group"
		const pauseLabel = document.createElement("label")
		pauseLabel.textContent = "Pause Simulation"
		pauseGroup.appendChild(pauseLabel)
		const pauseCheckbox = document.createElement("input")
		pauseCheckbox.type = "checkbox"
		pauseCheckbox.checked = false
		pauseCheckbox.dataset.setting = "simPaused"
		pauseCheckbox.addEventListener("change", () => {
			window.dispatchEvent(
				new CustomEvent("sim-pause", {
					detail: { paused: pauseCheckbox.checked },
				}),
			)
		})
		window.addEventListener("sim-pause", ((
			e: CustomEvent<{ paused: boolean }>,
		) => {
			pauseCheckbox.checked = e.detail.paused
		}) as EventListener)
		pauseGroup.appendChild(pauseCheckbox)
		container.appendChild(pauseGroup)

		// Auto Balance toggle (top-level, like Enable Sound in music)
		const autoBalanceGroup = document.createElement("div")
		autoBalanceGroup.className = "control-group"
		const autoBalanceLabel = document.createElement("label")
		autoBalanceLabel.textContent = "Auto Balance"
		autoBalanceGroup.appendChild(autoBalanceLabel)
		const autoBalanceCheckbox = document.createElement("input")
		autoBalanceCheckbox.type = "checkbox"
		autoBalanceCheckbox.checked = this.autoBalanceEnabled
		autoBalanceCheckbox.dataset.setting = "autoBalance"
		autoBalanceGroup.appendChild(autoBalanceCheckbox)
		container.appendChild(autoBalanceGroup)

		// Read-only summary shown when auto-balance is ON
		const autoBalanceSummary = document.createElement("div")
		autoBalanceSummary.className = "auto-balance-summary"
		this._autoBalanceSummary = autoBalanceSummary
		this.renderAutoBalanceSummary()
		container.appendChild(autoBalanceSummary)

		// Container for all manual force sections (hidden when auto-balance ON)
		const forceSliders = document.createElement("div")

		// Force Reach
		const forceReach = this.makeSection("Force Reach", false)

		const affectRadiusEl = createNumberGroup({
			label: "Affect Radius",
			value: this.affectRadius,
			setting: "affectRadius",
			min: 1,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				this.affectRadius = v
			},
		})
		this._affectRadiusInput = affectRadiusEl
		forceReach.body.appendChild(affectRadiusEl)

		const forceRepelDistanceEl = createNumberGroup({
			label: "Force/Repel Distance",
			value: this.forceRepelDistance,
			setting: "forceRepelDistance",
			min: 0,
			step: 1,
			suffix: "px",
			onInput: (v) => {
				this.forceRepelDistance = v
			},
		})
		this._forceRepelDistanceInput = forceRepelDistanceEl
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
			"Dbl-click: add/remove \u2022 Right-click: remove \u2022 Drag handles"
		falloffGroup.appendChild(falloffHint)

		this.curveEditor = new CurveEditor(falloffGroup)
		const hiddenCurveInput = document.createElement("input")
		hiddenCurveInput.type = "hidden"
		hiddenCurveInput.dataset.setting = "falloffCurve"
		hiddenCurveInput.value = this.curveEditor.toJSON()
		this.curveEditor.onChange((lut) => {
			this.uploadFalloffLUT(lut)
			hiddenCurveInput.value = this.curveEditor!.toJSON()
			hiddenCurveInput.dispatchEvent(new Event("input", { bubbles: true }))
		})
		hiddenCurveInput.addEventListener("input", () => {
			if (
				this.curveEditor &&
				hiddenCurveInput.value !== this.curveEditor.toJSON()
			) {
				this.curveEditor.fromJSON(hiddenCurveInput.value)
				this.uploadFalloffLUT()
			}
		})
		falloffGroup.appendChild(hiddenCurveInput)
		forceReach.body.appendChild(falloffGroup)

		forceSliders.appendChild(forceReach.section)

		// Force Strength
		const forceStrength = this.makeSection("Force Strength", false)

		const forceStrengthVu = document.createElement("vu-meter") as VuMeter
		const baseStrengthEl = createNumberGroup({
			label: "Force Strength",
			value: this.baseStrength,
			setting: "baseStrength",
			min: 1,
			step: 1,
			onInput: (v) => {
				this.baseStrength = v
				forceStrengthVu.value = Math.min(1, v / 500)
			},
		})
		baseStrengthEl.appendChild(forceStrengthVu)
		forceStrengthVu.value = Math.min(1, this.baseStrength / 500)
		this._forceStrengthVu = forceStrengthVu
		this._baseStrengthInput = baseStrengthEl
		forceStrength.body.appendChild(baseStrengthEl)

		const repelStrengthVu = document.createElement("vu-meter") as VuMeter
		const repelStrengthEl = createNumberGroup({
			label: "Repel Strength",
			value: this.repelStrength,
			setting: "repelStrength",
			min: 0,
			step: 1,
			onInput: (v) => {
				this.repelStrength = v
				repelStrengthVu.value = Math.min(1, v / 500)
			},
		})
		repelStrengthEl.appendChild(repelStrengthVu)
		repelStrengthVu.value = Math.min(1, this.repelStrength / 500)
		this._repelStrengthVu = repelStrengthVu
		this._repelStrengthInput = repelStrengthEl
		forceStrength.body.appendChild(repelStrengthEl)

		forceSliders.appendChild(forceStrength.section)

		// Crowd Density
		const crowdDensity = this.makeSection("Crowd Density", false)

		const crowdLimitEl = createNumberGroup({
			label: "Crowd Limit",
			value: this.crowdLimit,
			setting: "crowdLimit",
			min: 1,
			step: 1,
			onInput: (v) => {
				this.crowdLimit = v
			},
		})
		this._crowdLimitInput = crowdLimitEl
		crowdDensity.body.appendChild(crowdLimitEl)

		const spreadGauge = document.createElement("mini-gauge") as MiniGauge
		const spreadEl = createNumberGroup({
			label: "Spread",
			value: this.spread,
			setting: "spread",
			min: 0,
			max: 100,
			step: 1,
			suffix: "%",
			onInput: (v) => {
				this.spread = v
				spreadGauge.value = v / 100
			},
		})
		spreadEl.appendChild(spreadGauge)
		spreadGauge.value = this.spread / 100
		this._spreadGauge = spreadGauge
		this._spreadInput = spreadEl
		crowdDensity.body.appendChild(spreadEl)

		forceSliders.appendChild(crowdDensity.section)

		// Toggle visibility based on auto-balance state
		const updateForceVisibility = () => {
			forceSliders.style.display = this.autoBalanceEnabled ? "none" : ""
			autoBalanceSummary.style.display = this.autoBalanceEnabled ? "" : "none"
		}
		updateForceVisibility()

		autoBalanceCheckbox.addEventListener("change", () => {
			this.autoBalanceEnabled = autoBalanceCheckbox.checked
			updateForceVisibility()
			if (this.autoBalanceEnabled) {
				this.predictionDirty = true
			}
		})

		container.appendChild(forceSliders)

		// Speed Limiter — always visible, independent of auto-balance
		const speedLimiter = this.makeSection("Speed Limiter", false)

		const maxSpeedEl = createNumberGroup({
			label: "Max Speed",
			value: this.maxSpeedPct,
			setting: "maxSpeedPct",
			min: 1,
			max: 100,
			step: 0.1,
			suffix: "%",
			onInput: (v) => {
				this.maxSpeedPct = v
			},
		})
		speedLimiter.body.appendChild(maxSpeedEl)

		container.appendChild(speedLimiter.section)
	}

	private buildParticlesWindow(container: HTMLElement) {
		this._particlesContainer = container
		// --- Species section ---
		const typesAccordion = this.makeSection("Species", false)

		const typeMap = new Map<string, CustomParticle[]>()
		for (const type of this.groupNames.keys()) {
			typeMap.set(type, [])
		}
		for (const p of this.particles) {
			let list = typeMap.get(p.groupId)
			if (!list) {
				list = []
				typeMap.set(p.groupId, list)
			}
			list.push(p)
		}

		const typesSection = document.createElement("div")
		typesSection.className = "particle-types-section"

		const rebuildParticles = (c: HTMLElement) => this.buildParticlesWindow(c)
		for (const [type, members] of typeMap) {
			this.buildTypeRow(
				typesSection,
				container,
				type,
				members,
				rebuildParticles,
			)
		}

		// Randomize counts button
		const randomizeCountsBtn = document.createElement("button")
		randomizeCountsBtn.className = "force-matrix-randomize"
		randomizeCountsBtn.textContent = "Randomize"
		randomizeCountsBtn.title = "Randomize particle counts"
		randomizeCountsBtn.addEventListener("click", () => this.randomizeCounts())
		typesSection.appendChild(randomizeCountsBtn)

		const autoCountsToggle = document.createElement("div")
		autoCountsToggle.className = "control-group"
		const autoCountsLabel = document.createElement("label")
		autoCountsLabel.textContent = "Auto Randomize"
		autoCountsToggle.appendChild(autoCountsLabel)
		const autoCountsCheckbox = document.createElement("input")
		autoCountsCheckbox.type = "checkbox"
		autoCountsCheckbox.checked = this.autoRandomizeCountsEnabled
		autoCountsCheckbox.dataset.setting = "autoRandomizeCounts"
		autoCountsCheckbox.addEventListener("change", () => {
			this.autoRandomizeCountsEnabled = autoCountsCheckbox.checked
		})
		autoCountsToggle.appendChild(autoCountsCheckbox)
		const autoCountsClock = document.createElement("mini-clock") as MiniClock
		autoCountsToggle.appendChild(autoCountsClock)
		this._autoRandomizeCountsClock = autoCountsClock
		typesSection.appendChild(autoCountsToggle)

		// Add Particle button
		const addParticleBtn = document.createElement("button")
		addParticleBtn.className = "add-particle-btn"
		addParticleBtn.textContent = "+"
		addParticleBtn.title = "Add particle type"
		addParticleBtn.addEventListener("click", () => {
			const newType = this.generateGroupId()
			const newName = this.generateName()
			this.groupNames.set(newType, newName)
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
			this.groupColors.set(newType, newColor)

			const newCount = Math.min(200, MAX_PARTICLES - this.particles.length)
			if (newCount <= 0) return
			const startIdx = this.particles.length
			for (let i = 0; i < newCount; i++) {
				this.particles.push(
					new CustomParticle(
						Math.random() * this.width,
						Math.random() * this.height,
						newType,
						[newColor[0], newColor[1], newColor[2]],
					),
				)
			}
			this.count = this.particles.length
			this.uploadParticleRange(startIdx, newCount)
			this.forceMatrixDirty = true

			const types = this.getTypeIds()
			this.forceMatrix = resizeMatrix(this.forceMatrix, types)

			// Rebuild this window, preserving accordion states
			const openSections = this.getOpenSections(container)
			container.innerHTML = ""
			this.buildParticlesWindow(container)
			this.restoreOpenSections(container, openSections)
			container.dispatchEvent(new Event("change", { bubbles: true }))
		})
		typesSection.appendChild(addParticleBtn)

		typesAccordion.body.appendChild(typesSection)
		container.appendChild(typesAccordion.section)

		// --- Interaction Matrix section ---
		const matrixAccordion = this.makeSection("Interaction Matrix", false)
		this.buildMatrixUI(matrixAccordion.body, container)

		const hiddenMatrixInput = document.createElement("input")
		hiddenMatrixInput.type = "hidden"
		hiddenMatrixInput.dataset.setting = "forceMatrix"
		hiddenMatrixInput.value = matrixToJSON(this.forceMatrix)
		hiddenMatrixInput.addEventListener("input", () => {
			const types = this.getTypeIds()
			const restored = matrixFromJSON(hiddenMatrixInput.value, types)
			if (matrixToJSON(restored) !== matrixToJSON(this.forceMatrix)) {
				this.forceMatrix = restored
				this.forceMatrixDirty = true
				const matrixContainer = matrixAccordion.body.querySelector(
					".force-matrix-container",
				)
				if (matrixContainer) {
					matrixContainer.remove()
					this.buildMatrixUI(matrixAccordion.body)
				}
			}
		})
		matrixAccordion.body.appendChild(hiddenMatrixInput)

		container.appendChild(matrixAccordion.section)
	}

	private buildMusicWindow(container: HTMLElement) {
		// --- Status Bar (always visible, not collapsible) ---
		const statusBar = document.createElement("div")

		const soundToggleGroup = document.createElement("div")
		soundToggleGroup.className = "control-group"
		const soundToggleLabel = document.createElement("label")
		soundToggleLabel.textContent = "Enable Sound"
		soundToggleGroup.appendChild(soundToggleLabel)
		const soundCheckbox = document.createElement("input")
		soundCheckbox.type = "checkbox"
		soundCheckbox.checked = false
		soundCheckbox.dataset.setting = "soundEnabled"
		soundCheckbox.addEventListener("change", () => {
			if (soundCheckbox.checked) {
				this.audioGraph.enable()
				this.tSoundStart = this.audioGraph.currentTime
				this.musicBarNumber = -1
				if (this.audioGraph.context && this.audioGraph.masterDestination) {
					this.bassLayer.init(
						this.audioGraph.context,
						this.audioGraph.masterDestination,
					)
					this.bassLayer.setMaxQuartalVoices(Math.max(2, Math.floor(this.voiceBudget / 4)))
					if (this.staccatoCurveEditor)
						this.bassLayer.setStaccatoLUT(this.staccatoCurveEditor.getLUT())
				}
			} else {
				this.audioGraph.disable()
				this.bassLayer.dispose()
			}
		})
		soundToggleGroup.appendChild(soundCheckbox)
		statusBar.appendChild(soundToggleGroup)

		const scrubToggleGroup = document.createElement("div")
		scrubToggleGroup.className = "control-group"
		const scrubToggleLabel = document.createElement("label")
		scrubToggleLabel.textContent = "Show Scrub Bar"
		scrubToggleGroup.appendChild(scrubToggleLabel)
		const scrubCheckbox = document.createElement("input")
		scrubCheckbox.type = "checkbox"
		scrubCheckbox.checked = true
		scrubCheckbox.dataset.setting = "showScrubBar"
		scrubCheckbox.addEventListener("change", () => {
			if (scrubCheckbox.checked) showBarVisualizer()
			else hideBarVisualizer()
		})
		scrubToggleGroup.appendChild(scrubCheckbox)
		statusBar.appendChild(scrubToggleGroup)
		// Sync initial checked state with bar visualizer
		if (scrubCheckbox.checked) showBarVisualizer()

		const toggleSeparator = document.createElement("div")
		toggleSeparator.style.height = "8px"
		statusBar.appendChild(toggleSeparator)

		const NOTE_NAMES = [
			"C",
			"C#",
			"D",
			"D#",
			"E",
			"F",
			"F#",
			"G",
			"G#",
			"A",
			"A#",
			"B",
		] as const
		const midiToNoteName = (midi: number) => {
			const note = NOTE_NAMES[((midi % 12) + 12) % 12]
			const octave = Math.floor(midi / 12) - 1
			return `${note}${octave}`
		}

		const makeStatusCell = (label: string) => {
			const cell = document.createElement("div")
			cell.className = "status-cell"
			const lbl = document.createElement("div")
			lbl.className = "status-cell-label"
			lbl.textContent = label
			const val = document.createElement("div")
			val.className = "status-cell-value"
			val.textContent = "\u2014"
			cell.appendChild(lbl)
			cell.appendChild(val)
			return { cell, val }
		}

		const modeCell = makeStatusCell("Mode")
		statusBar.appendChild(modeCell.cell)

		const statusGrid = document.createElement("div")
		statusGrid.className = "status-grid"
		const rootCell = makeStatusCell("Root")
		const stabilityCell = makeStatusCell("Stability")
		const stabilityBars = document.createElement(
			"stability-bars",
		) as StabilityBars
		stabilityCell.val.appendChild(stabilityBars)
		this._stabilityBars = stabilityBars
		statusGrid.appendChild(rootCell.cell)
		statusGrid.appendChild(stabilityCell.cell)
		statusBar.appendChild(statusGrid)

		setInterval(() => {
			if (!this.musicState) {
				modeCell.val.textContent = "\u2014"
				rootCell.val.textContent = "\u2014"
				stabilityBars.value = 0
				return
			}
			modeCell.val.textContent = this.musicState.currentMode.name
			rootCell.val.textContent = midiToNoteName(this.musicState.currentRootMidi)
			const s = this.musicState.netStability
			stabilityCell.val.textContent = s.toFixed(2)
			stabilityBars.value = s
			const r = Math.round(255 * (1 - s))
			const g = Math.round(200 * s)
			stabilityCell.val.style.color = `rgb(${r},${g},100)`
		}, 500)

		container.appendChild(statusBar)

		// Mix
		const mix = this.makeSection("Mix", false)
		const volumeVu = document.createElement("vu-meter") as VuMeter
		const volumeEl = createNumberGroup({
			label: "Volume",
			value: 50,
			setting: "soundVolume",
			min: 0,
			max: 100,
			step: 1,
			suffix: "%",
			onInput: (v) => {
				this.audioGraph.setVolume(v / 100)
				volumeVu.value = v / 100
			},
		})
		volumeEl.appendChild(volumeVu)
		volumeVu.value = 0.5
		this._volumeVu = volumeVu
		mix.body.appendChild(volumeEl)
		const multiplierInput = createNumberGroup({
			label: "Volume Multiplier",
			value: 1,
			setting: "soundVolumeMultiplier",
			min: 0.1,
			step: 0.1,
			suffix: "x",
			onInput: (v) => {
				const clamped = Math.max(0.1, v)
				this.audioGraph.setVolumeMultiplier(clamped)
			},
		})
		mix.body.appendChild(multiplierInput)
		container.appendChild(mix.section)

		// Master Filter (bus-wide LPF driven by curve editor)
		const filterSection = this.makeSection("Master Filter", false)

		const lpfQInput = createNumberGroup({
			label: "Resonance (Q)",
			value: 0.707,
			setting: "lpfQ",
			min: 0.1,
			max: 20,
			step: 0.1,
			onInput: (v) => {
				this.audioGraph.setLpfQ(Math.max(0.1, v))
			},
		})
		filterSection.body.appendChild(lpfQInput)

		const lpfCurveGroup = document.createElement("div")
		lpfCurveGroup.className = "control-group control-group-column"
		const lpfCurveLabel = document.createElement("label")
		lpfCurveLabel.className = "control-label"
		lpfCurveLabel.textContent = "Cutoff Curve"
		lpfCurveGroup.appendChild(lpfCurveLabel)
		const lpfCurveHint = document.createElement("div")
		lpfCurveHint.className = "control-hint"
		lpfCurveHint.textContent = "X: position in bar \u2022 Y: cutoff frequency"
		lpfCurveGroup.appendChild(lpfCurveHint)
		this.lpfCurveEditor = new CurveEditor(lpfCurveGroup)
		const hiddenLpfCurveInput = document.createElement("input")
		hiddenLpfCurveInput.type = "hidden"
		hiddenLpfCurveInput.dataset.setting = "lpfCurve"
		hiddenLpfCurveInput.value = this.lpfCurveEditor.toJSON()
		this.lpfCurveEditor.onChange((lut) => {
			this.audioGraph.setLpfLUT(lut)
			hiddenLpfCurveInput.value = this.lpfCurveEditor!.toJSON()
			hiddenLpfCurveInput.dispatchEvent(new Event("input", { bubbles: true }))
		})
		this.audioGraph.setLpfLUT(this.lpfCurveEditor.getLUT())
		hiddenLpfCurveInput.addEventListener("input", () => {
			if (
				this.lpfCurveEditor &&
				hiddenLpfCurveInput.value !== this.lpfCurveEditor.toJSON()
			) {
				this.lpfCurveEditor.fromJSON(hiddenLpfCurveInput.value)
				this.audioGraph.setLpfLUT(this.lpfCurveEditor.getLUT())
			}
		})
		lpfCurveGroup.appendChild(hiddenLpfCurveInput)
		filterSection.body.appendChild(lpfCurveGroup)
		container.appendChild(filterSection.section)

		// Tempo & Meter
		const tempoMeter = this.makeSection("Tempo & Meter", false)
		const bpmGauge = document.createElement("mini-gauge") as MiniGauge
		const bpmEl = createNumberGroup({
			label: "BPM",
			value: this.musicBpm,
			setting: "musicBpm",
			min: 20,
			max: 300,
			step: 5,
			suffix: "bpm",
			onInput: (v) => {
				this.musicBpm = v
				this.resetBarGrid()
				bpmGauge.value = (v - 20) / 280
			},
		})
		bpmEl.appendChild(bpmGauge)
		bpmGauge.value = (this.musicBpm - 20) / 280
		this._bpmGauge = bpmGauge
		tempoMeter.body.appendChild(bpmEl)

		const timeMultGroup = document.createElement("div")
		timeMultGroup.className = "control-group"
		const timeMultLabel = document.createElement("label")
		timeMultLabel.textContent = "Time Multiplier"
		timeMultGroup.appendChild(timeMultLabel)
		const timeMultRadios = document.createElement("div")
		timeMultRadios.className = "control-radios"
		for (const [label, val] of [
			["0.5x", 0.5],
			["1x", 1],
			["2x", 2],
		] as const) {
			const radio = document.createElement("input")
			radio.type = "radio"
			radio.name = "musicTimeMultiplier"
			radio.value = String(val)
			radio.checked = val === this.musicTimeMultiplier
			radio.dataset.setting = "musicTimeMultiplier"
			radio.addEventListener("change", () => {
				this.musicTimeMultiplier = val
				this.resetBarGrid()
			})
			const radioLabel = document.createElement("label")
			radioLabel.textContent = label
			radioLabel.prepend(radio)
			timeMultRadios.appendChild(radioLabel)
		}
		timeMultGroup.appendChild(timeMultRadios)
		tempoMeter.body.appendChild(timeMultGroup)

		tempoMeter.body.appendChild(
			createNumberGroup({
				label: "Beats Per Bar",
				value: this.musicBeatsPerBar,
				setting: "musicBeatsPerBar",
				min: 2,
				max: 8,
				step: 1,
				suffix: "beats",
				onInput: (v) => {
					this.musicBeatsPerBar = v
					this.resetBarGrid()
				},
			}),
		)
		tempoMeter.body.appendChild(
			createNumberGroup({
				label: "Voice Budget",
				value: this.voiceBudget,
				setting: "voiceBudget",
				min: 8,
				max: 64,
				step: 4,
				suffix: "voices",
				onInput: (v) => {
					this.voiceBudget = v
					this.bassLayer.setMaxQuartalVoices(Math.max(2, Math.floor(v / 4)))
				},
			}),
		)
		container.appendChild(tempoMeter.section)

		// Harmonic Engine
		const harmonicEngine = this.makeSection("Harmonic Engine", false)
		const phaseClock = document.createElement("mini-clock") as MiniClock
		const phaseRateEl = createNumberGroup({
			label: "Overtone Phase Rate",
			value: this.overtonePhaseRate,
			setting: "overtonePhaseRate",
			min: 1,
			max: 16,
			step: 1,
			suffix: "bars/phase",
			onInput: (v) => {
				this.overtonePhaseRate = v
			},
		})
		phaseRateEl.appendChild(phaseClock)
		this._phaseClock = phaseClock
		harmonicEngine.body.appendChild(phaseRateEl)
		harmonicEngine.body.appendChild(
			createNumberGroup({
				label: "Qualification",
				value: this.qualificationFraction,
				setting: "qualificationFraction",
				min: 0.1,
				max: 4,
				step: 0.1,
				suffix: "bars",
				onInput: (v) => {
					this.qualificationFraction = Math.round(v * 10) / 10
				},
			}),
		)
		const niceModeGroup = document.createElement("div")
		niceModeGroup.className = "control-group"
		const niceModeLabel = document.createElement("label")
		niceModeLabel.textContent = "Prefer Nice Modes"
		niceModeGroup.appendChild(niceModeLabel)
		const niceModeCheckbox = document.createElement("input")
		niceModeCheckbox.type = "checkbox"
		niceModeCheckbox.checked = this.preferNiceModes
		niceModeCheckbox.dataset.setting = "preferNiceModes"
		niceModeCheckbox.addEventListener("change", () => {
			this.preferNiceModes = niceModeCheckbox.checked
		})
		niceModeGroup.appendChild(niceModeCheckbox)
		harmonicEngine.body.appendChild(niceModeGroup)

		container.appendChild(harmonicEngine.section)

		// Bass voicing (bass envelope + staccato + phrase)
		const bassVoicing = this.makeSection("Bass Voicing", false)

		const bassEnvelopeHint = document.createElement("div")
		bassEnvelopeHint.className = "control-hint"
		bassEnvelopeHint.textContent =
			"Dbl-click: add/remove \u2022 Right-click: remove \u2022 Drag dividers"
		bassVoicing.body.appendChild(bassEnvelopeHint)

		this.bassEnvelopeEditor = new EnvelopeEditor(bassVoicing.body)
		const hiddenBassEnvelopeInput = document.createElement("input")
		hiddenBassEnvelopeInput.type = "hidden"
		hiddenBassEnvelopeInput.dataset.setting = "bassEnvelopeShape"
		hiddenBassEnvelopeInput.value = this.bassEnvelopeEditor.toJSON()
		this.bassEnvelopeEditor.onChange((shape) => {
			this.bassLayer.setBassEnvelope(shape)
			hiddenBassEnvelopeInput.value = this.bassEnvelopeEditor!.toJSON()
			hiddenBassEnvelopeInput.dispatchEvent(
				new Event("input", { bubbles: true }),
			)
		})
		this.bassLayer.setBassEnvelope(this.bassEnvelopeEditor.getShape())
		hiddenBassEnvelopeInput.addEventListener("input", () => {
			if (
				this.bassEnvelopeEditor &&
				hiddenBassEnvelopeInput.value !== this.bassEnvelopeEditor.toJSON()
			) {
				this.bassEnvelopeEditor.fromJSON(hiddenBassEnvelopeInput.value)
				this.bassLayer.setBassEnvelope(this.bassEnvelopeEditor.getShape())
			}
		})
		bassVoicing.body.appendChild(hiddenBassEnvelopeInput)

		const staccatoGroup = document.createElement("div")
		staccatoGroup.className = "control-group control-group-column"
		const staccatoLabel = document.createElement("label")
		staccatoLabel.className = "control-label"
		staccatoLabel.textContent = "Staccato Curve"
		staccatoGroup.appendChild(staccatoLabel)
		const staccatoHint = document.createElement("div")
		staccatoHint.className = "control-hint"
		staccatoHint.textContent = "X: particle velocity \u2022 Y: staccato amount"
		staccatoGroup.appendChild(staccatoHint)
		this.staccatoCurveEditor = new CurveEditor(staccatoGroup)
		const hiddenStaccatoInput = document.createElement("input")
		hiddenStaccatoInput.type = "hidden"
		hiddenStaccatoInput.dataset.setting = "staccatoCurve"
		hiddenStaccatoInput.value = this.staccatoCurveEditor.toJSON()
		this.staccatoCurveEditor.onChange((lut) => {
			this.bassLayer.setStaccatoLUT(lut)
			hiddenStaccatoInput.value = this.staccatoCurveEditor!.toJSON()
			hiddenStaccatoInput.dispatchEvent(new Event("input", { bubbles: true }))
		})
		hiddenStaccatoInput.addEventListener("input", () => {
			if (
				this.staccatoCurveEditor &&
				hiddenStaccatoInput.value !== this.staccatoCurveEditor.toJSON()
			) {
				this.staccatoCurveEditor.fromJSON(hiddenStaccatoInput.value)
				this.bassLayer.setStaccatoLUT(this.staccatoCurveEditor.getLUT())
			}
		})
		staccatoGroup.appendChild(hiddenStaccatoInput)
		bassVoicing.body.appendChild(staccatoGroup)

		// Phrase persistence
		const hiddenPhraseInput = document.createElement("input")
		hiddenPhraseInput.type = "hidden"
		hiddenPhraseInput.dataset.setting = "phrasePattern"
		hiddenPhraseInput.value = this.phrasePattern.join(",")
		hiddenPhraseInput.addEventListener("input", () => {
			const parts = hiddenPhraseInput.value.split(",") as BassDensity[]
			if (
				parts.length === 12 &&
				parts.every((p) => ["W", "H", "Q", "E"].includes(p))
			) {
				this.phrasePattern = parts
				setPhraseStripCells(this.phrasePattern, this.phraseMirror)
			}
		})
		bassVoicing.body.appendChild(hiddenPhraseInput)

		const hiddenMirrorInput = document.createElement("input")
		hiddenMirrorInput.type = "hidden"
		hiddenMirrorInput.dataset.setting = "phraseMirror"
		hiddenMirrorInput.value = String(this.phraseMirror)
		hiddenMirrorInput.addEventListener("input", () => {
			this.phraseMirror = hiddenMirrorInput.value === "true"
			setPhraseStripCells(this.phrasePattern, this.phraseMirror)
		})
		bassVoicing.body.appendChild(hiddenMirrorInput)

		setPhraseStripCells(this.phrasePattern, this.phraseMirror)
		onPhraseChange((cells, mirror) => {
			this.phrasePattern = [...cells]
			this.phraseMirror = mirror
			hiddenPhraseInput.value = cells.join(",")
			hiddenPhraseInput.dispatchEvent(new Event("input", { bubbles: true }))
			hiddenMirrorInput.value = String(mirror)
			hiddenMirrorInput.dispatchEvent(new Event("input", { bubbles: true }))
		})

		container.appendChild(bassVoicing.section)

		// Melody Envelope
		const melodyEnvelope = this.makeSection("Melody Envelope", false)
		const envelopeHint = document.createElement("div")
		envelopeHint.className = "control-hint"
		envelopeHint.textContent =
			"Dbl-click: add/remove \u2022 Right-click: remove \u2022 Drag dividers"
		melodyEnvelope.body.appendChild(envelopeHint)

		this.envelopeEditor = new EnvelopeEditor(melodyEnvelope.body)
		const hiddenEnvelopeInput = document.createElement("input")
		hiddenEnvelopeInput.type = "hidden"
		hiddenEnvelopeInput.dataset.setting = "envelopeShape"
		hiddenEnvelopeInput.value = this.envelopeEditor.toJSON()
		this.envelopeEditor.onChange((shape) => {
			this.audioGraph.setManualEnvelope(shape)
			hiddenEnvelopeInput.value = this.envelopeEditor!.toJSON()
			hiddenEnvelopeInput.dispatchEvent(new Event("input", { bubbles: true }))
		})
		this.audioGraph.setManualEnvelope(this.envelopeEditor.getShape())
		hiddenEnvelopeInput.addEventListener("input", () => {
			if (
				this.envelopeEditor &&
				hiddenEnvelopeInput.value !== this.envelopeEditor.toJSON()
			) {
				this.envelopeEditor.fromJSON(hiddenEnvelopeInput.value)
				this.audioGraph.setManualEnvelope(this.envelopeEditor.getShape())
			}
		})
		melodyEnvelope.body.appendChild(hiddenEnvelopeInput)
		container.appendChild(melodyEnvelope.section)
	}

	private buildDetectionWindow(container: HTMLElement) {
		// Organelle Detection (algorithm parameters only)
		const organelleDet = this.makeSection("Organelle Detection", true)

		organelleDet.body.appendChild(
			createNumberGroup({
				label: "Search Radius",
				value: this.detectionConfig.proximityRadius,
				setting: "detProximityRadius",
				min: 5,
				max: 200,
				step: 1,
				suffix: "px",
				onInput: (v) => {
					this.detectionConfig = { ...this.detectionConfig, proximityRadius: v }
					this.detectionState = null
				},
			}),
		)
		organelleDet.body.appendChild(
			createNumberGroup({
				label: "Coherence",
				value: this.detectionConfig.coherenceThreshold,
				setting: "detCoherenceThreshold",
				min: 1,
				max: 200,
				step: 1,
				suffix: "px",
				onInput: (v) => {
					this.detectionConfig = {
						...this.detectionConfig,
						coherenceThreshold: v,
					}
					this.detectionState = null
				},
			}),
		)
		organelleDet.body.appendChild(
			createNumberGroup({
				label: "Min Size",
				value: this.detectionConfig.minOrganelleSize,
				setting: "detMinOrganelleSize",
				min: 2,
				max: 50,
				step: 1,
				suffix: "particles",
				onInput: (v) => {
					this.detectionConfig = {
						...this.detectionConfig,
						minOrganelleSize: v,
					}
					this.detectionState = null
				},
			}),
		)
		const latchClock = document.createElement("mini-clock") as MiniClock
		const latchBeatsEl = createNumberGroup({
			label: "Latch Beats",
			value: this.detectionConfig.organelleLatchBeats,
			setting: "detOrganelleLatchBeats",
			min: 1,
			max: 16,
			step: 1,
			suffix: "beats",
			onInput: (v) => {
				this.detectionConfig = {
					...this.detectionConfig,
					organelleLatchBeats: v,
				}
				this.detectionState = null
			},
		})
		latchBeatsEl.appendChild(latchClock)
		this._latchClock = latchClock
		organelleDet.body.appendChild(latchBeatsEl)

		container.appendChild(organelleDet.section)

		// Organism Detection (algorithm parameters only)
		const organismDet = this.makeSection("Organism Detection", true)

		organismDet.body.appendChild(
			createNumberGroup({
				label: "Search Radius",
				value: this.detectionConfig.organismProximityRadius,
				setting: "detOrganismProximityRadius",
				min: 20,
				max: 400,
				step: 10,
				suffix: "px",
				onInput: (v) => {
					this.detectionConfig = {
						...this.detectionConfig,
						organismProximityRadius: v,
					}
					this.detectionState = null
				},
			}),
		)
		organismDet.body.appendChild(
			createNumberGroup({
				label: "Coherence",
				value: this.detectionConfig.organismCoherenceThreshold,
				setting: "detOrganismCoherenceThreshold",
				min: 10,
				max: 200,
				step: 10,
				suffix: "px",
				onInput: (v) => {
					this.detectionConfig = {
						...this.detectionConfig,
						organismCoherenceThreshold: v,
					}
					this.detectionState = null
				},
			}),
		)

		container.appendChild(organismDet.section)

		// Detection Overlays (all visual toggles and rendering params)
		const overlays = this.makeSection("Detection Overlays", true)

		// Organelle overlay toggle
		const orgOverlayGroup = document.createElement("div")
		orgOverlayGroup.className = "control-group"
		const orgOverlayLabel = document.createElement("label")
		orgOverlayLabel.textContent = "Organelle Overlay"
		orgOverlayGroup.appendChild(orgOverlayLabel)
		const orgOverlayCheckbox = document.createElement("input")
		orgOverlayCheckbox.type = "checkbox"
		orgOverlayCheckbox.checked = this.showOrganelleOverlay
		orgOverlayCheckbox.dataset.setting = "detShowOrganelleOverlay"
		orgOverlayCheckbox.addEventListener("change", () => {
			this.showOrganelleOverlay = orgOverlayCheckbox.checked
			if (!orgOverlayCheckbox.checked && this.detectionBuffer && this.device) {
				this.device.queue.writeBuffer(
					this.detectionBuffer,
					0,
					new Uint32Array(this.count),
				)
			}
		})
		orgOverlayGroup.appendChild(orgOverlayCheckbox)
		overlays.body.appendChild(orgOverlayGroup)

		// Organism outline toggle
		const osmOverlayGroup = document.createElement("div")
		osmOverlayGroup.className = "control-group"
		const osmOverlayLabel = document.createElement("label")
		osmOverlayLabel.textContent = "Organism Outline"
		osmOverlayGroup.appendChild(osmOverlayLabel)
		const osmOverlayCheckbox = document.createElement("input")
		osmOverlayCheckbox.type = "checkbox"
		osmOverlayCheckbox.checked = this.showOrganismOverlay
		osmOverlayCheckbox.dataset.setting = "detShowOrganismOverlay"
		osmOverlayCheckbox.addEventListener("change", () => {
			this.showOrganismOverlay = osmOverlayCheckbox.checked
		})
		osmOverlayGroup.appendChild(osmOverlayCheckbox)
		overlays.body.appendChild(osmOverlayGroup)

		// Organism centroids + connectors toggle
		const osmCentroidGroup = document.createElement("div")
		osmCentroidGroup.className = "control-group"
		const osmCentroidLabel = document.createElement("label")
		osmCentroidLabel.textContent = "Centroids & Connectors"
		osmCentroidGroup.appendChild(osmCentroidLabel)
		const osmCentroidCheckbox = document.createElement("input")
		osmCentroidCheckbox.type = "checkbox"
		osmCentroidCheckbox.checked = this.showOrganismCentroids
		osmCentroidCheckbox.dataset.setting = "detShowOrganismCentroids"
		osmCentroidCheckbox.addEventListener("change", () => {
			this.showOrganismCentroids = osmCentroidCheckbox.checked
			if (!osmCentroidCheckbox.checked) {
				this.organismCentroidCount = 0
				this.organismLineCount = 0
				this.organismLineEdges = []
			}
		})
		osmCentroidGroup.appendChild(osmCentroidCheckbox)
		overlays.body.appendChild(osmCentroidGroup)

		// Bubble boundary rendering controls
		overlays.body.appendChild(
			createNumberGroup({
				label: "Bubble Threshold",
				value: this.bubbleThreshold,
				setting: "detBubbleThreshold",
				min: 10,
				max: 150,
				step: 1,
				suffix: "px",
				onInput: (v) => {
					this.bubbleThreshold = v
					this.uploadBubbleParams()
				},
			}),
		)
		overlays.body.appendChild(
			createNumberGroup({
				label: "Edge Width",
				value: this.bubbleEdgeWidth,
				setting: "detBubbleEdgeWidth",
				min: 1,
				max: 10,
				step: 0.5,
				suffix: "px",
				onInput: (v) => {
					this.bubbleEdgeWidth = v
					this.uploadBubbleParams()
				},
			}),
		)

		container.appendChild(overlays.section)

		// Ledger
		this.initLedger()
	}

	private buildShadersWindow(container: HTMLElement) {
		// Hidden inputs for shader effect persistence
		const hiddenParticleEffect = document.createElement("input")
		hiddenParticleEffect.type = "hidden"
		hiddenParticleEffect.dataset.setting = "particleEffect"
		hiddenParticleEffect.value = this.activeParticleEffect.id
		hiddenParticleEffect.addEventListener("input", () => {
			if (hiddenParticleEffect.value !== this.activeParticleEffect.id) {
				this.switchParticleShader(hiddenParticleEffect.value)
				this.onParticleEffectChanged?.(hiddenParticleEffect.value)
			}
		})
		container.appendChild(hiddenParticleEffect)

		const hiddenPostEffect = document.createElement("input")
		hiddenPostEffect.type = "hidden"
		hiddenPostEffect.dataset.setting = "postEffect"
		hiddenPostEffect.value = this.activePostEffect.id
		hiddenPostEffect.addEventListener("input", () => {
			if (hiddenPostEffect.value !== this.activePostEffect.id) {
				this.switchPostShader(hiddenPostEffect.value)
				this.onPostEffectChanged?.(hiddenPostEffect.value)
			}
		})
		container.appendChild(hiddenPostEffect)

		this.onParticleEffectChanged = null
		this.onPostEffectChanged = null
		this._hiddenParticleEffect = hiddenParticleEffect
		this._hiddenPostEffect = hiddenPostEffect

		// Hidden inputs for per-effect param persistence
		const hiddenParticleParams = document.createElement("input")
		hiddenParticleParams.type = "hidden"
		hiddenParticleParams.dataset.setting = "particleEffectParams"
		hiddenParticleParams.value = JSON.stringify(this.particleEffectParams)
		hiddenParticleParams.addEventListener("input", () => {
			try {
				const parsed = JSON.parse(hiddenParticleParams.value)
				if (typeof parsed === "object" && parsed !== null) {
					this.particleEffectParams = parsed
					this.uploadRenderParams()
				}
			} catch {
				/* ignore corrupt data */
			}
		})
		container.appendChild(hiddenParticleParams)
		this._hiddenParticleParams = hiddenParticleParams

		const hiddenPostParams = document.createElement("input")
		hiddenPostParams.type = "hidden"
		hiddenPostParams.dataset.setting = "postEffectParams"
		hiddenPostParams.value = JSON.stringify(this.postEffectParams)
		hiddenPostParams.addEventListener("input", () => {
			try {
				const parsed = JSON.parse(hiddenPostParams.value)
				if (typeof parsed === "object" && parsed !== null) {
					this.postEffectParams = parsed
					this.uploadQuadParams()
				}
			} catch {
				/* ignore corrupt data */
			}
		})
		container.appendChild(hiddenPostParams)
		this._hiddenPostParams = hiddenPostParams
	}

	/* ================================================================ */
	/*  Detection ledger UI                                              */
	/* ================================================================ */

	private initLedger() {
		this.ledgerToggle = document.getElementById("ledger-toggle")
		this.ledgerPanels = document.getElementById("ledger-panels")
		this.ledgerBackdrop = document.getElementById("ledger-backdrop")
		this.ledgerOrganellesEl = document.getElementById("ledger-organelles")
		this.ledgerOrganismsEl = document.getElementById("ledger-organisms")
		this.ledgerPredictionsEl = document.getElementById("ledger-predictions")

		this.ledgerToggle?.classList.remove("hidden")

		this.ledgerToggle?.addEventListener("click", () => {
			const open = this.ledgerPanels?.classList.toggle("open")
			this.ledgerBackdrop?.classList.toggle("open", open)
		})
		this.ledgerBackdrop?.addEventListener("click", () => this.closeLedger())
		document.addEventListener("keydown", (e) => {
			if (e.key === "Escape" && this.ledgerPanels?.classList.contains("open"))
				this.closeLedger()
		})

		// Restore muted organisms from persisted settings
		const savedMuted = this.ledgerOrganismsEl?.dataset.mutedOrganisms
		if (savedMuted) {
			for (const sig of savedMuted.split(",").filter(Boolean))
				this.mutedOrganisms.add(sig)
		}

		// Mute button delegation on organism panel
		this.ledgerOrganismsEl?.addEventListener("click", (e) => {
			const btn = (e.target as HTMLElement).closest<HTMLElement>(".ledger-mute")
			if (!btn) return
			const sig = btn.dataset.sig
			if (!sig) return
			const wasMuted = this.mutedOrganisms.has(sig)
			if (wasMuted) {
				this.mutedOrganisms.delete(sig)
			} else {
				this.mutedOrganisms.add(sig)
			}
			this.syncMutedAttribute()
			this.updateLedgerUI()
		})
	}

	private syncMutedAttribute() {
		const muted = this.mutedOrganisms
		if (this.ledgerOrganismsEl) {
			if (muted.size > 0) {
				this.ledgerOrganismsEl.dataset.mutedOrganisms = [...muted].join(",")
			} else {
				delete this.ledgerOrganismsEl.dataset.mutedOrganisms
			}
		}
	}

	private closeLedger() {
		this.ledgerPanels?.classList.remove("open")
		this.ledgerBackdrop?.classList.remove("open")
	}

	private updateLedgerUI() {
		const orgEl = this.ledgerOrganellesEl
		const osmEl = this.ledgerOrganismsEl
		if (!orgEl || !osmEl) return

		const frame = this.detectionState
		if (!frame) {
			orgEl.innerHTML = ""
			osmEl.innerHTML = ""
			this.organelleRows.clear()
			this.organismRows.clear()
			this.organelleHeading = null
			this.organismHeading = null
			return
		}

		const types = this.getTypeIds()

		// ── Organelles ──────────────────────────────────────────────────
		const activeTypeIds = new Set<number>()
		if (frame.ledger.organellesByType.size > 0) {
			if (!this.organelleHeading) {
				this.organelleHeading = document.createElement("div")
				this.organelleHeading.className = "ledger-heading"
				this.organelleHeading.textContent = "Organelles"
				orgEl.prepend(this.organelleHeading)
			}

			const sorted = [...frame.ledger.organellesByType.entries()].sort(
				(a, b) => {
					const sa = frame.ledger.organelleStability.get(a[0]) ?? 0
					const sb = frame.ledger.organelleStability.get(b[0]) ?? 0
					return sb - sa || b[1] - a[1]
				},
			)

			for (const [typeId, count] of sorted) {
				activeTypeIds.add(typeId)
				let entry = this.organelleRows.get(typeId)
				if (!entry) {
					const row = document.createElement("div")
					row.className = "ledger-row"
					const dot = document.createElement("span")
					dot.className = "ledger-dot"
					const countEl = document.createElement("span")
					countEl.className = "ledger-count"
					row.append(dot, countEl)
					entry = { row, countEl }
					this.organelleRows.set(typeId, entry)
				}
				// Update color
				const groupId = types[typeId] ?? `t${typeId}`
				const rgb = this.groupColors.get(groupId) ?? [1, 1, 1]
				const hex = this.rgbToHex(rgb)
				const dot = entry.row.firstElementChild as HTMLElement
				if (dot.style.background !== hex) {
					dot.style.background = hex
					dot.style.color = hex
				}
				// Update count
				const countStr = String(count)
				if (entry.countEl.textContent !== countStr)
					entry.countEl.textContent = countStr
				// Insert into DOM only if not already there
				if (!entry.row.parentNode) orgEl.append(entry.row)
			}
		} else if (this.organelleHeading) {
			this.organelleHeading.remove()
			this.organelleHeading = null
		}

		// Remove stale organelle rows
		for (const [typeId, entry] of this.organelleRows) {
			if (!activeTypeIds.has(typeId)) {
				entry.row.remove()
				this.organelleRows.delete(typeId)
			}
		}

		// ── Organisms ───────────────────────────────────────────────────
		const activeSigs = new Set<string>()
		if (frame.ledger.organismsBySignature.size > 0) {
			if (!this.organismHeading) {
				this.organismHeading = document.createElement("div")
				this.organismHeading.className = "ledger-heading"
				const title = document.createElement("span")
				title.textContent = "Organisms"
				this.unmuteAllBtn = document.createElement("button")
				this.unmuteAllBtn.className = "ledger-unmute-all"
				this.unmuteAllBtn.textContent = "\u{1F50A}"
				this.unmuteAllBtn.title = "Unmute all"
				this.unmuteAllBtn.addEventListener("click", () => {
					this.mutedOrganisms.clear()
					this.syncMutedAttribute()
					this.updateLedgerUI()
				})
				this.organismHeading.append(title, this.unmuteAllBtn)
				osmEl.prepend(this.organismHeading)
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
				let entry = this.organismRows.get(sig)
				if (!entry) {
					const row = document.createElement("div")
					row.className = "ledger-row"
					const typeIds = sig.split("+").map(Number)
					for (const tid of typeIds) {
						const dot = document.createElement("span")
						dot.className = "ledger-dot"
						const gid = types[tid] ?? `t${tid}`
						const rgb = this.groupColors.get(gid) ?? [1, 1, 1]
						const hex = this.rgbToHex(rgb)
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
					this.organismRows.set(sig, entry)
				}
				// Update count
				const countStr = String(count)
				if (entry.countEl.textContent !== countStr)
					entry.countEl.textContent = countStr
				// Update mute state
				const muted = this.mutedOrganisms.has(sig)
				const wantClass = muted ? "ledger-mute muted" : "ledger-mute"
				if (entry.muteBtn.className !== wantClass)
					entry.muteBtn.className = wantClass
				const wantIcon = muted ? "\u{1F507}" : "\u{1F509}"
				if (entry.muteBtn.textContent !== wantIcon)
					entry.muteBtn.textContent = wantIcon
				const wantTitle = muted ? "Unmute" : "Mute"
				if (entry.muteBtn.title !== wantTitle) entry.muteBtn.title = wantTitle
				// Insert into DOM only if not already there
				if (!entry.row.parentNode) osmEl.append(entry.row)
			}
		} else if (this.organismHeading) {
			this.organismHeading.remove()
			this.organismHeading = null
		}

		// Remove stale organism rows
		for (const [sig, entry] of this.organismRows) {
			if (!activeSigs.has(sig)) {
				entry.row.remove()
				this.organismRows.delete(sig)
			}
		}

		// Show/hide unmute-all button
		if (this.unmuteAllBtn) {
			const anyMuted = this.mutedOrganisms.size > 0
			const wantDisplay = anyMuted ? "" : "none"
			if (this.unmuteAllBtn.style.display !== wantDisplay) {
				this.unmuteAllBtn.style.display = wantDisplay
			}
		}

		// ── Predicted Species ────────────────────────────────────────────
		this.updatePredictionLedger()
	}

	private updatePredictionLedger() {
		const predEl = this.ledgerPredictionsEl
		if (!predEl) return

		const prediction = this.organismPrediction
		if (!prediction || prediction.organisms.length === 0) {
			if (this.predictionHeading) {
				this.predictionHeading.remove()
				this.predictionHeading = null
			}
			if (this.speciesDecaySlider) {
				this.speciesDecaySlider.parentElement?.remove()
				this.speciesDecaySlider = null
			}
			for (const [, entry] of this.predictionRows) entry.row.remove()
			this.predictionRows.clear()
			this.speciesPresence.clear()
			this.speciesBrightness.clear()
			this.lastPredictionTime = 0
			return
		}

		// Timing for decay
		const now = performance.now()
		const dt =
			this.lastPredictionTime > 0
				? Math.min((now - this.lastPredictionTime) / 1000, 0.2) // seconds, capped
				: 0
		this.lastPredictionTime = now

		// Decay rate: how fast presence drains per second when not observed
		const decayRate = 0.15

		if (!this.predictionHeading) {
			this.predictionHeading = document.createElement("div")
			this.predictionHeading.className = "ledger-heading"
			this.predictionHeading.textContent = "Predicted Species"
			predEl.prepend(this.predictionHeading)
		}

		// Decay threshold slider
		if (!this.speciesDecaySlider) {
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
			slider.value = String(this.speciesDecayThreshold)
			slider.style.flex = "1"
			slider.style.height = "12px"
			slider.style.accentColor = "#888"
			slider.addEventListener("input", () => {
				this.speciesDecayThreshold = parseFloat(slider.value)
			})
			sliderRow.append(label, slider)
			this.speciesDecaySlider = slider
			predEl.prepend(sliderRow)
			predEl.prepend(this.predictionHeading)
		}

		// Cross-reference with observed organisms
		const observedSigs =
			this.detectionState?.ledger.organismsBySignature ?? new Map()

		// Update presence for all tracked signatures
		// Boost observed ones, decay unobserved ones
		const allTracked = new Set<string>()
		for (const org of prediction.organisms) allTracked.add(org.signature)
		for (const [sig] of this.speciesPresence) allTracked.add(sig)
		for (const [sig] of observedSigs) allTracked.add(sig)

		for (const sig of allTracked) {
			const prev = this.speciesPresence.get(sig) ?? 0
			const prevBright = this.speciesBrightness.get(sig) ?? 0
			const isObserved = observedSigs.has(sig)
			// Instant attack to 1 when observed, slow decay when not
			const next = isObserved ? 1 : prev - dt * decayRate
			const nextBright = isObserved
				? Math.min(1, prevBright + 0.15)
				: prevBright - dt * decayRate * 3
			if (next <= 0) {
				this.speciesPresence.delete(sig)
				this.speciesBrightness.delete(sig)
			} else {
				this.speciesPresence.set(sig, next)
				this.speciesBrightness.set(sig, Math.max(0, nextBright))
			}
		}

		// Build unified list: predicted + unpredicted, all with presence
		const predictedBysig = new Map(
			prediction.organisms.map((o) => [o.signature, o] as const),
		)
		const types = this.getTypeIds()

		// Collect all entries above threshold, sorted by presence (stability)
		const allEntries: {
			sig: string
			presence: number
			predicted: boolean
			typeKeys: ReadonlyArray<string>
		}[] = []

		for (const [sig, presence] of this.speciesPresence) {
			if (presence < this.speciesDecayThreshold && !observedSigs.has(sig))
				continue
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
		// Also include predicted entries that haven't been observed yet (no presence yet)
		for (const org of prediction.organisms) {
			if (this.speciesPresence.has(org.signature)) continue
			allEntries.push({
				sig: org.signature,
				presence: 0,
				predicted: true,
				typeKeys: org.typeKeys,
			})
		}

		allEntries.sort((a, b) => b.presence - a.presence)

		// Compute score range for arrow display (predicted entries only)
		const scores = prediction.organisms.map((o) => o.stabilityScore)
		const minScore = Math.min(...scores)
		const maxScore = Math.max(...scores)
		const scoreRange = maxScore - minScore

		const activeSigs = new Set<string>()
		for (const { sig, predicted, typeKeys } of allEntries) {
			activeSigs.add(sig)
			const entry = this.ensurePredictionRow(sig, typeKeys)

			if (predicted) {
				const org = predictedBysig.get(sig)!
				const t =
					scoreRange > 0 ? (org.stabilityScore - minScore) / scoreRange : 1
				const [arrow, color] =
					t >= 0.75
						? ["\u21c8", "#4caf50"]
						: t >= 0.4
							? ["\u2191", "#81c784"]
							: t >= 0.15
								? ["\u2193", "#e57373"]
								: ["\u21ca", "#f44336"]
				if (entry.scoreEl.textContent !== arrow)
					entry.scoreEl.textContent = arrow
				if (entry.scoreEl.style.color !== color)
					entry.scoreEl.style.color = color
			} else {
				if (entry.scoreEl.textContent !== "?") entry.scoreEl.textContent = "?"
				if (entry.scoreEl.style.color !== "#ffb74d")
					entry.scoreEl.style.color = "#ffb74d"
			}

			const brightness = this.speciesBrightness.get(sig) ?? 0
			const wantOpacity = String(Math.max(0.2, brightness))
			if (entry.row.style.opacity !== wantOpacity)
				entry.row.style.opacity = wantOpacity
			predEl.append(entry.row)
		}

		// Remove rows no longer active
		for (const [sig, entry] of this.predictionRows) {
			if (!activeSigs.has(sig)) {
				entry.row.remove()
				this.predictionRows.delete(sig)
			}
		}
	}

	private ensurePredictionRow(
		sig: string,
		typeKeys: ReadonlyArray<string>,
	): { row: HTMLElement; scoreEl: HTMLElement } {
		let entry = this.predictionRows.get(sig)
		if (!entry) {
			const row = document.createElement("div")
			row.className = "ledger-row"
			for (const tk of typeKeys) {
				const dot = document.createElement("span")
				dot.className = "ledger-dot"
				const rgb = this.groupColors.get(tk) ?? [1, 1, 1]
				const hex = this.rgbToHex(rgb)
				dot.style.background = hex
				dot.style.color = hex
				row.append(dot)
			}
			const scoreEl = document.createElement("span")
			scoreEl.className = "ledger-likelihood"
			row.append(scoreEl)
			entry = { row, scoreEl }
			this.predictionRows.set(sig, entry)
		}
		return entry
	}

	/* ================================================================ */
	/*  UI helpers (unchanged from WebGL2 version)                       */
	/* ================================================================ */

	private buildTypeRow(
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
			: (this.groupColors.get(type) ?? [1, 1, 1])

		const row = document.createElement("div")
		row.className = "particle-type-row"

		const deleteBtn = document.createElement("button")
		deleteBtn.className = "particle-card-delete"
		deleteBtn.textContent = "\u00d7"
		deleteBtn.title = "Delete particle type"
		deleteBtn.addEventListener("click", (e) => {
			e.stopPropagation()
			const removeIndices: number[] = []
			for (let i = 0; i < this.particles.length; i++) {
				if (this.particles[i].groupId === type) removeIndices.push(i)
			}
			this.removeParticlesByIndices(removeIndices)
			this.groupNames.delete(type)
			this.groupColors.delete(type)
			this.forceMatrixDirty = true

			const types = this.getTypeIds()
			this.forceMatrix = resizeMatrix(this.forceMatrix, types)
			// audio types updated via bar-boundary snapshot

			const openSections = this.getOpenSections(container)
			container.innerHTML = ""
			const rebuild = rebuildFn
			rebuild(container)
			this.restoreOpenSections(container, openSections)
			container.dispatchEvent(new Event("change", { bubbles: true }))
		})
		row.appendChild(deleteBtn)

		const picker = new ColorPicker(this.rgbToHex(color))
		picker.element.classList.add("particle-type-swatch")
		picker.input.dataset.setting = `particle:${type}:color`
		picker.onChange((hex) => {
			const rgb = this.hexToRgb(hex)
			this.groupColors.set(type, rgb)
			for (const p of this.particles) {
				if (p.groupId === type) {
					p.color[0] = rgb[0]
					p.color[1] = rgb[1]
					p.color[2] = rgb[2]
				}
			}
			this.uploadParticleColors()
			this.syncMatrixHeaders(container)
			// color updates flow through bar-boundary snapshot
		})
		row.appendChild(picker.element)

		const nameInput = document.createElement("input")
		nameInput.type = "text"
		nameInput.className = "particle-type-name"
		nameInput.value = this.groupNames.get(type) || type
		nameInput.dataset.setting = `particle:${type}:name`
		nameInput.addEventListener("input", () => {
			this.groupNames.set(type, nameInput.value)
			this.syncMatrixHeaders(container)
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
			const currentOfType = this.particles.filter((p) => p.groupId === type)
			const currentCount = currentOfType.length

			if (desired === currentCount) return

			if (desired > currentCount) {
				const startIdx = this.particles.length
				const toAdd = Math.min(
					desired - currentCount,
					MAX_PARTICLES - this.particles.length,
				)
				if (toAdd <= 0) return
				const liveColor = this.groupColors.get(type) ?? color
				for (let i = 0; i < toAdd; i++) {
					this.particles.push(
						new CustomParticle(
							Math.random() * this.width,
							Math.random() * this.height,
							type,
							[liveColor[0], liveColor[1], liveColor[2]],
						),
					)
				}
				this.count = this.particles.length
				this.uploadParticleRange(startIdx, toAdd)
			} else {
				const removeIndices: number[] = []
				for (
					let i = this.particles.length - 1;
					i >= 0 && removeIndices.length < currentCount - desired;
					i--
				) {
					if (this.particles[i].groupId === type) removeIndices.push(i)
				}
				this.removeParticlesByIndices(removeIndices)
			}

			members.length = 0
			for (const p of this.particles) {
				if (p.groupId === type) members.push(p)
			}
		})
		row.appendChild(countInput)

		section.appendChild(row)
	}

	private buildMatrixUI(container: HTMLElement, rootContainer?: HTMLElement) {
		const types = this.getTypeIds()
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
		randomBtn.title = "Randomize force matrix"
		randomBtn.addEventListener("click", () => {
			this.forceMatrix = randomizeMatrix(types)
			this.forceMatrixDirty = true
			this.speciesPresence.clear()
			this.speciesBrightness.clear()
			this.syncMatrixUI(wrapper, types)
			this.syncMatrixHidden(container)

			const queryRoot = rootContainer ?? container
			queryRoot.dispatchEvent(new Event("change", { bubbles: true }))
		})
		headerRow.appendChild(randomBtn)

		const clearBtn = document.createElement("button")
		clearBtn.className = "force-matrix-clear"
		clearBtn.textContent = "Clear"
		clearBtn.title = "Clear force matrix"
		clearBtn.addEventListener("click", () => {
			this.forceMatrix = emptyMatrix(types)
			this.forceMatrixDirty = true
			this.syncMatrixUI(wrapper, types)
			this.syncMatrixHidden(container)
			container.dispatchEvent(new Event("change", { bubbles: true }))
		})
		headerRow.appendChild(clearBtn)

		wrapper.appendChild(headerRow)

		const autoToggle = document.createElement("div")
		autoToggle.className = "control-group"
		const autoLabel = document.createElement("label")
		autoLabel.textContent = "Auto Randomize"
		autoToggle.appendChild(autoLabel)
		const autoCheckbox = document.createElement("input")
		autoCheckbox.type = "checkbox"
		autoCheckbox.checked = this.autoRandomizeMatrixEnabled
		autoCheckbox.dataset.setting = "autoRandomize"
		autoCheckbox.addEventListener("change", () => {
			this.autoRandomizeMatrixEnabled = autoCheckbox.checked
		})
		autoToggle.appendChild(autoCheckbox)
		const autoRandomizeClock = document.createElement("mini-clock") as MiniClock
		autoToggle.appendChild(autoRandomizeClock)
		this._autoRandomizeMatrixClock = autoRandomizeClock
		wrapper.appendChild(autoToggle)

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
			const [r, g, b] = this.getTypeColor(tgt)
			hdr.innerHTML = `<span class="force-matrix-swatch" style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})"></span>`
			hdr.title = this.groupNames.get(tgt) || tgt
			grid.appendChild(hdr)
		}

		// Rows
		for (const src of types) {
			const rowHdr = document.createElement("div")
			rowHdr.className = "force-matrix-header-cell"
			const [r, g, b] = this.getTypeColor(src)
			rowHdr.innerHTML = `<span class="force-matrix-swatch" style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})"></span>`
			rowHdr.title = this.groupNames.get(src) || src
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
				input.value = String(this.forceMatrix[src]?.[tgt] ?? 0)
				this.colorizeCell(input)

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
						updated[s] = { ...this.forceMatrix[s] }
					}
					updated[src][tgt] = val
					this.forceMatrix = updated
					this.forceMatrixDirty = true
					this.colorizeCell(input)
					this.syncMatrixHidden(container)
				})

				grid.appendChild(input)
			}
		}

		wrapper.appendChild(grid)

		this._matrixWrapper = wrapper
		this._matrixContainer = container
		this._matrixRootContainer = rootContainer ?? container

		container.appendChild(wrapper)
	}

	private colorizeCell(input: HTMLInputElement) {
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

	private syncMatrixUI(wrapper: HTMLElement, types: readonly string[]) {
		for (const src of types) {
			for (const tgt of types) {
				const input = wrapper.querySelector<HTMLInputElement>(
					`input[data-src="${src}"][data-tgt="${tgt}"]`,
				)
				if (input) {
					input.value = String(this.forceMatrix[src]?.[tgt] ?? 0)
					this.colorizeCell(input)
				}
			}
		}
	}

	private syncMatrixHeaders(container: HTMLElement) {
		const matrixWrapper = container.querySelector(".force-matrix-container")
		if (!matrixWrapper) return
		const headers = matrixWrapper.querySelectorAll<HTMLElement>(
			".force-matrix-header-cell",
		)
		const types = this.getTypeIds()
		for (let i = 0; i < types.length; i++) {
			const type = types[i]
			const [r, g, b] = this.getTypeColor(type)
			const colorStr = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`
			const name = this.groupNames.get(type) || type
			// Column header (first N header-cells)
			const colHdr = headers[i]
			if (colHdr) {
				const sw = colHdr.querySelector<HTMLElement>(".force-matrix-swatch")
				if (sw) sw.style.background = colorStr
				colHdr.title = name
			}
			// Row header (next N header-cells)
			const rowHdr = headers[types.length + i]
			if (rowHdr) {
				const sw = rowHdr.querySelector<HTMLElement>(".force-matrix-swatch")
				if (sw) sw.style.background = colorStr
				rowHdr.title = name
			}
		}
	}

	private syncMatrixHidden(container: HTMLElement) {
		const hidden = container.querySelector<HTMLInputElement>(
			'input[data-setting="forceMatrix"]',
		)
		if (hidden) {
			hidden.value = matrixToJSON(this.forceMatrix)
		}
	}

	private getTypeIds(): string[] {
		const seen = new Set<string>()
		const result: string[] = []
		// Include all registered types (preserving registration order)
		for (const type of this.groupNames.keys()) {
			if (!seen.has(type)) {
				seen.add(type)
				result.push(type)
			}
		}
		// Include any particle types not yet registered (shouldn't happen, but safe)
		for (const p of this.particles) {
			if (!seen.has(p.groupId)) {
				seen.add(p.groupId)
				result.push(p.groupId)
			}
		}
		return result
	}

	/** Randomize particle counts per type and sync UI. */
	private randomizeCounts() {
		const types = this.getTypeIds()
		const allRemoveIndices: number[] = []
		const pendingAdds: Array<{ type: string; count: number }> = []

		for (const type of types) {
			const desired =
				Math.random() < 0.1 ? 0 : Math.round(400 + Math.random() * 1100)
			const currentCount = this.particles.filter(
				(p) => p.groupId === type,
			).length

			if (desired > currentCount) {
				pendingAdds.push({ type, count: desired - currentCount })
			} else if (desired < currentCount) {
				let toRemove = currentCount - desired
				for (let i = this.particles.length - 1; i >= 0 && toRemove > 0; i--) {
					if (this.particles[i].groupId === type) {
						allRemoveIndices.push(i)
						toRemove--
					}
				}
			}
		}

		if (allRemoveIndices.length > 0) {
			this.removeParticlesByIndices(allRemoveIndices)
		}

		for (const { type, count } of pendingAdds) {
			const typeColor = this.getTypeColor(type)
			const startIdx = this.particles.length
			const capped = Math.min(count, MAX_PARTICLES - this.particles.length)
			if (capped <= 0) break
			for (let i = 0; i < capped; i++) {
				this.particles.push(
					new CustomParticle(
						Math.random() * this.width,
						Math.random() * this.height,
						type,
						[typeColor[0], typeColor[1], typeColor[2]],
					),
				)
			}
			this.count = this.particles.length
			this.uploadParticleRange(startIdx, capped)
		}

		if (this._particlesContainer) {
			for (const type of types) {
				const countInput =
					this._particlesContainer.querySelector<HTMLInputElement>(
						`[data-setting="particle:${type}:count"]`,
					)
				if (countInput) {
					const count = this.particles.filter((p) => p.groupId === type).length
					countInput.value = String(count)
				}
			}
			this._particlesContainer.dispatchEvent(
				new Event("change", { bubbles: true }),
			)
		}
	}

	/** Returns the current phrase cycle position (0-based bar index), or -1 if unavailable. */
	private phrasePosition(barNumber: number): { idx: number; len: number } {
		const origin = this.bassLayer.cycleOrigin
		const phrase = expandPhrase(this.phrasePattern, this.phraseMirror)
		const len = phrase.length
		if (len === 0 || origin == null) return { idx: -1, len: 0 }
		return { idx: (((barNumber - origin) % len) + len) % len, len }
	}

	private getTypeColor(type: string): [number, number, number] {
		const gc = this.groupColors.get(type)
		if (gc) return gc
		for (const p of this.particles) {
			if (p.groupId === type) return p.color
		}
		return [1, 1, 1]
	}

	/* ================================================================ */
	/*  teardown / cleanup                                               */
	/* ================================================================ */

	teardown() {
		this.cleanup()
	}

	private cleanup() {
		if (this.canvas) {
			if (this.boundMouseMove)
				this.canvas.removeEventListener("mousemove", this.boundMouseMove)
			if (this.boundMouseDown)
				this.canvas.removeEventListener("mousedown", this.boundMouseDown)
			if (this.boundContextMenu)
				this.canvas.removeEventListener("contextmenu", this.boundContextMenu)
		}
		if (this.boundMouseUp)
			window.removeEventListener("mouseup", this.boundMouseUp)
		this.boundMouseMove = null
		this.boundMouseDown = null
		this.boundMouseUp = null
		this.boundContextMenu = null
		this.mouseLeft = false
		this.mouseRight = false

		this.closeLedger()
		this.ledgerToggle?.classList.add("hidden")
		if (this.ledgerOrganellesEl) this.ledgerOrganellesEl.innerHTML = ""
		if (this.ledgerOrganismsEl) this.ledgerOrganismsEl.innerHTML = ""
		this.organelleRows.clear()
		this.organismRows.clear()
		this.organelleHeading = null
		this.organismHeading = null
		this.unmuteAllBtn = null
		this.ledgerToggle = null
		this.ledgerPanels = null
		this.ledgerBackdrop = null
		this.ledgerOrganellesEl = null
		this.ledgerOrganismsEl = null
		this._matrixWrapper = null
		this._matrixContainer = null
		this._matrixRootContainer = null
		this._particlesContainer = null
		this._autoRandomizeMatrixClock = null
		this._autoRandomizeCountsClock = null

		this.bassLayer.dispose()
		this.audioGraph.dispose()
		this.readbackBuffer?.destroy()
		this.readbackBuffer = null
		this.detectionBuffer?.destroy()
		this.detectionBuffer = null
		this.radiusScaleBuffer?.destroy()
		this.radiusScaleBuffer = null
		this.activePulses.clear()
		this.readbackPending = false

		this.particleBuffers[0]?.destroy()
		this.particleBuffers[1]?.destroy()
		this.particleStagingBuffer?.destroy()
		this.simParamsBuffer?.destroy()
		this.forceMatrixBuffer?.destroy()
		this.stressBuffer?.destroy()
		this.renderParamsBuffer?.destroy()
		this.circleRenderParamsBuffer?.destroy()
		this.detectionFillParamsBuffer?.destroy()
		this.organismFillParamsBuffer?.destroy()
		this.organismCentroidBuffer?.destroy()
		this.osmLevelCentroidBuffer?.destroy()
		this.organismCentroidParamsBuffer?.destroy()
		this.organismLineBuffer?.destroy()
		this.quadParamsBuffer?.destroy()
		this.stainParamsBuffer?.destroy()
		for (const t of this.stainTextures) t?.destroy()
		this.falloffTexture?.destroy()
		this.offscreenTexture?.destroy()
		this.detectionIdTexture?.destroy()
		this.organismIdTexture?.destroy()
		this.organismDepthTexture?.destroy()

		this.particleBuffers = [null, null]
		this.particleStagingBuffer = null
		this.computeBindGroups = [null, null]
		this.particleRenderBindGroups = [null, null]
		this.circleRenderBindGroups = [null, null]
		this.simParamsBuffer = null
		this.forceMatrixBuffer = null
		this.stressBuffer = null
		this.renderParamsBuffer = null
		this.circleRenderParamsBuffer = null
		this.detectionFillParamsBuffer = null
		this.detectionFillBindGroups = [null, null]
		this.detectionFillPipeline = null
		this.detectionEdgePipeline = null
		this.quadParamsBuffer = null
		this.stainParamsBuffer = null
		this.stainTextures = [null, null]
		this.stainViews = [null, null]
		this.stainPipeline = null
		this.stainBindGroupLayout = null
		this.stainBindGroups = [null, null]
		this.stainPingPong = 0
		this.falloffTexture = null
		this.falloffSampler = null
		this.offscreenTexture = null
		this.offscreenView = null
		this.detectionIdTexture = null
		this.detectionColorTexture = null
		this.detectionEdgeBindGroupLayout = null
		this.detectionEdgeBindGroup = null
		this.organismIdTexture = null
		this.organismDepthTexture = null
		this.organismFillPipeline = null
		this.organismFillParamsBuffer = null
		this.organismFillBindGroups = [null, null]
		this.organismCentroidPipeline = null
		this.organismCentroidBindGroupLayout = null
		this.organismCentroidBindGroup = null
		this.organismCentroidBuffer = null
		this.organismCentroidParamsBuffer = null
		this.organismCentroidCount = 0
		this.organismCentroidSnapshot = []
		this.osmLevelCentroidPipeline = null
		this.osmLevelCentroidBindGroup = null
		this.osmLevelCentroidBuffer = null
		this.osmLevelCentroidCount = 0
		this.osmLevelCentroidSnapshot = []
		this.organismLinePipeline = null
		this.organismLineFillPipeline = null
		this.organismLineBuffer = null
		this.organismLineBindGroup = null
		this.organismLineCount = 0
		this.organismLineEdges = []
		this.organismCentroidFillPipeline = null
		this.organismEdgePipeline = null
		this.organismEdgeBindGroupLayout = null
		this.organismEdgeBindGroup = null
		for (const t of this.jfaOrganelleTextures) t?.destroy()
		for (const t of this.jfaOrganismTextures) t?.destroy()
		this.jfaOrganelleTextures = [null, null]
		this.jfaOrganismTextures = [null, null]
		for (const buf of this.jfaParamsBuffers) buf.destroy()
		this.jfaParamsBuffers = []
		this.jfaComputePipeline = null
		this.jfaComputeBindGroupLayout = null
		this.jfaOrganelleBindGroups = [[], []]
		this.jfaOrganismBindGroups = [[], []]
		this.jfaOrganelleEdgeBindGroups = [null, null]
		this.jfaOrganismEdgeBindGroups = [null, null]
		this.jfaEdgeBindGroupLayout = null
		this.jfaOrganelleEdgePipeline = null
		this.jfaOrganismEdgePipeline = null
		this.jfaOrganelleSeedPipeline = null
		this.jfaOrganismSeedPipeline = null
		this.jfaOrganismCentroidSeedPipeline = null
		this.jfaOrganismLineSeedPipeline = null
		this.bubbleParamsBuffer?.destroy()
		this.bubbleParamsBuffer = null
		this.quadBindGroups = [null, null]
		this.computePipeline = null
		this.peakUpdatePipeline = null
		this.particleRenderPipeline = null
		this.circleRenderPipeline = null
		this.prevVelX = null
		this.prevVelY = null
		this.quadPipeline = null

		if (this.curveEditor) {
			this.curveEditor.destroy()
			this.curveEditor = null
		}
	}

	private readSavedParticleConfig(): {
		types: {
			type: string
			name: string
			count: number
			color: [number, number, number]
		}[]
		matrix: string | null
	} | null {
		try {
			const raw = localStorage.getItem("particle-sim:settings:" + this.name)
			if (!raw) return null
			const parsed = JSON.parse(raw)
			const envelope =
				"data" in parsed && typeof parsed.data === "object"
					? parsed
					: { data: parsed }
			if (envelope.version !== this.settingsVersion) return null

			const data: Record<string, string> = envelope.data
			const types: {
				type: string
				name: string
				count: number
				color: [number, number, number]
			}[] = []

			for (const key of Object.keys(data)) {
				const match = key.match(/^particle:(\w+):count$/)
				if (!match) continue
				const typeName = match[1]
				const count = Math.max(0, Math.min(5000, Number(data[key]) || 0))

				const name = data[`particle:${typeName}:name`] || typeName
				const colorHex = data[`particle:${typeName}:color`]
				const color: [number, number, number] = colorHex
					? this.hexToRgb(colorHex)
					: [1, 1, 1]

				types.push({ type: typeName, name, count, color })
			}

			const matrix = data["forceMatrix"] ?? null

			return types.length > 0 ? { types, matrix } : null
		} catch {
			return null
		}
	}

	private rgbToHex(rgb: [number, number, number]): string {
		return (
			"#" +
			rgb
				.map((v) =>
					Math.round(v * 255)
						.toString(16)
						.padStart(2, "0"),
				)
				.join("")
		)
	}

	private hexToRgb(hex: string): [number, number, number] {
		const r = parseInt(hex.slice(1, 3), 16) / 255
		const g = parseInt(hex.slice(3, 5), 16) / 255
		const b = parseInt(hex.slice(5, 7), 16) / 255
		return [r, g, b]
	}

	private generateGroupId(): string {
		return "p" + this.nextGroupId++
	}

	private generateName(): string {
		const existingNames = new Set(this.groupNames.values())
		if (!existingNames.has("particle")) return "particle"
		let i = 1
		while (existingNames.has(`particle ${i}`)) i++
		return `particle ${i}`
	}
}
