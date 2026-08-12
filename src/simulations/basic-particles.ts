import { autoBalance } from "../auto-balance"
import {
    DEFAULT_DETECTION_CONFIG,
    DENSITY_TARGET,
    MAX_PARTICLES,
    MAX_TYPES,
    MIN_PER_ACTIVE_TYPE,
    PARTICLE_STRIDE,
    WORKGROUP_SIZE,
} from "../constants"
import { ColorPicker } from "../ui/color-picker"
import { CurveEditor } from "../ui/curve-editor"
import {
    toroidalDelta,
    updateRegistry,
    type DetectionConfig,
    type DetectionFrame,
    type OrganelleState,
    type OrganelleTreeNode,
    type OrganismRegistry,
} from "../detection"
import { deserializeDetectionFrame } from "../detection/serialization"
import type {
    DetectionFrameWire,
    DetectionWorkerRequest,
    DetectionWorkerResponse,
} from "../detection/worker-types"
import { applyStepDelta } from "../ui/number-scroll"
import {
    predictOrganisms,
    type OrganismPrediction,
} from "../detection/organism-prediction"
import { CustomParticle, type ForceMatrix } from "../particles"
import type { GpuContext, Simulation, WindowDefinition } from "../types"
import { createNumberGroup } from "../ui/ui-helpers"
import type { VuMeter } from "../ui/widgets/vu-meter"
import type { MiniGauge } from "../ui/widgets/mini-gauge"

import { MusicEngine } from "../music2"

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
import { buildDisplayWindow } from "./windows/display"
import { buildPhysicsWindow } from "./windows/physics"
import { buildParticlesWindow } from "./windows/particles"
import { buildMusicWindow } from "./windows/music"
import { buildDetectionWindow } from "./windows/detection"
import { buildShadersWindow } from "./windows/shaders"
import { closeLedger, updateLedgerUI } from "./ledger"
import { syncMatrixHidden, syncMatrixUI } from "./force-matrix-ui"
import detectionFillFrag from "../shaders/basic-particles/detection-fill.frag.wgsl?raw"
import detectionEdgeSrc from "../shaders/basic-particles/detection-edge.wgsl?raw"
import organismFillFrag from "../shaders/basic-particles/organism-fill.frag.wgsl?raw"
import centroidPrefix from "../shaders/basic-particles/centroid-prefix.wgsl?raw"
import centroidVisualFrag from "../shaders/basic-particles/centroid-visual.frag.wgsl?raw"
import centroidThinRingFrag from "../shaders/basic-particles/centroid-thin-ring.frag.wgsl?raw"
import centroidFillPrefixSrc from "../shaders/basic-particles/centroid-fill-prefix.wgsl?raw"
import centroidFillFrag from "../shaders/basic-particles/centroid-fill.frag.wgsl?raw"
import linePrefix from "../shaders/basic-particles/line-prefix.wgsl?raw"
import lineVisualFrag from "../shaders/basic-particles/line-visual.frag.wgsl?raw"
import lineFillPrefixSrc from "../shaders/basic-particles/line-fill-prefix.wgsl?raw"
import lineFillFrag from "../shaders/basic-particles/line-fill.frag.wgsl?raw"
import organismEdgeSrc from "../shaders/basic-particles/organism-edge.wgsl?raw"
import jfaOrganelleSeedFrag from "../shaders/basic-particles/jfa-organelle-seed.frag.wgsl?raw"
import jfaOrganismSeedFrag from "../shaders/basic-particles/jfa-organism-seed.frag.wgsl?raw"
import centroidSeedFrag from "../shaders/basic-particles/centroid-seed.frag.wgsl?raw"
import lineSeedFrag from "../shaders/basic-particles/line-seed.frag.wgsl?raw"
import jfaEdgePrefix from "../shaders/basic-particles/jfa-edge-prefix.wgsl?raw"
import jfaOrganelleEdgeFrag from "../shaders/basic-particles/jfa-organelle-edge.frag.wgsl?raw"
import jfaOrganismEdgeFrag from "../shaders/basic-particles/jfa-organism-edge.frag.wgsl?raw"

/* ------------------------------------------------------------------ */
/*  GPU pipeline helpers — reduce repeated blend/target boilerplate.   */
/* ------------------------------------------------------------------ */

/** Standard alpha blend: `src * srcAlpha + dst * (1 - srcAlpha)`. Used by every
 *  overlay pipeline that composites onto the canvas. */
const ALPHA_BLEND: GPUBlendState = {
	color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
	alpha: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
}

/** Depth state for organism/centroid/line fill pipelines that rank overlaps by depth. */
const DEPTH_LESS_EQUAL: GPUDepthStencilState = {
	format: "depth24plus",
	depthWriteEnabled: true,
	depthCompare: "less-equal",
}

/* ------------------------------------------------------------------ */
/*  Pure helpers for the force matrix                                  */
/* ------------------------------------------------------------------ */

export function emptyMatrix(types: readonly string[]): ForceMatrix {
	const m: Record<string, Record<string, number>> = {}
	for (const src of types) {
		const row: Record<string, number> = {}
		for (const tgt of types) row[tgt] = 0
		m[src] = row
	}
	return m
}

export function randomizeMatrix(types: readonly string[]): ForceMatrix {
	const m: Record<string, Record<string, number>> = {}
	for (const src of types) {
		const row: Record<string, number> = {}
		for (const tgt of types)
			row[tgt] = Math.round((Math.random() * 2 - 1) * 100) / 100
		m[src] = row
	}
	return m
}

export function resizeMatrix(
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

export function matrixToJSON(matrix: ForceMatrix): string {
	return JSON.stringify(matrix)
}

export function matrixFromJSON(json: string, types: readonly string[]): ForceMatrix {
	try {
		const raw = JSON.parse(json) as Record<string, Record<string, number>>
		return resizeMatrix(raw, types)
	} catch {
		return emptyMatrix(types)
	}
}

/* ------------------------------------------------------------------ */
/*  Simulation                                                         */
/* ------------------------------------------------------------------ */

export class RandomDots implements Simulation {
	name = "Random Dots"
	settingsVersion = "2026-08-12-per-type-progression-v3"

	// GPU resources
	device: GPUDevice | null = null
	canvas: HTMLCanvasElement | null = null
	canvasContext: GPUCanvasContext | null = null
	canvasFormat: GPUTextureFormat = "bgra8unorm"

	// Compute pipeline
	computePipeline: GPUComputePipeline | null = null
	computeBindGroupLayout: GPUBindGroupLayout | null = null
	computeBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	particleBuffers: [GPUBuffer | null, GPUBuffer | null] = [null, null]
	particleStagingBuffer: GPUBuffer | null = null
	simParamsBuffer: GPUBuffer | null = null
	forceMatrixBuffer: GPUBuffer | null = null
	stressBuffer: GPUBuffer | null = null
	peakUpdatePipeline: GPUComputePipeline | null = null
	pingPong = 0

	// Render pipeline (particles)
	particleRenderPipeline: GPURenderPipeline | null = null
	particleRenderBindGroupLayout: GPUBindGroupLayout | null = null
	particleRenderBindGroups: [GPUBindGroup | null, GPUBindGroup | null] =
		[null, null]
	renderParamsBuffer: GPUBuffer | null = null
	falloffTexture: GPUTexture | null = null
	falloffSampler: GPUSampler | null = null

	// Render pipeline (circle overlay) — same shaders, different blend + mode
	circleRenderPipeline: GPURenderPipeline | null = null
	circleRenderParamsBuffer: GPUBuffer | null = null
	circleRenderBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]

	// Detection pipelines: fill organelle IDs to R8Uint + color to RGBA8, edge-detect outlines
	detectionIdTexture: GPUTexture | null = null
	detectionColorTexture: GPUTexture | null = null
	detectionFillPipeline: GPURenderPipeline | null = null
	detectionEdgePipeline: GPURenderPipeline | null = null
	detectionFillParamsBuffer: GPUBuffer | null = null
	detectionFillBindGroups: [GPUBindGroup | null, GPUBindGroup | null] =
		[null, null]
	detectionEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	detectionEdgeBindGroup: GPUBindGroup | null = null

	// Organism outline pipelines: fill organism IDs to R8Uint (inflated particles), edge-detect white outlines
	organismIdTexture: GPUTexture | null = null
	organismFillPipeline: GPURenderPipeline | null = null
	organismFillParamsBuffer: GPUBuffer | null = null
	organismFillBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	organismEdgePipeline: GPURenderPipeline | null = null
	organismEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	organismEdgeBindGroup: GPUBindGroup | null = null

	// JFA bubble boundary system
	jfaOrganelleTextures: [GPUTexture | null, GPUTexture | null] = [
		null,
		null,
	]
	jfaOrganismTextures: [GPUTexture | null, GPUTexture | null] = [
		null,
		null,
	]
	jfaComputePipeline: GPUComputePipeline | null = null
	jfaComputeBindGroupLayout: GPUBindGroupLayout | null = null
	jfaParamsBuffers: GPUBuffer[] = []
	jfaOrganelleBindGroups: [GPUBindGroup[], GPUBindGroup[]] = [[], []]
	jfaOrganismBindGroups: [GPUBindGroup[], GPUBindGroup[]] = [[], []]
	jfaPassCount = 0
	jfaOrganelleEdgeBindGroups: [
		GPUBindGroup | null,
		GPUBindGroup | null,
	] = [null, null]
	jfaOrganismEdgeBindGroups: [
		GPUBindGroup | null,
		GPUBindGroup | null,
	] = [null, null]
	jfaEdgeBindGroupLayout: GPUBindGroupLayout | null = null
	jfaOrganelleEdgePipeline: GPURenderPipeline | null = null
	jfaOrganismEdgePipeline: GPURenderPipeline | null = null
	jfaOrganelleSeedPipeline: GPURenderPipeline | null = null
	jfaOrganismSeedPipeline: GPURenderPipeline | null = null
	jfaOrganismCentroidSeedPipeline: GPURenderPipeline | null = null
	jfaOrganismLineSeedPipeline: GPURenderPipeline | null = null
	bubbleParamsBuffer: GPUBuffer | null = null
	bubbleThreshold = 5
	bubbleEdgeWidth = 3

	// Organism centroid circle overlay
	organismCentroidPipeline: GPURenderPipeline | null = null
	organismCentroidBindGroupLayout: GPUBindGroupLayout | null = null
	organismCentroidBindGroup: GPUBindGroup | null = null
	organismCentroidBuffer: GPUBuffer | null = null
	organismCentroidParamsBuffer: GPUBuffer | null = null
	organismCentroidCount = 0
	organismCentroidSnapshot: {
		cx: number
		cy: number
		vx: number
		vy: number
		id: number
	}[] = []
	organismCentroidSnapshotTime = 0

	// Organism connection lines (edges between organelle centroids in same organism)
	organismLinePipeline: GPURenderPipeline | null = null
	organismLineFillPipeline: GPURenderPipeline | null = null
	organismLineBuffer: GPUBuffer | null = null
	organismLineBindGroup: GPUBindGroup | null = null
	organismLineCount = 0
	organismLineEdges: [number, number][] = [] // pairs of indices into centroid snapshot

	// Fill-variant pipeline for centroid circles (writes organism ID to r8uint)
	organismCentroidFillPipeline: GPURenderPipeline | null = null

	// Organism-level centroid circles (larger, wrapping organelle centroids)
	osmLevelCentroidPipeline: GPURenderPipeline | null = null
	osmLevelCentroidBindGroup: GPUBindGroup | null = null
	osmLevelCentroidBuffer: GPUBuffer | null = null
	osmLevelCentroidCount = 0
	osmLevelCentroidSnapshot: {
		cx: number
		cy: number
		vx: number
		vy: number
		id: number
	}[] = []
	osmLevelCentroidSnapshotTime = 0

	// Quad (fullscreen normalization) pipeline
	quadPipeline: GPURenderPipeline | null = null
	quadBindGroupLayout: GPUBindGroupLayout | null = null
	quadBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	quadParamsBuffer: GPUBuffer | null = null
	offscreenTexture: GPUTexture | null = null
	offscreenView: GPUTextureView | null = null
	offscreenSampler: GPUSampler | null = null

	// Stain (phosphor persistence) — ping-pong pair
	stainTextures: [GPUTexture | null, GPUTexture | null] = [null, null]
	stainViews: [GPUTextureView | null, GPUTextureView | null] = [
		null,
		null,
	]
	stainPingPong = 0
	stainPipeline: GPURenderPipeline | null = null
	stainBindGroupLayout: GPUBindGroupLayout | null = null
	stainBindGroups: [GPUBindGroup | null, GPUBindGroup | null] = [
		null,
		null,
	]
	stainParamsBuffer: GPUBuffer | null = null

	// Simulation state (CPU-side, for UI and buffer uploads)
	count = 500
	width = 0
	height = 0
	particles: CustomParticle[] = []
	nextGroupId = 0
	groupNames = new Map<string, string>()
	groupColors = new Map<string, [number, number, number]>()
	showCircleOverlay = false
	prevVelX: Float32Array | null = null // previous-frame per-particle vx
	prevVelY: Float32Array | null = null // previous-frame per-particle vy
	pointSize = 27.0
	pulseScale = 6
	curveEditor: CurveEditor | null = null

	// Per-effect shader params (effectId → [param0, param1, ...])
	particleEffectParams: Record<string, number[]> = {
		gradient: [1.1, 0, 0, 0],
		solid: [1, 0.4, 0, 0],
		"speed-color": [0.66, 0.5, 0, 0],
		"stress-color": [1, 0.65, 0, 0],
	}
	postEffectParams: Record<string, number[]> = {
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
	forceMatrix: ForceMatrix = {}
	affectRadius = 61.1
	forceRepelDistance = 40.72
	baseStrength = 207.94

	// Density regulation (user-facing controls)
	crowdLimit = 29.06
	spread = 26 // 0–100%
	maxSpeedPct = 100 // 1–100%, soft speed limiter

	// Universal repulsion between all particles
	repelStrength = 147.61

	// Scale: spatial zoom multiplier (1 = default, 2 = everything twice as large)
	scale = 0.5

	// Auto-balance: derive physics params from force matrix + particle counts
	autoBalanceEnabled = true

	// Accumulated time for animated shaders
	time = 0

	// Active shader effects (one per category)
	activeParticleEffect: ShaderEffect = particleEffects[0]
	activePostEffect: ShaderEffect = findPostEffect("metaball")

	// Callbacks for shader menu sync
	onParticleEffectChanged: ((id: string) => void) | null = null
	onPostEffectChanged: ((id: string) => void) | null = null

	// Hidden inputs for settings persistence
	_hiddenParticleEffect: HTMLInputElement | null = null
	_hiddenPostEffect: HTMLInputElement | null = null
	_hiddenParticleParams: HTMLInputElement | null = null
	_hiddenPostParams: HTMLInputElement | null = null

	// Mouse interaction state
	mouseX = 0
	mouseY = 0
	mouseLeft = false
	mouseRight = false
	mouseForceRadius = 200
	mouseForceStrength = 5000
	boundMouseMove: ((e: MouseEvent) => void) | null = null
	boundMouseDown: ((e: MouseEvent) => void) | null = null
	boundMouseUp: ((e: MouseEvent) => void) | null = null
	boundContextMenu: ((e: MouseEvent) => void) | null = null

	// Slider references for auto-balance sync
	_affectRadiusInput: HTMLElement | null = null
	_forceRepelDistanceInput: HTMLElement | null = null
	_baseStrengthInput: HTMLElement | null = null
	_repelStrengthInput: HTMLElement | null = null
	_crowdLimitInput: HTMLElement | null = null
	_spreadInput: HTMLElement | null = null
	_autoBalanceSummary: HTMLElement | null = null

	// Skeuomorphic widget references (updated in update loop)
	_volumeVu: VuMeter | null = null
	_forceStrengthVu: VuMeter | null = null
	_repelStrengthVu: VuMeter | null = null
	_spreadGauge: MiniGauge | null = null

	// DOM refs for auto-randomize UI sync from update loop
	_matrixWrapper: HTMLElement | null = null
	_matrixContainer: HTMLElement | null = null
	_matrixRootContainer: HTMLElement | null = null
	_particlesContainer: HTMLElement | null = null

	// Dirty flags — avoid re-uploading every frame
	forceMatrixDirty = true
	particleBufferDirty = true

	// Music engine — organelle-driven, no grid, no schedule
	music = new MusicEngine()
	/** Live radius-pulse state, keyed by typeId. Populated when
	 *  MusicEngine.tick reports a formation event; consumed by
	 *  uploadRadiusScales each frame. */
	activeMusicPulses = new Map<
		number,
		{
			readonly startTime: number
			readonly duration: number
			readonly particleIndices: Uint32Array
		}
	>()
	mutedOrganisms = new Set<string>()
	readbackBuffer: GPUBuffer | null = null
	readbackPending = false
	frameCounter = 0

	// Detection (offloaded to worker)
	detectionState: DetectionFrame | null = null
	detectionConfig: DetectionConfig = { ...DEFAULT_DETECTION_CONFIG }
	detectionWorker: Worker | null = null
	detectionMsgId = 0
	detectionWorkerBusy = false
	/** Serialized prev frame kept in sync for the worker. */
	prevFrameWire: DetectionFrameWire | null = null
	/** Last particle count from readback, used by worker response for overlay upload. */
	lastParticleCount = 0

	// Organism registry (stable identity across frames)
	organismRegistry: OrganismRegistry | null = null
	lastReadbackTime = 0
	ledgerToggle: HTMLElement | null = null
	ledgerPanels: HTMLElement | null = null
	ledgerBackdrop: HTMLElement | null = null
	ledgerOrganellesEl: HTMLElement | null = null
	ledgerOrganismsEl: HTMLElement | null = null
	organelleRows = new Map<
		number,
		{ row: HTMLElement; countEl: HTMLElement }
	>()
	organismRows = new Map<
		string,
		{ row: HTMLElement; countEl: HTMLElement; muteBtn: HTMLElement }
	>()
	organelleHeading: HTMLElement | null = null
	organismHeading: HTMLElement | null = null
	unmuteAllBtn: HTMLElement | null = null
	showOrganelleOverlay = false
	showOrganismOverlay = false
	showOrganismCentroids = false
	organismDepthRanks = new Map<number, number>() // osmId (1-based) → depth rank
	organismDepthTexture: GPUTexture | null = null
	organismPrediction: OrganismPrediction | null = null
	predictionDirty = true
	ledgerPredictionsEl: HTMLElement | null = null
	predictionHeading: HTMLElement | null = null
	predictionRows = new Map<
		string,
		{ row: HTMLElement; scoreEl: HTMLElement }
	>()
	speciesPresence = new Map<string, number>() // sig → presence score [0,1]
	speciesBrightness = new Map<string, number>() // sig → visual brightness [0,1]
	speciesDecayThreshold = 0.05
	speciesDecaySlider: HTMLInputElement | null = null
	lastPredictionTime = 0
	detectionBuffer: GPUBuffer | null = null
	radiusScaleBuffer: GPUBuffer | null = null

	/* ================================================================ */
	/*  effective params — base values × scale                           */
	/* ================================================================ */

	getEffectiveParams() {
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

	syncAutoBalanceSliders() {
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

	renderAutoBalanceSummary() {
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
		this.music = new MusicEngine()
		this.music.onRandomizeRequest = () => this.randomizeForceMatrixAndCounts()
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

		// Organelle-driven music: each frame, ask the engine to observe the
		// current organelle set and re-attack any voice whose type has a new
		// formation. Formation events are returned so the radius pulses on
		// the associated particles can stay in sync with the audio.
		if (this.music.isEnabled && this.detectionState) {
			const formations = this.music.tick(this.detectionState.organelles)
			if (formations.length > 0) {
				const now = performance.now() / 1000
				for (const f of formations) {
					this.activeMusicPulses.set(f.typeId, {
						startTime: now,
						duration: f.sustainSec,
						particleIndices: f.particleIndices,
					})
				}
			}
		}

		this.uploadRadiusScales()


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

					// Run detection pipeline in worker
					const now = performance.now() / 1000
					const dt =
						this.lastReadbackTime > 0 ? now - this.lastReadbackTime : 0.1
					this.lastReadbackTime = now

					const scaledDetConfig: DetectionConfig = {
						...this.detectionConfig,
						proximityRadius: this.detectionConfig.proximityRadius * this.scale,
						organismProximityRadius:
							this.detectionConfig.organismProximityRadius * this.scale,
					}

					if (!this.detectionWorkerBusy) {
						this.detectionWorkerBusy = true
						const worker = this.ensureDetectionWorker()
						const reqId = ++this.detectionMsgId

						const req: DetectionWorkerRequest = {
							id: reqId,
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
							config: scaledDetConfig,
							dt,
							forceMatrix: this.forceMatrix,
							typeKeys: types,
							prevFrame: this.prevFrameWire,
						}

						this.lastParticleCount = n
						worker.postMessage(req)
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
		if (
			this.activePostEffect.id === "stain" &&
			this.stainPipeline &&
			this.stainBindGroups[this.stainPingPong]
		) {
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
			const quadIdx =
				this.activePostEffect.id === "stain" ? 1 - this.stainPingPong : 0
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
	getParticleShaderParams(): import("../ui/shader-menu").ShaderParamDef[] {
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
	getPostShaderParams(): import("../ui/shader-menu").ShaderParamDef[] {
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

	createBuffers() {
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

	createComputePipeline() {
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

	createRenderPipelines() {
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

	rebuildParticleRenderPipeline() {
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

	rebuildCircleOverlayPipeline() {
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
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
			},
			primitive: { topology: "triangle-list" },
		})

		// Detection fill pipeline: writes organelle ID to R8Uint + color to RGBA8
		const fillModule = device.createShaderModule({
			code: PARTICLE_PREFIX + detectionFillFrag,
		})
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

		const edgeModule = device.createShaderModule({ code: detectionEdgeSrc })
		this.detectionEdgePipeline = device.createRenderPipeline({
			layout: edgePipelineLayout,
			vertex: { module: edgeModule, entryPoint: "vs_main" },
			fragment: {
				module: edgeModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
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
		const osmFillModule = device.createShaderModule({
			code: osmFillPrefix + organismFillFrag,
		})
		this.organismFillPipeline = device.createRenderPipeline({
			layout: particlePipelineLayout,
			vertex: { module: osmFillModule, entryPoint: "vs_main" },
			fragment: {
				module: osmFillModule,
				entryPoint: "fs_main",
				targets: [{ format: "r8uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: DEPTH_LESS_EQUAL,
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

		const centroidThinRingModule = device.createShaderModule({
			code: centroidPrefix + centroidThinRingFrag,
		})
		this.osmLevelCentroidPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidThinRingModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidThinRingModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
			},
			primitive: { topology: "triangle-list" },
		})

		const centroidModule = device.createShaderModule({
			code: centroidPrefix + centroidVisualFrag,
		})
		const centroidFillModule = device.createShaderModule({
			code: centroidFillPrefixSrc + centroidFillFrag,
		})
		this.organismCentroidPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: centroidModule, entryPoint: "vs_main" },
			fragment: {
				module: centroidModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
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
			depthStencil: DEPTH_LESS_EQUAL,
		})

		// Organism connection line pipeline: draws white lines between linked organelle centroids.
		// Reuses the same bind group layout as centroid circles (storage + uniform).
		const lineModule = device.createShaderModule({
			code: linePrefix + lineVisualFrag,
		})
		const lineFillModule = device.createShaderModule({
			code: lineFillPrefixSrc + lineFillFrag,
		})
		this.organismLinePipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: lineModule, entryPoint: "vs_main" },
			fragment: {
				module: lineModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
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
			depthStencil: DEPTH_LESS_EQUAL,
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

		const osmEdgeModule = device.createShaderModule({ code: organismEdgeSrc })
		this.organismEdgePipeline = device.createRenderPipeline({
			layout: osmEdgePipelineLayout,
			vertex: { module: osmEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: osmEdgeModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
			},
			primitive: { topology: "triangle-list" },
		})

		// --- JFA seed pipelines ---
		const jfaOrganelleSeedModule = device.createShaderModule({
			code: PARTICLE_PREFIX + jfaOrganelleSeedFrag,
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

		const jfaOrganismSeedModule = device.createShaderModule({
			code: osmFillPrefix + jfaOrganismSeedFrag,
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
			depthStencil: DEPTH_LESS_EQUAL,
		})

		const centroidSeedModule = device.createShaderModule({
			code: centroidFillPrefixSrc + centroidSeedFrag,
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
			depthStencil: DEPTH_LESS_EQUAL,
		})

		const lineSeedModule = device.createShaderModule({
			code: lineFillPrefixSrc + lineSeedFrag,
		})
		this.jfaOrganismLineSeedPipeline = device.createRenderPipeline({
			layout: centroidPipelineLayout,
			vertex: { module: lineSeedModule, entryPoint: "vs_main" },
			fragment: {
				module: lineSeedModule,
				entryPoint: "fs_main",
				targets: [{ format: "rg32uint" }],
			},
			primitive: { topology: "triangle-list" },
			depthStencil: DEPTH_LESS_EQUAL,
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

		const jfaOrgEdgeModule = device.createShaderModule({
			code: jfaEdgePrefix + jfaOrganelleEdgeFrag,
		})
		this.jfaOrganelleEdgePipeline = device.createRenderPipeline({
			layout: jfaEdgePipelineLayout,
			vertex: { module: jfaOrgEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOrgEdgeModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
			},
			primitive: { topology: "triangle-list" },
		})

		const jfaOsmEdgeModule = device.createShaderModule({
			code: jfaEdgePrefix + jfaOrganismEdgeFrag,
		})
		this.jfaOrganismEdgePipeline = device.createRenderPipeline({
			layout: jfaEdgePipelineLayout,
			vertex: { module: jfaOsmEdgeModule, entryPoint: "vs_main" },
			fragment: {
				module: jfaOsmEdgeModule,
				entryPoint: "fs_main",
				targets: [{ format: this.canvasFormat, blend: ALPHA_BLEND }],
			},
			primitive: { topology: "triangle-list" },
		})

		this.rebuildCircleRenderBindGroups()
	}

	createOffscreenTexture() {
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

	createQuadPipeline() {
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

	rebuildQuadPipeline() {
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

	createStainPipeline() {
		const device = this.device!

		this.stainParamsBuffer = device.createBuffer({
			size: 16, // bgR, bgG, bgB, decayRate
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		})

		this.stainBindGroupLayout = device.createBindGroupLayout({
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
					texture: { sampleType: "float" },
				},
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: "uniform" },
				},
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

	clearStainTextures() {
		const device = this.device!
		const encoder = device.createCommandEncoder()
		for (let i = 0; i < 2; i++) {
			if (!this.stainViews[i]) continue
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{
						view: this.stainViews[i]!,
						clearValue: { r: 0, g: 0, b: 0, a: 0 },
						loadOp: "clear",
						storeOp: "store",
					},
				],
			})
			pass.end()
		}
		device.queue.submit([encoder.finish()])
		this.stainPingPong = 0
	}

	rebuildStainBindGroups() {
		const device = this.device!
		if (
			!this.stainBindGroupLayout ||
			!this.stainViews[0] ||
			!this.stainViews[1]
		)
			return

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

	rebuildAllBindGroups() {
		this.rebuildComputeBindGroups()
		this.rebuildParticleRenderBindGroups()
		this.rebuildCircleRenderBindGroups()
		this.rebuildQuadBindGroup()
		this.rebuildStainBindGroups()
	}

	rebuildComputeBindGroups() {
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

	rebuildParticleRenderBindGroups() {
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

	rebuildCircleRenderBindGroups() {
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

	rebuildJfaBindGroups() {
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

	rebuildQuadBindGroup() {
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
					{
						binding: 3,
						resource: this.stainViews[stainReadIdx] ?? this.offscreenView!,
					},
				],
			})
		}
	}

	/* ================================================================ */
	/*  Data upload helpers                                               */
	/* ================================================================ */

	uploadBubbleParams() {
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

	uploadParticleData() {
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

	uploadForceMatrix() {
		const device = this.device!
		const types = this.getTypeIds()
		const n = types.length
		const flat = new Float32Array(MAX_TYPES * MAX_TYPES)

		// Also build a dense N×N view for the music engine so it can
		// derive key/mode from the current matrix (recomputed once per
		// change; O(N²) is negligible for typical N ≤ 12).
		const dense: number[][] = new Array(n)
		for (let si = 0; si < n; si++) {
			const row = this.forceMatrix[types[si]]
			const denseRow = new Array(n).fill(0)
			if (row) {
				for (let ti = 0; ti < n; ti++) {
					const v = row[types[ti]] ?? 0
					flat[si * MAX_TYPES + ti] = v
					denseRow[ti] = v
				}
			}
			dense[si] = denseRow
		}

		device.queue.writeBuffer(this.forceMatrixBuffer!, 0, flat)
		this.music.setForceMatrix(dense)
	}

	/**
	 * Upload per-particle detection IDs into the detection buffer.
	 * This is a separate GPU buffer (1 u32 per particle), avoiding strided writes
	 * into the particle buffer and keeping detection data cleanly separated.
	 */
	uploadDetectionIds(frame: DetectionFrame, n: number) {
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
	getActiveParticleParams(): number[] {
		const id = this.activeParticleEffect.id
		return (
			this.particleEffectParams[id] ?? effectDefaults(this.activeParticleEffect)
		)
	}

	/** Get current param values for the active post effect */
	getActivePostParams(): number[] {
		const id = this.activePostEffect.id
		return this.postEffectParams[id] ?? effectDefaults(this.activePostEffect)
	}

	/** Ensure defaults exist for all effects */
	initEffectParams() {
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
	syncParamHiddenInputs() {
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

	uploadRenderParams() {
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
	 * Called when BPM or time multiplier change mid-playback.
	 */
	ensureDetectionWorker(): Worker {
		if (!this.detectionWorker) {
			this.detectionWorker = new Worker(
				new URL("../detection/worker.ts", import.meta.url),
				{ type: "module" },
			)
			this.detectionWorker.onmessage = (
				e: MessageEvent<DetectionWorkerResponse>,
			) => {
				const resp = e.data
				this.detectionWorkerBusy = false

				// Deserialize the detection frame
				const frame = deserializeDetectionFrame(resp.frame)
				this.detectionState = frame
				this.prevFrameWire = resp.frame

				// Upload overlays if enabled
				if (
					this.showOrganelleOverlay ||
					this.showOrganismOverlay ||
					this.showOrganismCentroids
				) {
					this.uploadDetectionIds(frame, this.lastParticleCount)
				}
				if (this.showOrganismOverlay || this.showOrganismCentroids) {
					this.uploadOrganismCentroids(frame)
					this.uploadOsmLevelCentroids(frame)
				}
				updateLedgerUI(this)

				// Update organism registry for stable identity tracking
				const organelleMap = new Map<number, OrganelleState>()
				for (const org of frame.organelles) {
					organelleMap.set(org.id, org)
				}
				this.organismRegistry = updateRegistry(
					this.organismRegistry,
					frame,
					organelleMap,
					0.1, // approximate dt — registry matching is tolerant
					80,
					performance.now() / 1000,
					this.width,
					this.height,
				)
			}
		}
		return this.detectionWorker
	}

	/** Fill the per-particle radius-scale buffer. Applies a decaying pulse
	 *  envelope to every particle whose organelle triggered a note recently
	 *  (via activeMusicPulses, populated in the animate loop from
	 *  music.tick's formation events). */
	uploadRadiusScales() {
		const device = this.device!
		const buf = this.radiusScaleBuffer
		if (!buf) return

		const n = this.count
		const scales = new Float32Array(n)
		scales.fill(1.0)

		if (this.activeMusicPulses.size > 0) {
			const now = performance.now() / 1000
			const pulseScale = this.getEffectiveParams().pulseScale
			const expired: number[] = []
			for (const [typeId, pulse] of this.activeMusicPulses) {
				const elapsed = now - pulse.startTime
				if (elapsed >= pulse.duration) {
					expired.push(typeId)
					continue
				}
				// Envelope matches the audio: fast ~30 ms attack, then a slow
				// exponential decay over `duration`. Both computed relative
				// to elapsed / duration so shorter notes pulse faster.
				const x = elapsed / pulse.duration
				const attackFrac = Math.min(0.15, 0.03 / pulse.duration)
				let env: number
				if (x < attackFrac) {
					env = x / attackFrac
				} else {
					const decayX = (x - attackFrac) / (1 - attackFrac)
					env = Math.exp(-decayX * 5)
				}
				const scale = 1 + env * pulseScale
				const idxs = pulse.particleIndices
				for (let i = 0; i < idxs.length; i++) {
					const idx = idxs[i]
					if (idx < n && scale > scales[idx]) scales[idx] = scale
				}
			}
			for (const typeId of expired) this.activeMusicPulses.delete(typeId)
		}

		device.queue.writeBuffer(buf, 0, scales.buffer, 0, n * 4)
	}

	uploadOrganismCentroids(frame: DetectionFrame) {
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
				const dx = toroidalDelta(a.centroidX, b.centroidX, this.width)
				const dy = toroidalDelta(a.centroidY, b.centroidY, this.height)
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
	extrapolateOrganismCentroids() {
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

		const w = this.width
		const h = this.height
		for (let i = 0; i < snap.length; i++) {
			const s = snap[i]
			const off = i * 4
			f32[off + 0] = (((s.cx + s.vx * dt) % w) + w) % w
			f32[off + 1] = (((s.cy + s.vy * dt) % h) + h) % h
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
				// Use toroidal delta so lines connect across the boundary correctly
				const ax = (((a.cx + a.vx * dt) % w) + w) % w
				const ay = (((a.cy + a.vy * dt) % h) + h) % h
				const bx = ax + toroidalDelta(ax, (((b.cx + b.vx * dt) % w) + w) % w, w)
				const by = ay + toroidalDelta(ay, (((b.cy + b.vy * dt) % h) + h) % h, h)
				lineF32[off + 0] = ax
				lineF32[off + 1] = ay
				lineF32[off + 2] = bx
				lineF32[off + 3] = by
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
	uploadOsmLevelCentroids(frame: DetectionFrame) {
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
	extrapolateOsmLevelCentroids() {
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

		const w = this.width
		const h = this.height
		for (let i = 0; i < snap.length; i++) {
			const s = snap[i]
			const off = i * 4
			f32[off + 0] = (((s.cx + s.vx * dt) % w) + w) % w
			f32[off + 1] = (((s.cy + s.vy * dt) % h) + h) % h
			f32[off + 2] = radius
			u32[off + 3] = s.id
		}

		device.queue.writeBuffer(buf, 0, data)
	}

	uploadQuadParams() {
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

	uploadFalloffLUT(lut?: Float32Array) {
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

	rebuildBuffers() {
		this.count = this.particles.length
		this.particleBufferDirty = true
		this.forceMatrixDirty = true
		this.uploadRenderParams()
	}

	/** Write only color (and type-id) fields into the GPU buffers,
	 *  leaving positions and velocities untouched so the simulation
	 *  continues from its current state. */
	uploadParticleColors() {
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
	uploadParticleRange(startIdx: number, count: number) {
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
	removeParticlesByIndices(indices: number[]) {
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
				build: (c) => buildDisplayWindow(this, c),
			},
			{
				id: "physics",
				title: "Physics",
				icon: "\u2699\uFE0F",
				category: "simulation",
				defaultVisible: false,
				defaultPosition: { x: 12, y: 300 },
				defaultWidth: 280,
				build: (c) => buildPhysicsWindow(this, c),
			},
			{
				id: "particles",
				title: "Particles",
				icon: "\uD83D\uDFE2",
				category: "simulation",
				defaultVisible: false,
				defaultPosition: { x: 300, y: 60 },
				defaultWidth: 280,
				build: (c) => buildParticlesWindow(this, c),
			},
			{
				id: "music",
				title: "Music",
				icon: "\uD83C\uDFB5",
				category: "music",
				defaultVisible: false,
				defaultPosition: { x: 300, y: 300 },
				defaultWidth: 280,
				build: (c) => buildMusicWindow(this, c),
			},
			{
				id: "detection",
				title: "Detection",
				icon: "\uD83D\uDD2C",
				category: "detection",
				defaultVisible: false,
				defaultPosition: { x: 600, y: 60 },
				defaultWidth: 280,
				build: (c) => buildDetectionWindow(this, c),
			},
			{
				id: "shaders",
				title: "Shader Effects",
				icon: "\u2728",
				category: "visual",
				defaultVisible: false,
				defaultPosition: { x: 600, y: 300 },
				defaultWidth: 280,
				build: (c) => buildShadersWindow(this, c),
			},
		]
	}

	getTypeIds(): string[] {
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

	/**
	 * Randomize the force matrix and particle counts, then push both to
	 * the UI. Used by the wall-clock auto-randomize and by the music
	 * engine's per-progression-loop randomize request.
	 */
	randomizeForceMatrixAndCounts() {
		const types = this.getTypeIds()
		this.forceMatrix = randomizeMatrix(types)
		this.forceMatrixDirty = true
		this.randomizeCounts()
		if (this._matrixWrapper) syncMatrixUI(this, this._matrixWrapper, types)
		if (this._matrixContainer) syncMatrixHidden(this, this._matrixContainer)
		if (this._matrixRootContainer) {
			this._matrixRootContainer.dispatchEvent(
				new Event("change", { bubbles: true }),
			)
		}
	}

	/** Randomize particle counts per type and sync UI. */
	randomizeCounts() {
		const types = this.getTypeIds()
		const allRemoveIndices: number[] = []
		const pendingAdds: Array<{ type: string; count: number }> = []

		// Phase 1: decide which types are active (10% chance each is zeroed)
		const activeFlags = types.map(() => Math.random() >= 0.1)
		if (!activeFlags.some(Boolean)) {
			activeFlags[Math.floor(Math.random() * activeFlags.length)] = true
		}
		const activeCount = activeFlags.filter(Boolean).length

		// Phase 2: density-based total from screen area and scale
		const screenArea = this.width * this.height
		const rawTotal = Math.round((DENSITY_TARGET * screenArea) / this.scale)
		const minTotal = activeCount * MIN_PER_ACTIVE_TYPE
		const finalTotal = Math.min(Math.max(rawTotal, minTotal), MAX_PARTICLES)

		// Phase 3: distribute across types via random weights
		const weights = types.map((_, i) => (activeFlags[i] ? Math.random() : 0))
		const weightSum = weights.reduce((a, b) => a + b, 0)
		const remainder = finalTotal - minTotal
		const desired = types.map((_, i) => {
			if (!activeFlags[i]) return 0
			if (remainder <= 0) return MIN_PER_ACTIVE_TYPE
			return (
				MIN_PER_ACTIVE_TYPE + Math.round((weights[i] / weightSum) * remainder)
			)
		})

		// Phase 4: single-pass rounding correction
		let sum = desired.reduce((a, b) => a + b, 0)
		let diff = finalTotal - sum
		const activeIndices = types.map((_, i) => i).filter((i) => activeFlags[i])
		let idx = 0
		while (diff !== 0 && idx < activeIndices.length * 2) {
			const i = activeIndices[idx % activeIndices.length]
			if (diff > 0) {
				desired[i]++
				diff--
			} else if (desired[i] > MIN_PER_ACTIVE_TYPE) {
				desired[i]--
				diff++
			}
			idx++
		}

		// Apply desired counts
		for (let t = 0; t < types.length; t++) {
			const type = types[t]
			const currentCount = this.particles.filter(
				(p) => p.groupId === type,
			).length

			if (desired[t] > currentCount) {
				pendingAdds.push({ type, count: desired[t] - currentCount })
			} else if (desired[t] < currentCount) {
				let toRemove = currentCount - desired[t]
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

	getTypeColor(type: string): [number, number, number] {
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

	cleanup() {
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

		closeLedger(this)
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

		this.detectionWorker?.terminate()
		this.detectionWorker = null
		this.detectionWorkerBusy = false
		this.prevFrameWire = null
		this.music.dispose()
		this.activeMusicPulses.clear()
		this.readbackBuffer?.destroy()
		this.readbackBuffer = null
		this.detectionBuffer?.destroy()
		this.detectionBuffer = null
		this.radiusScaleBuffer?.destroy()
		this.radiusScaleBuffer = null
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

	readSavedParticleConfig(): {
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

	rgbToHex(rgb: [number, number, number]): string {
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

	hexToRgb(hex: string): [number, number, number] {
		const r = parseInt(hex.slice(1, 3), 16) / 255
		const g = parseInt(hex.slice(3, 5), 16) / 255
		const b = parseInt(hex.slice(5, 7), 16) / 255
		return [r, g, b]
	}

	generateGroupId(): string {
		return "p" + this.nextGroupId++
	}

	generateName(): string {
		const existingNames = new Set(this.groupNames.values())
		if (!existingNames.has("particle")) return "particle"
		let i = 1
		while (existingNames.has(`particle ${i}`)) i++
		return `particle ${i}`
	}
}
