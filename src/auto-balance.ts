import type { ForceMatrix } from "./particles"
import { clamp } from "./math-utils"

export interface AutoBalanceInputs {
	readonly forceMatrix: ForceMatrix
	readonly typeKeys: ReadonlyArray<string>
	readonly typeCounts: Readonly<Record<string, number>>
	readonly totalParticles: number
	readonly worldWidth: number
	readonly worldHeight: number
}

export interface AutoBalanceResult {
	readonly affectRadius: number
	readonly forceRepelDistance: number
	readonly baseStrength: number
	readonly repelStrength: number
	readonly crowdLimit: number
	readonly spread: number
}

const BASE_RADIUS_FACTOR = 4.0
const BASE_STRENGTH = 200
const MIN_AFFECT_RADIUS = 20
const MAX_AFFECT_RADIUS = 300
const MIN_CROWD_LIMIT = 8

function lerp(a: number, b: number, t: number): number {
	return a + (b - a) * t
}

/** Extract all force values from the matrix as a flat array */
function allForces(
	forceMatrix: ForceMatrix,
	typeKeys: ReadonlyArray<string>,
): number[] {
	const values: number[] = []
	for (const src of typeKeys) {
		for (const tgt of typeKeys) {
			values.push(forceMatrix[src]?.[tgt] ?? 0)
		}
	}
	return values
}

/** Mean of diagonal entries (self-attraction) */
function meanSelfAttraction(
	forceMatrix: ForceMatrix,
	typeKeys: ReadonlyArray<string>,
): number {
	if (typeKeys.length === 0) return 0
	let sum = 0
	for (const t of typeKeys) {
		sum += forceMatrix[t]?.[t] ?? 0
	}
	return sum / typeKeys.length
}

/** Mean of all force magnitudes (how "loud" the matrix is overall) */
function meanAbsForce(values: ReadonlyArray<number>): number {
	if (values.length === 0) return 0
	let sum = 0
	for (const v of values) sum += Math.abs(v)
	return sum / values.length
}

/** Standard deviation of force values (how varied the interactions are) */
function forceStdDev(values: ReadonlyArray<number>): number {
	if (values.length < 2) return 0
	let sum = 0
	let sumSq = 0
	for (const v of values) {
		sum += v
		sumSq += v * v
	}
	const mean = sum / values.length
	return Math.sqrt(sumSq / values.length - mean * mean)
}

/** Fraction of force pairs that are attractive (> 0) */
function attractionRatio(values: ReadonlyArray<number>): number {
	if (values.length === 0) return 0.5
	let count = 0
	for (const v of values) if (v > 0) count++
	return count / values.length
}

export function autoBalance(inputs: AutoBalanceInputs): AutoBalanceResult {
	const {
		forceMatrix,
		typeKeys,
		typeCounts,
		totalParticles,
		worldWidth,
		worldHeight,
	} = inputs

	// Guard against degenerate cases
	const particles = Math.max(1, totalParticles)
	const numTypes = Math.max(1, typeKeys.length)
	const worldArea = worldWidth * worldHeight

	// --- Matrix analysis ---
	const forces = allForces(forceMatrix, typeKeys)
	const meanAbs = meanAbsForce(forces)     // 0..1 — overall matrix intensity
	const stdDev = forceStdDev(forces)        // 0..~0.6 — interaction diversity
	const attrRatio = attractionRatio(forces) // 0..1 — how attractive overall
	const selfAttr = meanSelfAttraction(forceMatrix, typeKeys) // -1..1

	// --- affectRadius ---
	// Base: scale with mean inter-particle spacing
	const meanSpacing = Math.sqrt(worldArea / particles)
	// High-variance matrices benefit from tighter radius so distinct
	// interactions stay local; low-variance matrices need wider reach
	const varianceFactor = lerp(1.15, 0.85, clamp(stdDev / 0.6, 0, 1))
	const affectRadius = clamp(
		meanSpacing * BASE_RADIUS_FACTOR * varianceFactor,
		MIN_AFFECT_RADIUS,
		MAX_AFFECT_RADIUS,
	)

	// --- forceRepelDistance ---
	// High self-attraction needs larger repel zone to prevent collapse
	const normalizedSelfAttr = (clamp(selfAttr, -1, 1) + 1) / 2 // 0..1
	const repelFraction = lerp(0.3, 1.2, normalizedSelfAttr)
	const forceRepelDistance = affectRadius * repelFraction

	// --- baseStrength ---
	// Weak matrices need stronger amplification to produce visible structure;
	// strong matrices need less so they don't explode
	const intensityFactor = lerp(1.4, 0.7, clamp(meanAbs, 0, 1))
	const baseStrength = BASE_STRENGTH * intensityFactor

	// --- repelStrength ---
	// Mostly-attractive matrices need stronger repulsion to prevent collapse;
	// mostly-repulsive matrices need less to allow any structure to form
	const repelRatio = lerp(0.5, 1.0, clamp(attrRatio, 0, 1))
	const repelStrength = baseStrength * repelRatio

	// --- crowdLimit ---
	// High attraction ratio → particles clump more → allow larger crowds
	const crowdBase = Math.sqrt(particles / numTypes)
	const crowdFactor = lerp(0.8, 1.3, clamp(attrRatio, 0, 1))
	const crowdLimit = Math.max(MIN_CROWD_LIMIT, crowdBase * crowdFactor)

	// --- spread ---
	// High-variance matrices benefit from tighter spread (more locality);
	// uniform matrices need wider spread to find interaction partners
	const spread = Math.round(lerp(45, 25, clamp(stdDev / 0.6, 0, 1)))

	return {
		affectRadius,
		forceRepelDistance,
		baseStrength,
		repelStrength,
		crowdLimit,
		spread,
	}
}
