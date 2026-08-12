import type { RandomDots } from "../basic-particles"

export function buildShadersWindow(dots: RandomDots, container: HTMLElement) {
	// Hidden inputs for shader effect persistence
	const hiddenParticleEffect = document.createElement("input")
	hiddenParticleEffect.type = "hidden"
	hiddenParticleEffect.dataset.setting = "particleEffect"
	hiddenParticleEffect.value = dots.activeParticleEffect.id
	hiddenParticleEffect.addEventListener("input", () => {
		if (hiddenParticleEffect.value !== dots.activeParticleEffect.id) {
			dots.switchParticleShader(hiddenParticleEffect.value)
			dots.onParticleEffectChanged?.(hiddenParticleEffect.value)
		}
	})
	container.appendChild(hiddenParticleEffect)

	const hiddenPostEffect = document.createElement("input")
	hiddenPostEffect.type = "hidden"
	hiddenPostEffect.dataset.setting = "postEffect"
	hiddenPostEffect.value = dots.activePostEffect.id
	hiddenPostEffect.addEventListener("input", () => {
		if (hiddenPostEffect.value !== dots.activePostEffect.id) {
			dots.switchPostShader(hiddenPostEffect.value)
			dots.onPostEffectChanged?.(hiddenPostEffect.value)
		}
	})
	container.appendChild(hiddenPostEffect)

	dots.onParticleEffectChanged = null
	dots.onPostEffectChanged = null
	dots._hiddenParticleEffect = hiddenParticleEffect
	dots._hiddenPostEffect = hiddenPostEffect

	// Hidden inputs for per-effect param persistence
	const hiddenParticleParams = document.createElement("input")
	hiddenParticleParams.type = "hidden"
	hiddenParticleParams.dataset.setting = "particleEffectParams"
	hiddenParticleParams.value = JSON.stringify(dots.particleEffectParams)
	hiddenParticleParams.addEventListener("input", () => {
		try {
			const parsed = JSON.parse(hiddenParticleParams.value)
			if (typeof parsed === "object" && parsed !== null) {
				dots.particleEffectParams = parsed
				dots.uploadRenderParams()
			}
		} catch {
			/* ignore corrupt data */
		}
	})
	container.appendChild(hiddenParticleParams)
	dots._hiddenParticleParams = hiddenParticleParams

	const hiddenPostParams = document.createElement("input")
	hiddenPostParams.type = "hidden"
	hiddenPostParams.dataset.setting = "postEffectParams"
	hiddenPostParams.value = JSON.stringify(dots.postEffectParams)
	hiddenPostParams.addEventListener("input", () => {
		try {
			const parsed = JSON.parse(hiddenPostParams.value)
			if (typeof parsed === "object" && parsed !== null) {
				dots.postEffectParams = parsed
				dots.uploadQuadParams()
			}
		} catch {
			/* ignore corrupt data */
		}
	})
	container.appendChild(hiddenPostParams)
	dots._hiddenPostParams = hiddenPostParams
}
