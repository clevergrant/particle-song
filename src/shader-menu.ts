/* ------------------------------------------------------------------ */
/*  Shader Menu — effect selection + param panels for floating window  */
/* ------------------------------------------------------------------ */

import {
  type ShaderEffect,
  particleEffects,
  postEffects,
} from "./shader-registry";
import { NumInput } from "./num-input";

export interface ShaderMenuCallbacks {
  onParticleEffectChange: (effectId: string) => void;
  onPostEffectChange: (effectId: string) => void;
}

export interface ShaderParamDef {
  label: string;
  setting: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
  onChange: (value: number) => void;
}

export class ShaderMenu {
  private _particleParamsPanel: HTMLElement;
  private _postParamsPanel: HTMLElement;
  private activeParticleId: string;
  private activePostId: string;
  private particleButtons = new Map<string, HTMLElement>();
  private postButtons = new Map<string, HTMLElement>();
  private callbacks: ShaderMenuCallbacks;

  /**
   * Build the shader menu into the given container.
   * In window mode, `container` is the floating window's content element.
   * In legacy mode, pass a wrapper element.
   */
  constructor(callbacks: ShaderMenuCallbacks, initialParticle = "gradient", initialPost = "normalize") {
    this.callbacks = callbacks;
    this.activeParticleId = initialParticle;
    this.activePostId = initialPost;

    // Create internal panel elements (no longer query fixed DOM IDs)
    this._particleParamsPanel = document.createElement("div");
    this._particleParamsPanel.className = "shader-param-section";

    this._postParamsPanel = document.createElement("div");
    this._postParamsPanel.className = "shader-param-section";
  }

  /** Build the full effect selector + param panels into a container */
  buildInto(container: HTMLElement) {
    // Particle effects section
    const particleHeader = document.createElement("div");
    particleHeader.className = "shader-menu-category";
    particleHeader.textContent = "Particle";
    container.appendChild(particleHeader);

    const particleBtns = document.createElement("div");
    particleBtns.style.cssText = "display:flex; flex-direction:column; gap:2px;";
    for (const effect of particleEffects) {
      const btn = this.createEffectButton(effect, "particle");
      this.particleButtons.set(effect.id, btn);
      particleBtns.appendChild(btn);
    }
    container.appendChild(particleBtns);

    // Particle params
    container.appendChild(this._particleParamsPanel);

    // Divider
    const divider = document.createElement("div");
    divider.className = "shader-menu-divider";
    container.appendChild(divider);

    // Post-process effects section
    const postHeader = document.createElement("div");
    postHeader.className = "shader-menu-category";
    postHeader.textContent = "Post-Process";
    container.appendChild(postHeader);

    const postBtns = document.createElement("div");
    postBtns.style.cssText = "display:flex; flex-direction:column; gap:2px;";
    for (const effect of postEffects) {
      const btn = this.createEffectButton(effect, "postprocess");
      this.postButtons.set(effect.id, btn);
      postBtns.appendChild(btn);
    }
    container.appendChild(postBtns);

    // Post params
    container.appendChild(this._postParamsPanel);
  }

  private createEffectButton(effect: ShaderEffect, category: "particle" | "postprocess"): HTMLElement {
    const btn = document.createElement("button");
    btn.className = "shader-menu-item";
    btn.textContent = effect.name;
    btn.title = effect.name;

    const isActive = category === "particle"
      ? effect.id === this.activeParticleId
      : effect.id === this.activePostId;
    if (isActive) btn.classList.add("active");

    btn.addEventListener("click", () => {
      if (category === "particle") {
        this.setActiveParticle(effect.id);
        this.callbacks.onParticleEffectChange(effect.id);
      } else {
        this.setActivePost(effect.id);
        this.callbacks.onPostEffectChange(effect.id);
      }
    });

    return btn;
  }

  private setActiveParticle(id: string) {
    this.activeParticleId = id;
    for (const [effectId, btn] of this.particleButtons) {
      btn.classList.toggle("active", effectId === id);
    }
  }

  private setActivePost(id: string) {
    this.activePostId = id;
    for (const [effectId, btn] of this.postButtons) {
      btn.classList.toggle("active", effectId === id);
    }
  }

  /** Update callbacks when simulation changes */
  updateCallbacks(callbacks: ShaderMenuCallbacks) {
    this.callbacks = callbacks;
  }

  /** Update selections from outside (e.g., after loading saved settings) */
  syncSelections(particleId: string, postId: string) {
    this.setActiveParticle(particleId);
    this.setActivePost(postId);
  }

  /** Build param controls for the particle shader panel */
  setParticleParams(params: ShaderParamDef[]) {
    this.buildParamPanel(this._particleParamsPanel, "Particle Params", params);
  }

  /** Build param controls for the post-process shader panel */
  setPostParams(params: ShaderParamDef[]) {
    this.buildParamPanel(this._postParamsPanel, "Post-Process Params", params);
  }

  /** Clear both param panels */
  clearParams() {
    this._particleParamsPanel.innerHTML = "";
    this._postParamsPanel.innerHTML = "";
  }

  /** Expose the param panel elements for external save/load queries */
  get particleParamsEl(): HTMLElement { return this._particleParamsPanel; }
  get postParamsEl(): HTMLElement { return this._postParamsPanel; }

  private buildParamPanel(panel: HTMLElement, title: string, params: ShaderParamDef[]) {
    panel.innerHTML = "";

    if (params.length === 0) return;

    const titleEl = document.createElement("div");
    titleEl.className = "shader-param-panel-title";
    titleEl.style.marginTop = "8px";
    titleEl.textContent = title;
    panel.appendChild(titleEl);

    for (const param of params) {
      const el = document.createElement("num-input") as NumInput;
      el.className = "control-group";
      el.setAttribute("label", param.label);
      el.setAttribute("value", String(param.value));
      el.setAttribute("setting", param.setting);
      if (param.min != null) el.setAttribute("min", String(param.min));
      if (param.max != null) el.setAttribute("max", String(param.max));
      if (param.step != null) el.setAttribute("step", String(param.step));
      el.setAttribute("width", "60px");
      if (param.suffix) el.setAttribute("suffix", param.suffix);

      panel.appendChild(el);

      // Attach listener after connectedCallback has built the inner input
      queueMicrotask(() => {
        el.input.addEventListener("input", () => {
          param.onChange(Number(el.input.value));
        });
      });
    }
  }
}
