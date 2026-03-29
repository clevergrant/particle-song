/**
 * <num-input> — reusable number input web component with label, reset button,
 * wheel-to-adjust, and data-setting / data-default support.
 *
 * Attributes:
 *   label, setting, value, min, max, step, suffix, width
 *
 * The inner <input> carries data-setting and data-default so the existing
 * save/load system picks it up automatically.
 *
 * Events:
 *   "input" — bubbles whenever the value changes (typed, wheel, reset).
 */

import { applyStepDelta } from "./number-scroll";

export class NumInput extends HTMLElement {
  private _input!: HTMLInputElement;
  private _resetBtn!: HTMLButtonElement;
  private _defaultValue = "0";

  static get observedAttributes() {
    return ["label", "setting", "value", "min", "max", "step", "suffix", "width"];
  }

  connectedCallback() {
    this.render();
  }

  private render() {
    // Only build DOM once
    if (this._input) return;

    const label = document.createElement("label");
    label.textContent = this.getAttribute("label") ?? "";
    this.appendChild(label);

    const input = document.createElement("input");
    input.type = "number";
    const min = this.getAttribute("min");
    const max = this.getAttribute("max");
    if (min != null) input.min = min;
    if (max != null) input.max = max;
    input.step = this.getAttribute("step") ?? "1";
    input.value = this.getAttribute("value") ?? "0";
    input.style.width = this.getAttribute("width") ?? "70px";

    const setting = this.getAttribute("setting");
    if (setting) input.dataset.setting = setting;

    this._defaultValue = input.value;
    input.dataset.default = this._defaultValue;

    input.addEventListener("input", () => {
      this.toggleResetVisibility();
    });

    input.addEventListener("wheel", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const direction = e.deltaY > 0 ? -1 : e.deltaY < 0 ? 1 : 0;
      if (direction === 0) return;
      applyStepDelta(input, direction);
    });

    this._input = input;

    // Group reset + input + suffix to the right
    const right = document.createElement("span");
    right.className = "num-input-right";

    // Reset button (left of input)
    const resetBtn = document.createElement("button");
    resetBtn.className = "num-input-reset";
    resetBtn.title = "Reset to default";
    resetBtn.textContent = "\u21BA"; // ↺
    resetBtn.addEventListener("click", (e) => {
      e.preventDefault();
      input.value = this._defaultValue;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    this._resetBtn = resetBtn;
    right.appendChild(resetBtn);

    right.appendChild(input);

    // Always reserve space for suffix (uniform layout)
    const suffixSpan = document.createElement("span");
    suffixSpan.className = "num-input-suffix";
    const suffix = this.getAttribute("suffix");
    suffixSpan.textContent = suffix ?? "";
    right.appendChild(suffixSpan);

    this.appendChild(right);

    this.toggleResetVisibility();
  }

  private toggleResetVisibility() {
    if (!this._resetBtn || !this._input) return;
    const isDefault = this._input.value === this._defaultValue;
    this._resetBtn.style.visibility = isDefault ? "hidden" : "visible";
  }

  /** The underlying <input> element (for external listeners, value reads, etc.) */
  get input(): HTMLInputElement {
    return this._input;
  }

  /** Current numeric value */
  get value(): string {
    return this._input?.value ?? this.getAttribute("value") ?? "0";
  }

  set value(v: string) {
    if (this._input) {
      this._input.value = v;
      this.toggleResetVisibility();
    }
  }

  attributeChangedCallback(name: string, _old: string | null, val: string | null) {
    if (!this._input) return;
    switch (name) {
      case "label": {
        const lbl = this.querySelector("label");
        if (lbl) lbl.textContent = val ?? "";
        break;
      }
      case "value":
        this._input.value = val ?? "0";
        this._defaultValue = val ?? "0";
        this._input.dataset.default = this._defaultValue;
        this.toggleResetVisibility();
        break;
      case "min":
        if (val != null) this._input.min = val; else this._input.removeAttribute("min");
        break;
      case "max":
        if (val != null) this._input.max = val; else this._input.removeAttribute("max");
        break;
      case "step":
        this._input.step = val ?? "1";
        break;
      case "setting":
        if (val) this._input.dataset.setting = val; else delete this._input.dataset.setting;
        break;
    }
  }
}

customElements.define("num-input", NumInput);
