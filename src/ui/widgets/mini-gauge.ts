/**
 * <mini-gauge> - 20px arc with a needle (speedometer-style).
 * value: 0-1 maps needle from left to right across a 180° arc.
 */
const SIZE = 20;
const CX = SIZE / 2;
const CY = SIZE - 3;
const RADIUS = 7;
const NEEDLE_LEN = 6;

export class MiniGauge extends HTMLElement {
  private _needle: SVGLineElement;
  private _value = 0;

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: center;
          vertical-align: middle;
          width: ${SIZE}px;
          height: ${SIZE}px;
          flex-shrink: 0;
        }
        svg { display: block; }
      </style>
    `;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(SIZE));
    svg.setAttribute("height", String(SIZE));
    svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);

    // Arc background
    const arc = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const x1 = CX - RADIUS;
    const x2 = CX + RADIUS;
    arc.setAttribute("d", `M ${x1} ${CY} A ${RADIUS} ${RADIUS} 0 0 1 ${x2} ${CY}`);
    arc.setAttribute("fill", "none");
    arc.setAttribute("stroke", "var(--border-input)");
    arc.setAttribute("stroke-width", "1.5");
    arc.setAttribute("stroke-linecap", "round");

    // Tick marks at 0%, 50%, 100%
    const ticks = document.createDocumentFragment();
    for (const frac of [0, 0.5, 1]) {
      const angle = Math.PI + frac * Math.PI; // left to right
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      const outer = RADIUS + 1;
      const inner = RADIUS - 1.5;
      tick.setAttribute("x1", String(CX + Math.cos(angle) * inner));
      tick.setAttribute("y1", String(CY + Math.sin(angle) * inner));
      tick.setAttribute("x2", String(CX + Math.cos(angle) * outer));
      tick.setAttribute("y2", String(CY + Math.sin(angle) * outer));
      tick.setAttribute("stroke", "var(--border-hover)");
      tick.setAttribute("stroke-width", "0.8");
      ticks.appendChild(tick);
    }

    // Needle
    this._needle = document.createElementNS("http://www.w3.org/2000/svg", "line");
    this._needle.setAttribute("x1", String(CX));
    this._needle.setAttribute("y1", String(CY));
    this._needle.setAttribute("stroke", "var(--accent-hover)");
    this._needle.setAttribute("stroke-width", "1.5");
    this._needle.setAttribute("stroke-linecap", "round");

    // Center pivot
    const pivot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    pivot.setAttribute("cx", String(CX));
    pivot.setAttribute("cy", String(CY));
    pivot.setAttribute("r", "1.2");
    pivot.setAttribute("fill", "var(--toggle-knob)");

    svg.append(arc, ticks, this._needle, pivot);
    shadow.appendChild(svg);
    this._updateNeedle();
  }

  get value(): number {
    return this._value;
  }

  set value(v: number) {
    this._value = Math.max(0, Math.min(1, v));
    this._updateNeedle();
  }

  private _updateNeedle(): void {
    // 0 = left (π), 1 = right (2π / 0)
    const angle = Math.PI + this._value * Math.PI;
    const x2 = CX + Math.cos(angle) * NEEDLE_LEN;
    const y2 = CY + Math.sin(angle) * NEEDLE_LEN;
    this._needle.setAttribute("x2", String(x2));
    this._needle.setAttribute("y2", String(y2));
  }

  static get observedAttributes(): string[] {
    return ["value"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "value") this.value = parseFloat(val) || 0;
  }
}

customElements.define("mini-gauge", MiniGauge);
