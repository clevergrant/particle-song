/**
 * <mini-clock> - 16px SVG circle with a single sweeping hand.
 * value: 0-1 maps to 12 o'clock → full clockwise rotation.
 */
const SIZE = 16;
const CX = SIZE / 2;
const CY = SIZE / 2;
const HAND_LEN = 5.5;

export class MiniClock extends HTMLElement {
  private _svg: SVGSVGElement;
  private _hand: SVGLineElement;
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

    this._svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this._svg.setAttribute("width", String(SIZE));
    this._svg.setAttribute("height", String(SIZE));
    this._svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);

    // Circle outline
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(CX));
    circle.setAttribute("cy", String(CY));
    circle.setAttribute("r", String(CX - 1));
    circle.setAttribute("fill", "none");
    circle.setAttribute("stroke", "var(--text-dim)");
    circle.setAttribute("stroke-width", "1");

    // Center dot
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", String(CX));
    dot.setAttribute("cy", String(CY));
    dot.setAttribute("r", "1.2");
    dot.setAttribute("fill", "var(--toggle-knob)");

    // Hand
    this._hand = document.createElementNS("http://www.w3.org/2000/svg", "line");
    this._hand.setAttribute("x1", String(CX));
    this._hand.setAttribute("y1", String(CY));
    this._hand.setAttribute("stroke", "var(--accent-hover)");
    this._hand.setAttribute("stroke-width", "1.5");
    this._hand.setAttribute("stroke-linecap", "round");

    this._svg.append(circle, this._hand, dot);
    shadow.appendChild(this._svg);
    this._updateHand();
  }

  get value(): number {
    return this._value;
  }

  set value(v: number) {
    this._value = v;
    this._updateHand();
  }

  private _updateHand(): void {
    const angle = this._value * Math.PI * 2 - Math.PI / 2; // 0 = 12 o'clock
    const x2 = CX + Math.cos(angle) * HAND_LEN;
    const y2 = CY + Math.sin(angle) * HAND_LEN;
    this._hand.setAttribute("x2", String(x2));
    this._hand.setAttribute("y2", String(y2));
  }

  static get observedAttributes(): string[] {
    return ["value"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "value") this.value = parseFloat(val) || 0;
  }
}

customElements.define("mini-clock", MiniClock);
