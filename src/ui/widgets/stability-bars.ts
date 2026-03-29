/**
 * <stability-bars> - Signal-strength style bars (3-5 bars).
 * value: 0-1 maps to 0 bars → all bars lit.
 */
const BAR_COUNT = 5;

export class StabilityBars extends HTMLElement {
  private _bars: HTMLElement[] = [];
  private _value = 0;

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: flex-end;
          gap: 1px;
          vertical-align: middle;
          height: 14px;
          flex-shrink: 0;
        }
        .bar {
          width: 3px;
          background: var(--toggle-track);
          border-radius: 1px;
          transition: background 0.12s;
        }
        .bar.on {
          background: var(--accent-light);
        }
      </style>
    `;

    for (let i = 0; i < BAR_COUNT; i++) {
      const bar = document.createElement("div");
      bar.className = "bar";
      // Heights increase: 4px, 6px, 8px, 10px, 12px
      bar.style.height = `${4 + i * 2}px`;
      shadow.appendChild(bar);
      this._bars.push(bar);
    }
  }

  get value(): number {
    return this._value;
  }

  set value(v: number) {
    this._value = Math.max(0, Math.min(1, v));
    const lit = Math.round(this._value * BAR_COUNT);
    for (let i = 0; i < BAR_COUNT; i++) {
      this._bars[i].classList.toggle("on", i < lit);
    }
  }

  static get observedAttributes(): string[] {
    return ["value"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "value") this.value = parseFloat(val) || 0;
  }
}

customElements.define("stability-bars", StabilityBars);
