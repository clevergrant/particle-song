/**
 * <vu-meter> - Tiny vertical bar with green→yellow→red gradient.
 * value: 0-1 maps to empty→full.
 */
export class VuMeter extends HTMLElement {
  private _fill: HTMLElement;
  private _value = 0;

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: flex-end;
          vertical-align: middle;
          width: 6px;
          height: 16px;
          flex-shrink: 0;
        }
        .track {
          width: 100%;
          height: 100%;
          background: var(--surface-3);
          border-radius: 2px;
          overflow: hidden;
          display: flex;
          align-items: flex-end;
          border: 0.5px solid var(--toggle-track);
        }
        .fill {
          width: 100%;
          border-radius: 1px;
          transition: height 60ms linear;
          background: linear-gradient(to top, var(--success), var(--warning) 60%, var(--danger));
        }
      </style>
      <div class="track">
        <div class="fill"></div>
      </div>
    `;

    this._fill = shadow.querySelector(".fill")!;
  }

  get value(): number {
    return this._value;
  }

  set value(v: number) {
    this._value = Math.max(0, Math.min(1, v));
    this._fill.style.height = `${this._value * 100}%`;
  }

  static get observedAttributes(): string[] {
    return ["value"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "value") this.value = parseFloat(val) || 0;
  }
}

customElements.define("vu-meter", VuMeter);
