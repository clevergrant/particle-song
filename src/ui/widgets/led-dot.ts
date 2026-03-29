/**
 * <led-dot> - 8px circle with CSS glow when active.
 * Replaces checkboxes for on/off toggle states.
 * Has a `checked` property and fires `change` events.
 */
export class LedDot extends HTMLElement {
  private _dot: HTMLElement;
  private _checked = false;
  private _color = "var(--accent-light)";

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: center;
          vertical-align: middle;
          cursor: pointer;
          flex-shrink: 0;
        }
        .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--toggle-track);
          border: 1px solid var(--border-input);
          transition: background 0.15s, box-shadow 0.15s, border-color 0.15s;
        }
        :host(:hover) .dot {
          border-color: var(--border-hover);
        }
        .dot.on {
          border-color: transparent;
        }
      </style>
      <div class="dot"></div>
    `;

    this._dot = shadow.querySelector(".dot")!;
    this.addEventListener("click", () => {
      this.checked = !this._checked;
      this.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  get checked(): boolean {
    return this._checked;
  }

  set checked(v: boolean) {
    this._checked = v;
    this._dot.classList.toggle("on", v);
    if (v) {
      this._dot.style.background = this._color;
      this._dot.style.boxShadow = `0 0 6px ${this._color}`;
    } else {
      this._dot.style.background = "var(--toggle-track)";
      this._dot.style.boxShadow = "none";
    }
  }

  get color(): string {
    return this._color;
  }

  set color(v: string) {
    this._color = v;
    if (this._checked) this.checked = true; // refresh
  }

  static get observedAttributes(): string[] {
    return ["checked", "color"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "checked") this.checked = val !== null && val !== "false";
    if (name === "color") this.color = val;
  }
}

customElements.define("led-dot", LedDot);
