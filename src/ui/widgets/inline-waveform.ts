/**
 * <inline-waveform> - ~60x16px SVG sparkline for envelope shapes.
 * Accepts an array of {x, y} points normalized to [0,1].
 */
const WIDTH = 60;
const HEIGHT = 16;

export class InlineWaveform extends HTMLElement {
  private _path: SVGPathElement;
  private _points: Array<{ x: number; y: number }> = [];

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: center;
          vertical-align: middle;
          width: ${WIDTH}px;
          height: ${HEIGHT}px;
          flex-shrink: 0;
        }
        svg { display: block; }
      </style>
    `;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(WIDTH));
    svg.setAttribute("height", String(HEIGHT));
    svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);

    this._path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    this._path.setAttribute("fill", "none");
    this._path.setAttribute("stroke", "var(--accent-light)");
    this._path.setAttribute("stroke-width", "1.2");
    this._path.setAttribute("stroke-linecap", "round");
    this._path.setAttribute("stroke-linejoin", "round");

    svg.appendChild(this._path);
    shadow.appendChild(svg);
  }

  get points(): Array<{ x: number; y: number }> {
    return this._points;
  }

  set points(pts: Array<{ x: number; y: number }>) {
    this._points = pts;
    this._updatePath();
  }

  private _updatePath(): void {
    if (this._points.length < 2) {
      this._path.setAttribute("d", "");
      return;
    }

    const pad = 1;
    const w = WIDTH - pad * 2;
    const h = HEIGHT - pad * 2;

    const d = this._points
      .map((p, i) => {
        const x = pad + p.x * w;
        const y = pad + (1 - p.y) * h; // flip Y
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

    this._path.setAttribute("d", d);
  }
}

customElements.define("inline-waveform", InlineWaveform);
