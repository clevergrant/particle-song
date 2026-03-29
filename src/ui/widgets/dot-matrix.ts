/**
 * <dot-matrix> - Small NxN colored-dot grid for force matrix preview.
 * Accepts a flat array of colors and a size N.
 */
export class DotMatrix extends HTMLElement {
  private _grid: HTMLElement;
  private _size = 0;
  private _cells: HTMLElement[] = [];

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: "open" });

    shadow.innerHTML = `
      <style>
        :host {
          display: inline-flex;
          align-items: center;
          vertical-align: middle;
          flex-shrink: 0;
        }
        .grid {
          display: inline-grid;
          gap: 1px;
        }
        .dot {
          width: 4px;
          height: 4px;
          border-radius: 50%;
          background: var(--toggle-track);
        }
      </style>
      <div class="grid"></div>
    `;

    this._grid = shadow.querySelector(".grid")!;
  }

  /**
   * Set the matrix data.
   * @param size NxN dimension
   * @param values flat array of N*N values in range [-1, 1]
   */
  setMatrix(size: number, values: number[]): void {
    if (size !== this._size) {
      this._size = size;
      this._grid.style.gridTemplateColumns = `repeat(${size}, 4px)`;
      this._grid.innerHTML = "";
      this._cells = [];
      for (let i = 0; i < size * size; i++) {
        const dot = document.createElement("div");
        dot.className = "dot";
        this._grid.appendChild(dot);
        this._cells.push(dot);
      }
    }

    for (let i = 0; i < this._cells.length; i++) {
      const v = values[i] ?? 0;
      if (v > 0) {
        // Green for attraction
        const intensity = Math.min(1, v);
        this._cells[i].style.background = `rgba(76, 175, 80, ${0.3 + intensity * 0.7})`;
      } else if (v < 0) {
        // Red for repulsion
        const intensity = Math.min(1, -v);
        this._cells[i].style.background = `rgba(244, 67, 54, ${0.3 + intensity * 0.7})`;
      } else {
        this._cells[i].style.background = "var(--toggle-track)";
      }
    }
  }
}

customElements.define("dot-matrix", DotMatrix);
