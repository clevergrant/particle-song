import { computeSnap, clampToViewport, type Rect } from "./snap";

/** CSS for the shadow DOM chrome */
const CHROME_CSS = `
:host {
  position: fixed;
  z-index: 100;
  display: flex;
  flex-direction: column;
  will-change: transform;
  pointer-events: auto;
}

:host([hidden]) {
  display: none !important;
}

.fw-chrome {
  display: flex;
  flex-direction: column;
  background: var(--surface-panel-dense);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
  max-height: calc(100vh - 24px);
}

.fw-chrome:focus-within {
  border-color: var(--accent-border);
}

.fw-titlebar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  background: var(--surface-1);
  cursor: grab;
  user-select: none;
  flex-shrink: 0;
}

.fw-titlebar:active {
  cursor: grabbing;
}

.fw-icon {
  font-size: 14px;
  line-height: 1;
  flex-shrink: 0;
}

.fw-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--toggle-knob);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.fw-controls {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.fw-controls button {
  background: none;
  border: none;
  color: var(--text-dim);
  width: 20px;
  height: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  line-height: 1;
  border-radius: 3px;
  transition: background 0.12s, color 0.12s;
}

.fw-controls button:hover {
  background: var(--surface-overlay-hover);
  color: var(--text-muted);
}

.fw-controls .fw-close:hover {
  background: rgba(200, 60, 60, 0.3);
  color: var(--danger-text);
}

.fw-body {
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: none;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.fw-body::-webkit-scrollbar {
  display: none;
}

:host(.minimized) .fw-body {
  display: none;
}

:host(.minimized) .fw-chrome {
  max-height: none;
}
`;

/** Callback to get rects of other visible windows (injected by WindowManager) */
export type GetOtherRectsFn = (excludeId: string) => readonly Rect[];

export class FloatingWindow extends HTMLElement {
  private _shadow: ShadowRoot;
  private _titlebar: HTMLElement;
  private _titleEl: HTMLElement;
  private _iconEl: HTMLElement;
  private _body: HTMLElement;
  private _minimizeBtn: HTMLButtonElement;

  // Drag state
  private _dragging = false;
  private _dragOffsetX = 0;
  private _dragOffsetY = 0;
  private _posX = 0;
  private _posY = 0;
  private _width = 280;

  // Bound handlers
  private _onPointerMove: (e: PointerEvent) => void;
  private _onPointerUp: (e: PointerEvent) => void;

  /** Injected by WindowManager */
  getOtherRects: GetOtherRectsFn = () => [];
  onClose: (() => void) | null = null;
  onFocus: (() => void) | null = null;
  onPositionChange: ((x: number, y: number) => void) | null = null;

  constructor() {
    super();
    this._shadow = this.attachShadow({ mode: "open" });

    const sheet = new CSSStyleSheet();
    sheet.replaceSync(CHROME_CSS);
    this._shadow.adoptedStyleSheets = [sheet];

    const chrome = document.createElement("div");
    chrome.className = "fw-chrome";

    this._titlebar = document.createElement("div");
    this._titlebar.className = "fw-titlebar";

    this._iconEl = document.createElement("span");
    this._iconEl.className = "fw-icon";

    this._titleEl = document.createElement("span");
    this._titleEl.className = "fw-title";

    const controls = document.createElement("div");
    controls.className = "fw-controls";

    this._minimizeBtn = document.createElement("button");
    this._minimizeBtn.className = "fw-minimize";
    this._minimizeBtn.textContent = "\u2014"; // em dash
    this._minimizeBtn.title = "Minimize";
    this._minimizeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.minimized = !this.minimized;
    });

    const closeBtn = document.createElement("button");
    closeBtn.className = "fw-close";
    closeBtn.textContent = "\u2715"; // multiplication x
    closeBtn.title = "Close";
    closeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.onClose?.();
    });

    controls.append(this._minimizeBtn, closeBtn);
    this._titlebar.append(this._iconEl, this._titleEl, controls);

    this._body = document.createElement("div");
    this._body.className = "fw-body";
    const slot = document.createElement("slot");
    this._body.appendChild(slot);

    chrome.append(this._titlebar, this._body);
    this._shadow.appendChild(chrome);

    // Drag handlers
    this._onPointerMove = this._handlePointerMove.bind(this);
    this._onPointerUp = this._handlePointerUp.bind(this);

    this._titlebar.addEventListener("pointerdown", (e) => {
      // Only drag on primary button, ignore clicks on control buttons
      if (e.button !== 0) return;
      if ((e.target as HTMLElement).closest(".fw-controls")) return;
      this._startDrag(e);
    });

    // Focus on any click within the window
    this.addEventListener("pointerdown", () => {
      this.onFocus?.();
    });
  }

  // --- Public API ---

  get windowId(): string {
    return this.getAttribute("window-id") ?? "";
  }

  get windowTitle(): string {
    return this.getAttribute("window-title") ?? "";
  }

  set windowTitle(v: string) {
    this.setAttribute("window-title", v);
    this._titleEl.textContent = v;
  }

  get icon(): string {
    return this.getAttribute("icon") ?? "";
  }

  set icon(v: string) {
    this.setAttribute("icon", v);
    this._iconEl.textContent = v;
  }

  get posX(): number {
    return this._posX;
  }

  get posY(): number {
    return this._posY;
  }

  get minimized(): boolean {
    return this.classList.contains("minimized");
  }

  set minimized(v: boolean) {
    this.classList.toggle("minimized", v);
    this._minimizeBtn.textContent = v ? "\u25A1" : "\u2014"; // square vs em dash
    this._minimizeBtn.title = v ? "Restore" : "Minimize";
  }

  setPosition(x: number, y: number): void {
    this._posX = x;
    this._posY = y;
    this.style.transform = `translate(${x}px, ${y}px)`;
  }

  setWidth(w: number): void {
    this._width = w;
    this.style.width = `${w}px`;
  }

  getRect(): Rect {
    return {
      x: this._posX,
      y: this._posY,
      w: this._width,
      h: this.offsetHeight,
    };
  }

  /** Content container inside shadow DOM (for scrolling) */
  get bodyEl(): HTMLElement {
    return this._body;
  }

  // --- Drag logic ---

  private _startDrag(e: PointerEvent): void {
    this._dragging = true;
    this._dragOffsetX = e.clientX - this._posX;
    this._dragOffsetY = e.clientY - this._posY;
    this._titlebar.setPointerCapture(e.pointerId);
    this._titlebar.addEventListener("pointermove", this._onPointerMove);
    this._titlebar.addEventListener("pointerup", this._onPointerUp);
    this._titlebar.addEventListener("pointercancel", this._onPointerUp);
  }

  private _handlePointerMove(e: PointerEvent): void {
    if (!this._dragging) return;

    const rawX = e.clientX - this._dragOffsetX;
    const rawY = e.clientY - this._dragOffsetY;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const h = this.offsetHeight;

    // Hard clamp to viewport bounds
    const clampedX = Math.max(0, Math.min(vw - this._width, rawX));
    const clampedY = Math.max(0, Math.min(vh - h, rawY));

    const rect: Rect = {
      x: clampedX,
      y: clampedY,
      w: this._width,
      h,
    };

    const otherRects = this.getOtherRects(this.windowId);
    const snapped = computeSnap(rect, vw, vh, otherRects);

    // Re-clamp after snap (snap could push slightly out)
    const finalX = Math.max(0, Math.min(vw - this._width, snapped.x));
    const finalY = Math.max(0, Math.min(vh - h, snapped.y));

    this.setPosition(finalX, finalY);
  }

  private _handlePointerUp(e: PointerEvent): void {
    if (!this._dragging) return;
    this._dragging = false;
    this._titlebar.releasePointerCapture(e.pointerId);
    this._titlebar.removeEventListener("pointermove", this._onPointerMove);
    this._titlebar.removeEventListener("pointerup", this._onPointerUp);
    this._titlebar.removeEventListener("pointercancel", this._onPointerUp);

    // Final clamp
    const clamped = clampToViewport(
      this._posX,
      this._posY,
      this._width,
      this.offsetHeight,
      window.innerWidth,
      window.innerHeight,
    );
    this.setPosition(clamped.x, clamped.y);
    this.onPositionChange?.(clamped.x, clamped.y);
  }

  // --- Lifecycle ---

  connectedCallback(): void {
    this._titleEl.textContent = this.windowTitle;
    this._iconEl.textContent = this.icon;
    if (!this.style.width) {
      this.style.width = `${this._width}px`;
    }
  }

  static get observedAttributes(): string[] {
    return ["window-title", "icon"];
  }

  attributeChangedCallback(name: string, _old: string, val: string): void {
    if (name === "window-title") this._titleEl.textContent = val;
    if (name === "icon") this._iconEl.textContent = val;
  }
}

customElements.define("floating-window", FloatingWindow);
