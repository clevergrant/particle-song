import { FloatingWindow } from "./floating-window";
import { clampToViewport, type Rect } from "./snap";

export interface WindowConfig {
  readonly id: string;
  readonly title: string;
  readonly icon: string;
  readonly category?: string;
  readonly defaultVisible?: boolean;
  readonly defaultPosition?: { readonly x: number; readonly y: number };
  readonly defaultWidth?: number;
}

interface PersistedLayout {
  positions: Record<string, { x: number; y: number }>;
  visibility: Record<string, boolean>;
  minimized: Record<string, boolean>;
  zOrder: string[];
}

interface WindowEntry {
  readonly config: WindowConfig;
  readonly el: FloatingWindow;
  readonly contentEl: HTMLElement;
}

export class WindowManager {
  private _windows = new Map<string, WindowEntry>();
  private _zOrder: string[] = [];
  private _container: HTMLElement;
  private _storageKey: string;
  private _zBase = 100;
  private _saveTimeout: ReturnType<typeof setTimeout> | null = null;

  /** Fired when a window is toggled. Listeners can update FAB state. */
  onVisibilityChange: ((id: string, visible: boolean) => void) | null = null;

  constructor(container: HTMLElement, simulationName: string) {
    this._container = container;
    this._storageKey = `particle-sim:window-layout:${simulationName}`;

    window.addEventListener("resize", () => this._clampAll());
  }

  /** Change storage key when switching simulations */
  setSimulation(name: string): void {
    this._storageKey = `particle-sim:window-layout:${name}`;
  }

  register(config: WindowConfig, contentEl: HTMLElement): FloatingWindow {
    const el = document.createElement("floating-window") as FloatingWindow;
    el.setAttribute("window-id", config.id);
    el.setAttribute("window-title", config.title);
    el.setAttribute("icon", config.icon);

    if (config.defaultWidth) {
      el.setWidth(config.defaultWidth);
    }

    // Append content as light DOM child (preserves data-setting queryability)
    el.appendChild(contentEl);

    // Wire up callbacks
    el.getOtherRects = (excludeId) => this._getVisibleRects(excludeId);
    el.onClose = () => this.hide(config.id);
    el.onFocus = () => this.bringToFront(config.id);
    el.onPositionChange = () => this._scheduleSave();

    // Default hidden until layout is applied
    el.hidden = true;

    this._container.appendChild(el);

    const entry: WindowEntry = { config, el, contentEl };
    this._windows.set(config.id, entry);
    this._zOrder.push(config.id);

    return el;
  }

  /** Apply saved layout or defaults. Call after all windows are registered. */
  applyLayout(): void {
    const saved = this._loadLayout();

    for (const [id, entry] of this._windows) {
      const { config, el } = entry;

      // Position
      const pos = saved?.positions[id] ?? config.defaultPosition ?? { x: 50, y: 50 };
      el.setPosition(pos.x, pos.y);

      // Visibility
      const visible = saved?.visibility[id] ?? config.defaultVisible ?? false;
      el.hidden = !visible;

      // Minimized
      if (saved?.minimized[id]) {
        el.minimized = true;
      }
    }

    // Z-order
    if (saved?.zOrder) {
      this._zOrder = saved.zOrder.filter((id) => this._windows.has(id));
      // Add any new windows not in saved order
      for (const id of this._windows.keys()) {
        if (!this._zOrder.includes(id)) this._zOrder.push(id);
      }
    }
    this._applyZOrder();
    this._clampAll();
  }

  bringToFront(id: string): void {
    const idx = this._zOrder.indexOf(id);
    if (idx >= 0) this._zOrder.splice(idx, 1);
    this._zOrder.push(id);
    this._applyZOrder();
    this._scheduleSave();
  }

  toggle(id: string): void {
    const entry = this._windows.get(id);
    if (!entry) return;
    if (entry.el.hidden) {
      this.show(id);
    } else {
      this.hide(id);
    }
  }

  show(id: string): void {
    const entry = this._windows.get(id);
    if (!entry) return;
    entry.el.hidden = false;
    this.bringToFront(id);
    this.onVisibilityChange?.(id, true);
    this._scheduleSave();
  }

  hide(id: string): void {
    const entry = this._windows.get(id);
    if (!entry) return;
    entry.el.hidden = true;
    this.onVisibilityChange?.(id, false);
    this._scheduleSave();
  }

  isVisible(id: string): boolean {
    return !this._windows.get(id)?.el.hidden;
  }

  /** Get all content elements (for settings save/load sweep) */
  getAllContentElements(): HTMLElement[] {
    return [...this._windows.values()].map((e) => e.contentEl);
  }

  /** Get a specific window's content element */
  getContentElement(id: string): HTMLElement | undefined {
    return this._windows.get(id)?.contentEl;
  }

  /** Get the FloatingWindow element */
  getWindow(id: string): FloatingWindow | undefined {
    return this._windows.get(id)?.el;
  }

  /** Get all registered window configs (for FAB menu building) */
  getConfigs(): WindowConfig[] {
    return [...this._windows.values()].map((e) => e.config);
  }

  /** Remove all windows (for simulation switch) */
  clearAll(): void {
    for (const entry of this._windows.values()) {
      entry.el.remove();
    }
    this._windows.clear();
    this._zOrder = [];
  }

  resetLayout(): void {
    for (const [id, entry] of this._windows) {
      const { config, el } = entry;
      const pos = config.defaultPosition ?? { x: 50, y: 50 };
      el.setPosition(pos.x, pos.y);
      el.hidden = !(config.defaultVisible ?? false);
      el.minimized = false;
      this.onVisibilityChange?.(id, !el.hidden);
    }
    this._zOrder = [...this._windows.keys()];
    this._applyZOrder();
    this._clampAll();
    this._save();
  }

  // --- Private ---

  private _getVisibleRects(excludeId: string): readonly Rect[] {
    const rects: Rect[] = [];
    for (const [id, entry] of this._windows) {
      if (id === excludeId || entry.el.hidden) continue;
      rects.push(entry.el.getRect());
    }
    return rects;
  }

  private _applyZOrder(): void {
    for (let i = 0; i < this._zOrder.length; i++) {
      const entry = this._windows.get(this._zOrder[i]);
      if (entry) {
        entry.el.style.zIndex = String(this._zBase + i);
      }
    }
  }

  private _clampAll(): void {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    for (const entry of this._windows.values()) {
      if (entry.el.hidden) continue;
      const r = entry.el.getRect();
      const c = clampToViewport(r.x, r.y, r.w, r.h, vw, vh);
      entry.el.setPosition(c.x, c.y);
    }
  }

  private _scheduleSave(): void {
    if (this._saveTimeout) clearTimeout(this._saveTimeout);
    this._saveTimeout = setTimeout(() => this._save(), 200);
  }

  private _save(): void {
    const layout: PersistedLayout = {
      positions: {},
      visibility: {},
      minimized: {},
      zOrder: this._zOrder,
    };
    for (const [id, entry] of this._windows) {
      layout.positions[id] = { x: entry.el.posX, y: entry.el.posY };
      layout.visibility[id] = !entry.el.hidden;
      layout.minimized[id] = entry.el.minimized;
    }
    try {
      localStorage.setItem(this._storageKey, JSON.stringify(layout));
    } catch {
      // Storage full or unavailable
    }
  }

  private _loadLayout(): PersistedLayout | null {
    try {
      const raw = localStorage.getItem(this._storageKey);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
}
