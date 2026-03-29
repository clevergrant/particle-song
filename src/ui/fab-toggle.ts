import type { WindowManager, WindowConfig } from "./window-manager";

const FAB_CSS = `
.fab-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 9000;
  display: flex;
  flex-direction: column-reverse;
  align-items: center;
  gap: 6px;
  pointer-events: auto;
}

.fab-button {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: var(--surface-2);
  border: 1px solid var(--border-subtle);
  color: var(--text-primary);
  font-size: 22px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s, transform 0.2s;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
  flex-shrink: 0;
}

.fab-button:hover {
  background: var(--border-subtle);
}

.fab-button.open {
  transform: rotate(45deg);
}

.fab-menu {
  display: flex;
  flex-direction: column-reverse;
  align-items: center;
  gap: 4px;
  opacity: 0;
  transform: translateY(8px);
  pointer-events: none;
  transition: opacity 0.2s, transform 0.2s;
}

.fab-menu.open {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

.fab-divider {
  width: 28px;
  height: 1px;
  background: var(--toggle-track);
  margin: 2px auto;
}

.fab-item {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--surface-1);
  border: 1px solid var(--border-subtle);
  color: var(--toggle-knob);
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
  position: relative;
  flex-shrink: 0;
}

.fab-item:hover {
  background: var(--surface-2);
  color: var(--text-secondary);
}

.fab-item.active {
  background: var(--accent-bg);
  border-color: var(--accent);
  color: var(--accent-hover);
}

.fab-item .fab-tooltip {
  position: absolute;
  right: 52px;
  white-space: nowrap;
  background: var(--surface-1);
  border: 1px solid var(--toggle-track);
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 11px;
  color: var(--text-secondary);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s;
  font-family: system-ui, -apple-system, sans-serif;
}

.fab-item:hover .fab-tooltip {
  opacity: 1;
}

.fab-reset {
  width: 32px;
  height: 32px;
  font-size: 13px;
  background: var(--surface-1);
  border: 1px solid var(--toggle-track);
}

.fab-reset:hover {
  background: var(--danger-bg-hover);
  border-color: var(--danger-border);
  color: var(--danger-text);
}
`;

export class FabToggle {
  private _container: HTMLElement;
  private _manager: WindowManager;
  private _menuEl: HTMLElement;
  private _fabBtn: HTMLButtonElement;
  private _items = new Map<string, HTMLButtonElement>();
  private _isOpen = false;

  constructor(parent: HTMLElement, manager: WindowManager) {
    this._manager = manager;

    // Inject styles
    const style = document.createElement("style");
    style.textContent = FAB_CSS;
    document.head.appendChild(style);

    this._container = document.createElement("div");
    this._container.className = "fab-container";

    // FAB button
    this._fabBtn = document.createElement("button");
    this._fabBtn.className = "fab-button";
    this._fabBtn.textContent = "+";
    this._fabBtn.title = "Toggle windows";
    this._fabBtn.addEventListener("click", () => this._toggle());

    // Menu
    this._menuEl = document.createElement("div");
    this._menuEl.className = "fab-menu";

    this._container.append(this._fabBtn, this._menuEl);
    parent.appendChild(this._container);

    // Close on Escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this._isOpen) this._close();
    });

    // Close on outside click
    document.addEventListener("pointerdown", (e) => {
      if (this._isOpen && !this._container.contains(e.target as Node)) {
        this._close();
      }
    });

    // Listen for visibility changes from WindowManager
    manager.onVisibilityChange = (id, visible) => {
      const btn = this._items.get(id);
      if (btn) btn.classList.toggle("active", visible);
    };
  }

  /** Build the menu items from registered windows. Call after all windows are registered. */
  build(): void {
    this._menuEl.innerHTML = "";
    this._items.clear();

    const configs = this._manager.getConfigs();

    // Group by category
    const categories = new Map<string, WindowConfig[]>();
    for (const c of configs) {
      const cat = c.category ?? "other";
      if (!categories.has(cat)) categories.set(cat, []);
      categories.get(cat)!.push(c);
    }

    // Desired category order
    const order = ["system", "simulation", "music", "detection", "visual"];
    const sortedCats = [...categories.keys()].sort(
      (a, b) => (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) -
                (order.indexOf(b) === -1 ? 99 : order.indexOf(b)),
    );

    let first = true;
    for (const cat of sortedCats) {
      if (!first) {
        const div = document.createElement("div");
        div.className = "fab-divider";
        this._menuEl.appendChild(div);
      }
      first = false;

      for (const config of categories.get(cat)!) {
        const btn = document.createElement("button");
        btn.className = "fab-item";
        if (this._manager.isVisible(config.id)) {
          btn.classList.add("active");
        }
        btn.textContent = config.icon;

        const tooltip = document.createElement("span");
        tooltip.className = "fab-tooltip";
        tooltip.textContent = config.title;
        btn.appendChild(tooltip);

        btn.addEventListener("click", () => {
          this._manager.toggle(config.id);
        });

        this._items.set(config.id, btn);
        this._menuEl.appendChild(btn);
      }
    }

    // Reset layout button at the top
    const resetBtn = document.createElement("button");
    resetBtn.className = "fab-item fab-reset";
    resetBtn.textContent = "\u21BA"; // counterclockwise arrow
    const resetTooltip = document.createElement("span");
    resetTooltip.className = "fab-tooltip";
    resetTooltip.textContent = "Reset layout";
    resetBtn.appendChild(resetTooltip);
    resetBtn.addEventListener("click", () => {
      this._manager.resetLayout();
    });
    this._menuEl.appendChild(resetBtn);
  }

  private _toggle(): void {
    this._isOpen ? this._close() : this._open();
  }

  private _open(): void {
    this._isOpen = true;
    this._fabBtn.classList.add("open");
    this._menuEl.classList.add("open");
  }

  private _close(): void {
    this._isOpen = false;
    this._fabBtn.classList.remove("open");
    this._menuEl.classList.remove("open");
  }
}
