/**
 * Custom color picker with synced hex, RGB, and HSV text inputs.
 * Popup attaches to document.body to avoid scroll clipping.
 */

import { clamp } from "./math-utils";

// ── Color math ──────────────────────────────────────────────────────

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60)       { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else              { r = c; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === r)      h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else                h = 60 * ((r - g) / d + 4);
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  return [h, s, max];
}

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.replace("#", "");
  return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
}

function rgbToHex(r: number, g: number, b: number): string {
  return "#" + [r, g, b].map(c => clamp(c, 0, 255).toString(16).padStart(2, "0")).join("");
}

// ── Module-level state ──────────────────────────────────────────────

let activePicker: ColorPicker | null = null;

// ── Class ───────────────────────────────────────────────────────────

export class ColorPicker {
  readonly element: HTMLDivElement;
  readonly input: HTMLInputElement;

  private h = 0;    // 0–360
  private s = 1;    // 0–1
  private v = 1;    // 0–1

  private popup: HTMLDivElement | null = null;
  private hexInput: HTMLInputElement | null = null;
  private rInput: HTMLInputElement | null = null;
  private gInput: HTMLInputElement | null = null;
  private bInput: HTMLInputElement | null = null;
  private hInput: HTMLInputElement | null = null;
  private sInput: HTMLInputElement | null = null;
  private vInput: HTMLInputElement | null = null;
  private preview: HTMLDivElement | null = null;
  private nativeInput: HTMLInputElement | null = null;

  private onChangeCb: ((hex: string) => void) | null = null;
  private outsideClickHandler: ((e: MouseEvent) => void) | null = null;

  constructor(initialColor = "#ffffff") {
    const [r, g, b] = hexToRgb(initialColor);
    [this.h, this.s, this.v] = rgbToHsv(r, g, b);

    this.element = document.createElement("div");
    this.element.className = "color-picker-swatch";
    this.element.style.background = initialColor;
    this.element.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.popup) this.close();
      else this.open();
    });

    // Hidden input for data-setting save/load compatibility
    this.input = document.createElement("input");
    this.input.type = "hidden";
    this.input.value = initialColor;
    this.input.addEventListener("input", () => {
      const hex = this.input.value;
      if (/^#[0-9a-f]{6}$/i.test(hex)) {
        const [r, g, b] = hexToRgb(hex);
        [this.h, this.s, this.v] = rgbToHsv(r, g, b);
        this.element.style.background = hex;
        if (this.popup) this.syncAllInputs();
      }
    });
    this.element.appendChild(this.input);
  }

  get value(): string { return this.input.value; }
  set value(hex: string) {
    this.input.value = hex;
    const [r, g, b] = hexToRgb(hex);
    [this.h, this.s, this.v] = rgbToHsv(r, g, b);
    this.element.style.background = hex;
    if (this.popup) this.syncAllInputs();
  }

  onChange(cb: (hex: string) => void) { this.onChangeCb = cb; }

  destroy() {
    this.close();
    this.element.remove();
  }

  // ── Popup lifecycle ─────────────────────────────────────────────

  open() {
    if (activePicker && activePicker !== this) activePicker.close();
    activePicker = this;

    this.popup = document.createElement("div");
    this.popup.className = "color-picker-popup";

    // Color preview – click to open native browser color picker
    this.preview = document.createElement("div");
    this.preview.className = "color-picker-preview";
    this.preview.title = "Click to open color picker";
    this.preview.style.cursor = "pointer";

    this.nativeInput = document.createElement("input");
    this.nativeInput.type = "color";
    this.nativeInput.style.position = "absolute";
    this.nativeInput.style.opacity = "0";
    this.nativeInput.style.pointerEvents = "none";
    this.nativeInput.style.width = "0";
    this.nativeInput.style.height = "0";
    this.nativeInput.value = this.input.value;
    this.preview.appendChild(this.nativeInput);

    this.preview.addEventListener("click", () => this.nativeInput?.click());
    this.nativeInput.addEventListener("input", () => {
      const hex = this.nativeInput!.value;
      const [r, g, b] = hexToRgb(hex);
      [this.h, this.s, this.v] = rgbToHsv(r, g, b);
      this.applyColor();
    });

    this.popup.appendChild(this.preview);

    // Hex row
    this.hexInput = this.makeField(this.popup, "HEX", 7);
    this.hexInput.addEventListener("change", () => this.onHexEdit());

    // RGB row
    const [rgbRow, rgbInputs] = this.makeGroupRow("RGB", 3);
    [this.rInput, this.gInput, this.bInput] = rgbInputs;
    for (const inp of rgbInputs) inp.addEventListener("change", () => this.onRgbEdit());
    this.popup.appendChild(rgbRow);

    // HSV row
    const [hsvRow, hsvInputs] = this.makeGroupRow("HSV", 3);
    [this.hInput, this.sInput, this.vInput] = hsvInputs;
    for (const inp of hsvInputs) inp.addEventListener("change", () => this.onHsvEdit());
    this.popup.appendChild(hsvRow);

    document.body.appendChild(this.popup);
    this.positionPopup();
    this.syncAllInputs();

    // Pin the settings panel open while the picker is visible
    document.getElementById("controls-wrapper")?.classList.add("pinned");

    requestAnimationFrame(() => {
      this.outsideClickHandler = (e: MouseEvent) => {
        if (this.popup && !this.popup.contains(e.target as Node) && !this.element.contains(e.target as Node)) {
          this.close();
        }
      };
      document.addEventListener("mousedown", this.outsideClickHandler);
    });
  }

  close() {
    if (this.popup) {
      this.popup.remove();
      this.popup = null;
      this.preview = null;
      this.nativeInput = null;
      this.hexInput = null;
      this.rInput = this.gInput = this.bInput = null;
      this.hInput = this.sInput = this.vInput = null;
    }
    if (this.outsideClickHandler) {
      document.removeEventListener("mousedown", this.outsideClickHandler);
      this.outsideClickHandler = null;
    }
    if (activePicker === this) activePicker = null;

    // Unpin the settings panel unless the user has manually pinned it
    const pinToggle = document.getElementById("pin-toggle");
    if (!pinToggle?.classList.contains("active")) {
      document.getElementById("controls-wrapper")?.classList.remove("pinned");
    }
  }

  // ── Positioning ─────────────────────────────────────────────────

  private positionPopup() {
    if (!this.popup) return;
    const rect = this.element.getBoundingClientRect();
    const popupRect = this.popup.getBoundingClientRect();

    let top = rect.bottom + 6;
    let left = rect.left;

    if (top + popupRect.height > window.innerHeight) {
      top = rect.top - popupRect.height - 6;
    }
    if (left + popupRect.width > window.innerWidth) {
      left = window.innerWidth - popupRect.width - 8;
    }
    if (left < 4) left = 4;
    if (top < 4) top = 4;

    this.popup.style.left = `${left}px`;
    this.popup.style.top = `${top}px`;
  }

  // ── DOM helpers ─────────────────────────────────────────────────

  private makeField(parent: HTMLElement, label: string, maxLen: number): HTMLInputElement {
    const wrap = document.createElement("div");
    wrap.className = "color-picker-field";
    const lbl = document.createElement("span");
    lbl.className = "color-picker-label";
    lbl.textContent = label;
    wrap.appendChild(lbl);
    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = "color-picker-text";
    inp.maxLength = maxLen;
    wrap.appendChild(inp);
    parent.appendChild(wrap);
    return inp;
  }

  private makeGroupRow(label: string, count: number): [HTMLDivElement, HTMLInputElement[]] {
    const row = document.createElement("div");
    row.className = "color-picker-field-row";
    const lbl = document.createElement("span");
    lbl.className = "color-picker-label";
    lbl.textContent = label;
    row.appendChild(lbl);
    const inputs: HTMLInputElement[] = [];
    for (let i = 0; i < count; i++) {
      const inp = document.createElement("input");
      inp.type = "text";
      inp.className = "color-picker-text";
      inp.maxLength = 3;
      row.appendChild(inp);
      inputs.push(inp);
    }
    return [row, inputs];
  }

  // ── Sync all text inputs + preview from HSV state ───────────────

  private syncAllInputs() {
    const [r, g, b] = hsvToRgb(this.h, this.s, this.v);
    const hex = rgbToHex(r, g, b);

    if (this.hexInput) this.hexInput.value = hex;
    if (this.rInput) this.rInput.value = String(r);
    if (this.gInput) this.gInput.value = String(g);
    if (this.bInput) this.bInput.value = String(b);
    if (this.hInput) this.hInput.value = String(Math.round(this.h));
    if (this.sInput) this.sInput.value = String(Math.round(this.s * 100));
    if (this.vInput) this.vInput.value = String(Math.round(this.v * 100));
    if (this.preview) this.preview.style.background = hex;
    if (this.nativeInput) this.nativeInput.value = hex;
  }

  // ── Input handlers ──────────────────────────────────────────────

  private onHexEdit() {
    let val = this.hexInput?.value.trim() ?? "";
    if (!val.startsWith("#")) val = "#" + val;
    if (/^#[0-9a-f]{6}$/i.test(val)) {
      const [r, g, b] = hexToRgb(val);
      [this.h, this.s, this.v] = rgbToHsv(r, g, b);
      this.applyColor();
    }
  }

  private onRgbEdit() {
    const r = clamp(parseInt(this.rInput?.value ?? "0") || 0, 0, 255);
    const g = clamp(parseInt(this.gInput?.value ?? "0") || 0, 0, 255);
    const b = clamp(parseInt(this.bInput?.value ?? "0") || 0, 0, 255);
    [this.h, this.s, this.v] = rgbToHsv(r, g, b);
    this.applyColor();
  }

  private onHsvEdit() {
    this.h = clamp(parseInt(this.hInput?.value ?? "0") || 0, 0, 360);
    this.s = clamp((parseInt(this.sInput?.value ?? "100") || 0) / 100, 0, 1);
    this.v = clamp((parseInt(this.vInput?.value ?? "100") || 0) / 100, 0, 1);
    this.applyColor();
  }

  // ── Apply color to all outputs ──────────────────────────────────

  private applyColor() {
    const [r, g, b] = hsvToRgb(this.h, this.s, this.v);
    const hex = rgbToHex(r, g, b);

    this.element.style.background = hex;
    this.input.value = hex;
    this.input.dispatchEvent(new Event("input", { bubbles: true }));

    this.syncAllInputs();
    this.onChangeCb?.(hex);
  }
}
