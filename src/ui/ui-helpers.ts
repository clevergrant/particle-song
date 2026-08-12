/**
 * Shared helpers for building settings panel controls.
 */

import { NumInput } from "./num-input";

interface NumberInputOptions {
  label: string;
  value: number;
  setting: string;
  min?: number;
  max?: number;
  step?: number;
  width?: string;
  suffix?: string;
  onInput: (value: number) => void;
}

/**
 * Creates a <num-input> custom element wired up with data-setting,
 * an input callback, and a reset-to-default button.
 */
export function createNumberGroup(opts: NumberInputOptions): NumInput & { input: HTMLInputElement } {
  const el = document.createElement("num-input") as NumInput;
  el.className = "control-group";
  el.setAttribute("label", opts.label);
  el.setAttribute("value", String(opts.value));
  el.setAttribute("setting", opts.setting);
  if (opts.min != null) el.setAttribute("min", String(opts.min));
  if (opts.max != null) el.setAttribute("max", String(opts.max));
  if (opts.step != null) el.setAttribute("step", String(opts.step));
  if (opts.width) el.setAttribute("width", opts.width);
  if (opts.suffix) el.setAttribute("suffix", opts.suffix);

  // Listen on the outer <num-input> — the inner <input>'s events bubble
  // up, so this works immediately without waiting for connectedCallback.
  el.addEventListener("input", () => {
    opts.onInput(Number(el.input.value));
  });

  return el as NumInput & { input: HTMLInputElement };
}

interface ToggleGroupOptions {
  label: string;
  checked: boolean;
  setting: string;
  onChange: (checked: boolean) => void;
}

export interface ToggleGroup {
  readonly group: HTMLDivElement;
  readonly checkbox: HTMLInputElement;
}

/**
 * Creates a `.control-group` row with a label + checkbox, wired to
 * `data-setting` and an onChange callback. Returned so callers can
 * append the group, and mutate the checkbox (e.g. sync from events).
 */
export function createToggleGroup(opts: ToggleGroupOptions): ToggleGroup {
  const group = document.createElement("div");
  group.className = "control-group";
  const label = document.createElement("label");
  label.textContent = opts.label;
  group.appendChild(label);
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = opts.checked;
  checkbox.dataset.setting = opts.setting;
  checkbox.addEventListener("change", () => opts.onChange(checkbox.checked));
  group.appendChild(checkbox);
  return { group, checkbox };
}
