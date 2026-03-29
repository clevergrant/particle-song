/**
 * Shared helpers for building settings panel controls.
 */

import { NumInput } from "./num-input";

export interface NumberInputOptions {
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
