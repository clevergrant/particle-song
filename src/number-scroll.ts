/**
 * Scroll-to-adjust on number inputs: hover over a number input and
 * scroll the mouse wheel to increment/decrement the value.
 * Prevents the settings panel from scrolling while adjusting.
 */

export function attachNumberScroll(container: HTMLElement) {
  container.addEventListener("wheel", onWheel, { passive: false });
}

export function detachNumberScroll(container: HTMLElement) {
  container.removeEventListener("wheel", onWheel);
}

/** Compute the number of decimal places needed for a given step size. */
export function stepDecimals(step: number): number {
  return step < 1 ? (String(step).split(".")[1]?.length ?? 0) : 0;
}

/** Apply a stepped delta to a number input, clamping and rounding as needed. */
export function applyStepDelta(input: HTMLInputElement, delta: number) {
  const step = Number(input.step) || 1;
  const min = input.min !== "" ? Number(input.min) : -Infinity;
  const max = input.max !== "" ? Number(input.max) : Infinity;
  const decimals = stepDecimals(step);

  let newVal = (Number(input.value) || 0) + delta * step;
  newVal = Math.min(max, Math.max(min, newVal));
  if (decimals > 0) newVal = Number(newVal.toFixed(decimals));

  input.value = String(newVal);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

function onWheel(e: WheelEvent) {
  const input = e.target as HTMLInputElement;
  if (input.tagName !== "INPUT" || input.type !== "number") return;

  e.preventDefault();
  e.stopPropagation();

  const direction = e.deltaY > 0 ? -1 : e.deltaY < 0 ? 1 : 0;
  if (direction === 0) return;

  applyStepDelta(input, direction);
}
