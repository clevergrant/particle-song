/**
 * Click-and-drag on number inputs to slide values up/down.
 * Hold-click and drag vertically to adjust the value.
 */

import { DRAG_THRESHOLD } from "../constants";
import { stepDecimals } from "./number-scroll";

export function attachNumberDrag(container: HTMLElement) {
  container.addEventListener("pointerdown", onPointerDown);
}

function onPointerDown(e: PointerEvent) {
  const input = e.target as HTMLInputElement;
  if (input.tagName !== "INPUT" || input.type !== "number") return;

  const startY = e.clientY;
  const startValue = Number(input.value) || 0;
  const step = Number(input.step) || 1;
  const min = input.min !== "" ? Number(input.min) : -Infinity;
  const max = input.max !== "" ? Number(input.max) : Infinity;

  // Scale: pixels per step. Finer steps get more px-per-step so dragging feels natural.
  const pxPerStep = step < 1 ? Math.max(4, 2 / step) : 2;

  let dragging = false;

  function onMove(ev: PointerEvent) {
    // Negative dy = dragging up = increase value
    const dy = startY - ev.clientY;

    if (!dragging) {
      if (Math.abs(dy) < DRAG_THRESHOLD) return;
      dragging = true;
      input.setPointerCapture(ev.pointerId);
      input.style.cursor = "ns-resize";
      document.body.style.cursor = "ns-resize";
      input.blur();
    }

    const steps = Math.round(dy / pxPerStep);
    const decimals = stepDecimals(step);
    let newVal = startValue + steps * step;
    newVal = Math.min(max, Math.max(min, newVal));
    if (decimals > 0) newVal = Number(newVal.toFixed(decimals));

    input.value = String(newVal);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function onUp(ev: PointerEvent) {
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);

    if (dragging) {
      input.releasePointerCapture(ev.pointerId);
      input.style.cursor = "";
      document.body.style.cursor = "";
      ev.preventDefault();
    }
  }

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
}
