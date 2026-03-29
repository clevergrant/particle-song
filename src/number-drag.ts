/**
 * Godot-style click-and-drag on number inputs to slide values up/down.
 * Hold-click and drag horizontally to adjust the value.
 */

import { stepDecimals } from "./number-scroll";

const DRAG_THRESHOLD = 3; // px before drag starts (avoids hijacking normal clicks)

export function attachNumberDrag(container: HTMLElement) {
  container.addEventListener("pointerdown", onPointerDown);
}

export function detachNumberDrag(container: HTMLElement) {
  container.removeEventListener("pointerdown", onPointerDown);
}

function onPointerDown(e: PointerEvent) {
  const input = e.target as HTMLInputElement;
  if (input.tagName !== "INPUT" || input.type !== "number") return;

  const startX = e.clientX;
  const startValue = Number(input.value) || 0;
  const step = Number(input.step) || 1;
  const min = input.min !== "" ? Number(input.min) : -Infinity;
  const max = input.max !== "" ? Number(input.max) : Infinity;

  // Scale: pixels per step. Finer steps get more px-per-step so dragging feels natural.
  const pxPerStep = step < 1 ? Math.max(4, 2 / step) : 2;

  let dragging = false;

  function onMove(ev: PointerEvent) {
    const dx = ev.clientX - startX;

    if (!dragging) {
      if (Math.abs(dx) < DRAG_THRESHOLD) return;
      dragging = true;
      input.setPointerCapture(ev.pointerId);
      input.style.cursor = "ew-resize";
      document.body.style.cursor = "ew-resize";
      // Prevent text selection while dragging
      input.blur();
    }

    const steps = Math.round(dx / pxPerStep);
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
      // Prevent the click from focusing/selecting the input after drag
      ev.preventDefault();
    }
    // If not dragging, normal click behavior proceeds as usual
  }

  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
}
