/** Pure snap calculation functions for window positioning */

export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

export interface SnapResult {
  readonly x: number;
  readonly y: number;
  readonly snappedX: boolean;
  readonly snappedY: boolean;
}

const SNAP_THRESHOLD = 8;

/**
 * Snap a rect to viewport edges. Returns snapped position.
 */
function snapToEdges(
  rect: Rect,
  viewportW: number,
  viewportH: number,
  threshold: number,
): { x: number | null; y: number | null } {
  let x: number | null = null;
  let y: number | null = null;

  // Left edge
  if (Math.abs(rect.x) < threshold) x = 0;
  // Right edge
  else if (Math.abs(rect.x + rect.w - viewportW) < threshold)
    x = viewportW - rect.w;

  // Top edge
  if (Math.abs(rect.y) < threshold) y = 0;
  // Bottom edge
  else if (Math.abs(rect.y + rect.h - viewportH) < threshold)
    y = viewportH - rect.h;

  return { x, y };
}

/**
 * Snap a rect to other window edges. Returns snapped position.
 * Checks all four edges of each other rect against all four edges of this rect.
 */
function snapToWindows(
  rect: Rect,
  others: readonly Rect[],
  threshold: number,
): { x: number | null; y: number | null } {
  let bestX: number | null = null;
  let bestDx = threshold;
  let bestY: number | null = null;
  let bestDy = threshold;

  for (const other of others) {
    // Horizontal snaps: left-to-left, left-to-right, right-to-left, right-to-right
    const pairs_x = [
      { drag: rect.x, target: other.x, offset: 0 },
      { drag: rect.x, target: other.x + other.w, offset: 0 },
      { drag: rect.x + rect.w, target: other.x, offset: -rect.w },
      { drag: rect.x + rect.w, target: other.x + other.w, offset: -rect.w },
    ];

    for (const p of pairs_x) {
      const d = Math.abs(p.drag - p.target);
      if (d < bestDx) {
        bestDx = d;
        bestX = p.target + p.offset;
      }
    }

    // Vertical snaps: top-to-top, top-to-bottom, bottom-to-top, bottom-to-bottom
    const pairs_y = [
      { drag: rect.y, target: other.y, offset: 0 },
      { drag: rect.y, target: other.y + other.h, offset: 0 },
      { drag: rect.y + rect.h, target: other.y, offset: -rect.h },
      { drag: rect.y + rect.h, target: other.y + other.h, offset: -rect.h },
    ];

    for (const p of pairs_y) {
      const d = Math.abs(p.drag - p.target);
      if (d < bestDy) {
        bestDy = d;
        bestY = p.target + p.offset;
      }
    }
  }

  return { x: bestX, y: bestY };
}

/**
 * Compute final snapped position. Viewport edges take priority over window edges.
 */
export function computeSnap(
  rect: Rect,
  viewportW: number,
  viewportH: number,
  otherRects: readonly Rect[],
  threshold: number = SNAP_THRESHOLD,
): SnapResult {
  const edge = snapToEdges(rect, viewportW, viewportH, threshold);
  const win = snapToWindows(rect, otherRects, threshold);

  // Viewport edges take priority
  const x = edge.x ?? win.x ?? rect.x;
  const y = edge.y ?? win.y ?? rect.y;

  return {
    x,
    y,
    snappedX: edge.x !== null || win.x !== null,
    snappedY: edge.y !== null || win.y !== null,
  };
}

/**
 * Clamp position so the entire window stays within the viewport.
 */
export function clampToViewport(
  x: number,
  y: number,
  w: number,
  h: number,
  viewportW: number,
  viewportH: number,
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(viewportW - w, x)),
    y: Math.max(0, Math.min(viewportH - h, y)),
  };
}
