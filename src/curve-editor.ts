/**
 * Bezier curve editor widget for falloff curves.
 * Renders an interactive canvas where users can add/remove control points
 * and adjust bezier handles to shape a 0→1 curve.
 */

import { clamp, evalCubicBezier } from "./math-utils";

export interface CurveNode {
  x: number;
  y: number;
  /** Handle-in offset (relative to node). Points toward previous node. */
  handleInDx: number;
  handleInDy: number;
  /** Handle-out offset (relative to node). Points toward next node. */
  handleOutDx: number;
  handleOutDy: number;
}

interface DragTarget {
  type: "node" | "handleIn" | "handleOut";
  index: number;
}

const NODE_RADIUS = 5;
const HANDLE_RADIUS = 4;
const HIT_RADIUS = 10;
const MARGIN = 14;
const LUT_SIZE = 256;

export class CurveEditor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private nodes: CurveNode[];
  private lut = new Float32Array(LUT_SIZE);
  private dragging: DragTarget | null = null;
  private hovered: DragTarget | null = null;
  private onChangeCb: ((lut: Float32Array) => void) | null = null;
  private dpr: number;
  private cssW: number;
  private cssH: number;
  private drawW: number;
  private drawH: number;
  private resizeObserver: ResizeObserver;
  private static readonly ASPECT = 228 / 160;

  constructor(container: HTMLElement) {
    this.dpr = window.devicePixelRatio || 1;
    this.cssW = 228;
    this.cssH = 160;
    this.drawW = this.cssW - MARGIN * 2;
    this.drawH = this.cssH - MARGIN * 2;

    this.canvas = document.createElement("canvas");
    this.canvas.style.width = "100%";
    this.canvas.style.height = "0";
    this.canvas.style.borderRadius = "6px";
    this.canvas.style.cursor = "crosshair";
    this.canvas.style.display = "block";
    this.canvas.width = this.cssW * this.dpr;
    this.canvas.height = this.cssH * this.dpr;
    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.scale(this.dpr, this.dpr);

    container.appendChild(this.canvas);

    // Default: linear curve from (0,0) to (1,1)
    this.nodes = defaultLinearNodes();

    this.canvas.addEventListener("mousedown", this.onMouseDown);
    this.canvas.addEventListener("mousemove", this.onMouseMove);
    window.addEventListener("mouseup", this.onMouseUp);
    this.canvas.addEventListener("contextmenu", this.onContextMenu);
    this.canvas.addEventListener("dblclick", this.onDblClick);

    // Resize canvas to match container width, maintaining aspect ratio
    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) requestAnimationFrame(() => this.resize(w));
      }
    });
    this.resizeObserver.observe(container);

    this.buildLUT();
    this.render();
  }

  private resize(containerWidth: number) {
    const w = Math.round(containerWidth);
    const h = Math.round(w / CurveEditor.ASPECT);
    if (w === this.cssW && h === this.cssH) return;

    this.cssW = w;
    this.cssH = h;
    this.drawW = w - MARGIN * 2;
    this.drawH = h - MARGIN * 2;

    this.canvas.width = w * this.dpr;
    this.canvas.height = h * this.dpr;
    this.canvas.style.height = `${h}px`;
    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.scale(this.dpr, this.dpr);
    this.render();
  }

  onChange(cb: (lut: Float32Array) => void) {
    this.onChangeCb = cb;
  }

  getLUT(): Float32Array {
    return this.lut;
  }

  toJSON(): string {
    return JSON.stringify(this.nodes);
  }

  fromJSON(json: string) {
    try {
      const parsed = JSON.parse(json);
      if (Array.isArray(parsed) && parsed.length >= 2) {
        this.nodes = parsed;
        this.buildLUT();
        this.render();
      }
    } catch {
      /* ignore bad data */
    }
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("mousedown", this.onMouseDown);
    this.canvas.removeEventListener("mousemove", this.onMouseMove);
    window.removeEventListener("mouseup", this.onMouseUp);
    this.canvas.removeEventListener("contextmenu", this.onContextMenu);
    this.canvas.removeEventListener("dblclick", this.onDblClick);
    this.canvas.remove();
  }

  // ── Coordinate transforms ───────────────────────────────────────────

  private curveToCanvas(x: number, y: number): [number, number] {
    return [MARGIN + x * this.drawW, MARGIN + (1 - y) * this.drawH];
  }

  /** Map rendered-pixel mouse position to internal coordinate space */
  private toInternal(e: MouseEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (this.cssW / rect.width),
      (e.clientY - rect.top) * (this.cssH / rect.height),
    ];
  }

  private canvasToCurve(cx: number, cy: number): [number, number] {
    return [(cx - MARGIN) / this.drawW, 1 - (cy - MARGIN) / this.drawH];
  }

  // ── Hit testing ─────────────────────────────────────────────────────

  private hitTest(cx: number, cy: number): DragTarget | null {
    // Handles first (higher priority, drawn on top)
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      if (i < this.nodes.length - 1) {
        const [hx, hy] = this.curveToCanvas(
          n.x + n.handleOutDx,
          n.y + n.handleOutDy,
        );
        if (Math.hypot(cx - hx, cy - hy) < HIT_RADIUS)
          return { type: "handleOut", index: i };
      }
      if (i > 0) {
        const [hx, hy] = this.curveToCanvas(
          n.x + n.handleInDx,
          n.y + n.handleInDy,
        );
        if (Math.hypot(cx - hx, cy - hy) < HIT_RADIUS)
          return { type: "handleIn", index: i };
      }
    }
    // Then nodes
    for (let i = 0; i < this.nodes.length; i++) {
      const [nx, ny] = this.curveToCanvas(this.nodes[i].x, this.nodes[i].y);
      if (Math.hypot(cx - nx, cy - ny) < HIT_RADIUS)
        return { type: "node", index: i };
    }
    return null;
  }

  // ── Mouse handlers ──────────────────────────────────────────────────

  private onMouseDown = (e: MouseEvent) => {
    if (e.button !== 0) return;
    const [cx, cy] = this.toInternal(e);
    const hit = this.hitTest(cx, cy);
    if (hit) {
      this.dragging = hit;
      e.preventDefault();
    }
  };

  private onMouseMove = (e: MouseEvent) => {
    const [cx, cy] = this.toInternal(e);

    // Update hover state for cursor
    const hit = this.hitTest(cx, cy);
    if (!this.dragging) {
      const prev = this.hovered;
      this.hovered = hit;
      if (hit) {
        this.canvas.style.cursor =
          hit.type === "node" ? "grab" : "pointer";
      } else {
        this.canvas.style.cursor = "crosshair";
      }
      // Redraw if hover changed (for highlight)
      if (hit?.type !== prev?.type || hit?.index !== prev?.index) {
        this.render();
      }
      return;
    }

    const [curveX, curveY] = this.canvasToCurve(cx, cy);
    const { type, index } = this.dragging;
    const node = this.nodes[index];

    if (type === "node") {
      const isFirst = index === 0;
      const isLast = index === this.nodes.length - 1;
      // Endpoint x is locked; y is locked to 0/1
      if (isFirst || isLast) return;
      // Interior: constrain x between neighbors
      const minX = this.nodes[index - 1].x + 0.005;
      const maxX = this.nodes[index + 1].x - 0.005;
      node.x = clamp(curveX, minX, maxX);
      node.y = clamp(curveY, 0, 1);
    } else if (type === "handleIn") {
      node.handleInDx = Math.min(0, curveX - node.x); // must point left
      node.handleInDy = curveY - node.y;
    } else if (type === "handleOut") {
      node.handleOutDx = Math.max(0, curveX - node.x); // must point right
      node.handleOutDy = curveY - node.y;
    }

    this.canvas.style.cursor = "grabbing";
    this.buildLUT();
    this.render();
    this.onChangeCb?.(this.lut);
  };

  private onMouseUp = () => {
    if (this.dragging) {
      this.dragging = null;
      this.canvas.style.cursor = "crosshair";
    }
  };

  private onContextMenu = (e: MouseEvent) => {
    e.preventDefault();
    const [cx, cy] = this.toInternal(e);
    const hit = this.hitTest(cx, cy);
    if (
      hit &&
      hit.type === "node" &&
      hit.index > 0 &&
      hit.index < this.nodes.length - 1
    ) {
      this.nodes.splice(hit.index, 1);
      this.buildLUT();
      this.render();
      this.onChangeCb?.(this.lut);
    }
  };

  private onDblClick = (e: MouseEvent) => {
    const [cx, cy] = this.toInternal(e);

    // Double-click on existing interior node → delete it
    const hit = this.hitTest(cx, cy);
    if (hit && hit.type === "node") {
      if (hit.index > 0 && hit.index < this.nodes.length - 1) {
        this.nodes.splice(hit.index, 1);
        this.buildLUT();
        this.render();
        this.onChangeCb?.(this.lut);
      }
      return;
    }

    // Double-click on empty space → add a new node
    const [curveX, curveY] = this.canvasToCurve(cx, cy);
    if (curveX <= 0.01 || curveX >= 0.99) return;

    // Find insertion index (maintain sorted x order)
    let insertIdx = 1;
    for (let i = 1; i < this.nodes.length; i++) {
      if (this.nodes[i].x > curveX) {
        insertIdx = i;
        break;
      }
    }

    // y = current curve value to maintain shape
    const y = this.findYForX(curveX);

    // Handles: ~1/3 distance to neighbors, horizontal
    const prevX = this.nodes[insertIdx - 1].x;
    const nextX = this.nodes[insertIdx].x;
    const hLen = Math.min((curveX - prevX) / 3, (nextX - curveX) / 3, 0.15);

    this.nodes.splice(insertIdx, 0, {
      x: curveX,
      y: clamp(curveY, 0, 1),
      handleInDx: -hLen,
      handleInDy: 0,
      handleOutDx: hLen,
      handleOutDy: 0,
    });

    this.buildLUT();
    this.render();
    this.onChangeCb?.(this.lut);
  };

  // ── Bezier math ─────────────────────────────────────────────────────

  private findYForX(targetX: number): number {
    if (targetX <= 0) return this.nodes[0].y;
    if (targetX >= 1) return this.nodes[this.nodes.length - 1].y;

    for (let i = 0; i < this.nodes.length - 1; i++) {
      const n0 = this.nodes[i];
      const n1 = this.nodes[i + 1];
      if (targetX < n0.x || targetX > n1.x) continue;

      const p0x = n0.x,
        p0y = n0.y;
      const p1x = n0.x + n0.handleOutDx,
        p1y = n0.y + n0.handleOutDy;
      const p2x = n1.x + n1.handleInDx,
        p2y = n1.y + n1.handleInDy;
      const p3x = n1.x,
        p3y = n1.y;

      // Binary search for the parametric t where bezier x == targetX
      let lo = 0,
        hi = 1;
      for (let iter = 0; iter < 32; iter++) {
        const mid = (lo + hi) / 2;
        const [bx] = evalCubicBezier(
          p0x,
          p0y,
          p1x,
          p1y,
          p2x,
          p2y,
          p3x,
          p3y,
          mid,
        );
        if (bx < targetX) lo = mid;
        else hi = mid;
      }

      const [, by] = evalCubicBezier(
        p0x,
        p0y,
        p1x,
        p1y,
        p2x,
        p2y,
        p3x,
        p3y,
        (lo + hi) / 2,
      );
      return by;
    }
    return targetX; // fallback
  }

  private buildLUT() {
    for (let i = 0; i < LUT_SIZE; i++) {
      const x = i / (LUT_SIZE - 1);
      this.lut[i] = clamp(this.findYForX(x), 0, 1);
    }
  }

  // ── Rendering ───────────────────────────────────────────────────────

  private render() {
    const ctx = this.ctx;
    const w = this.cssW;
    const h = this.cssH;

    // Background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, h);

    // Border
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // Grid lines (4×4)
    ctx.strokeStyle = "#262626";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gx = MARGIN + (i / 4) * this.drawW;
      const gy = MARGIN + (i / 4) * this.drawH;
      ctx.beginPath();
      ctx.moveTo(gx, MARGIN);
      ctx.lineTo(gx, MARGIN + this.drawH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(MARGIN, gy);
      ctx.lineTo(MARGIN + this.drawW, gy);
      ctx.stroke();
    }

    // Diagonal reference (linear baseline)
    ctx.strokeStyle = "#2e2e2e";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const [dx0, dy0] = this.curveToCanvas(0, 0);
    const [dx1, dy1] = this.curveToCanvas(1, 1);
    ctx.beginPath();
    ctx.moveTo(dx0, dy0);
    ctx.lineTo(dx1, dy1);
    ctx.stroke();
    ctx.setLineDash([]);

    // Curve from LUT
    ctx.strokeStyle = "#7abecc";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < LUT_SIZE; i++) {
      const x = i / (LUT_SIZE - 1);
      const [px, py] = this.curveToCanvas(x, this.lut[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Handles and nodes
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      const [nx, ny] = this.curveToCanvas(n.x, n.y);

      // Handle-in line + circle (skip for first node)
      if (i > 0) {
        const [hx, hy] = this.curveToCanvas(
          n.x + n.handleInDx,
          n.y + n.handleInDy,
        );
        ctx.strokeStyle = "#556";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        const isHovered =
          this.hovered?.type === "handleIn" && this.hovered.index === i;
        ctx.fillStyle = isHovered ? "#aab" : "#778";
        ctx.beginPath();
        ctx.arc(hx, hy, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }

      // Handle-out line + circle (skip for last node)
      if (i < this.nodes.length - 1) {
        const [hx, hy] = this.curveToCanvas(
          n.x + n.handleOutDx,
          n.y + n.handleOutDy,
        );
        ctx.strokeStyle = "#556";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        const isHovered =
          this.hovered?.type === "handleOut" && this.hovered.index === i;
        ctx.fillStyle = isHovered ? "#aab" : "#778";
        ctx.beginPath();
        ctx.arc(hx, hy, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node circle
      const isNodeHovered =
        this.hovered?.type === "node" && this.hovered.index === i;
      ctx.fillStyle = isNodeHovered ? "#fff" : "#ddd";
      ctx.beginPath();
      ctx.arc(nx, ny, NODE_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#7abecc";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

function defaultLinearNodes(): CurveNode[] {
  return [
    {
      x: 0,
      y: 1,
      handleInDx: 0,
      handleInDy: 0,
      handleOutDx: 0.10413354717896839,
      handleOutDy: -0.43103448275862066,
    },
    {
      x: 1,
      y: 0,
      handleInDx: -0.6753122982685615,
      handleInDy: -0.034482758620689724,
      handleOutDx: 0,
      handleOutDy: 0,
    },
  ];
}


