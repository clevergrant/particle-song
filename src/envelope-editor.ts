/**
 * 4-section ADSR bezier envelope editor.
 *
 * - Attack: bezier curve, rises from 0 to peak level
 * - Decay: bezier curve, falls from peak level to sustain level
 * - Sustain: flat horizontal line at sustain level (duration is gate-driven)
 * - Release: bezier curve, falls from sustain level to 0
 * - Section proportions adjustable by dragging vertical dividers
 * - Double-click to add nodes, right-click to remove interior nodes
 * - Sustain section has no editable nodes (level set by decay's last node y)
 */

import type { EnvelopeShape, EnvelopeSection, EnvelopeNode } from "./music/types";
import { clamp, evalCubicBezier } from "./math-utils";

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const NODE_RADIUS = 5;
const HANDLE_RADIUS = 4;
const HIT_RADIUS = 10;
const MARGIN = 14;
const LUT_SIZE = 256;
const DIVIDER_HIT_W = 8; // pixels either side of divider for hit testing

/** Section keys for the editable bezier sections (attack, decay, release).
 *  Sustain is not a bezier section — it's a flat line derived from decay's end y. */
type EditableSectionKey = "attack" | "decay" | "release";
const EDITABLE_SECTION_KEYS: readonly EditableSectionKey[] = ["attack", "decay", "release"];

/** All 4 visual sections for proportions, rendering, and labels. */
type SectionKey = "attack" | "decay" | "sustain" | "release";
const SECTION_KEYS: readonly SectionKey[] = ["attack", "decay", "sustain", "release"];

const SECTION_COLORS: Record<SectionKey, string> = {
  attack: "rgba(120, 200, 120, 0.06)",
  decay: "rgba(120, 160, 200, 0.06)",
  sustain: "rgba(200, 200, 120, 0.06)",
  release: "rgba(200, 120, 120, 0.06)",
};

const SECTION_CURVE_COLORS: Record<SectionKey, string> = {
  attack: "#7acc7a",
  decay: "#7aaacc",
  sustain: "#cccc7a",
  release: "#cc7a7a",
};

const SECTION_LABELS: readonly string[] = ["A", "D", "S", "R"];

/* ------------------------------------------------------------------ */
/*  Drag targets                                                       */
/* ------------------------------------------------------------------ */

interface NodeDrag {
  type: "node" | "handleIn" | "handleOut";
  section: EditableSectionKey;
  index: number;
}

interface DividerDrag {
  type: "divider";
  /** Index of divider: 0 = A|D, 1 = D|S, 2 = S|R */
  dividerIndex: number;
}

type DragTarget = NodeDrag | DividerDrag;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function makeNode(
  x: number, y: number,
  hiDx = 0, hiDy = 0,
  hoDx = 0, hoDy = 0,
): EnvelopeNode {
  return { x, y, handleInDx: hiDx, handleInDy: hiDy, handleOutDx: hoDx, handleOutDy: hoDy };
}

/* ------------------------------------------------------------------ */
/*  Default shapes                                                     */
/* ------------------------------------------------------------------ */

function defaultAttackNodes(): EnvelopeNode[] {
  return [
    makeNode(0, 0, 0, 0, 0.3, 0),
    makeNode(1, 0.8, -0.3, 0, 0, 0),
  ];
}

function defaultDecayNodes(): EnvelopeNode[] {
  return [
    makeNode(0, 0.8, 0, 0, 0.3, 0),
    makeNode(1, 0.6, -0.3, 0, 0, 0),
  ];
}

function defaultReleaseNodes(): EnvelopeNode[] {
  return [
    makeNode(0, 0.6, 0, 0, 0.3, 0.1),
    makeNode(1, 0, -0.3, 0.1, 0, 0),
  ];
}

/* ------------------------------------------------------------------ */
/*  EnvelopeEditor                                                     */
/* ------------------------------------------------------------------ */

export class EnvelopeEditor {
  private canvas: HTMLCanvasElement;
  private ctx2d: CanvasRenderingContext2D;
  private dpr: number;
  private cssW: number;
  private cssH: number;
  private drawW: number;
  private drawH: number;
  private resizeObserver: ResizeObserver;
  private static readonly ASPECT = 340 / 160;

  // Editable bezier sections (attack, decay, release). Sustain has no nodes.
  private sections: Record<EditableSectionKey, EnvelopeNode[]> = {
    attack: defaultAttackNodes(),
    decay: defaultDecayNodes(),
    release: defaultReleaseNodes(),
  };

  // Section proportions for all 4 sections (sum to 1)
  private proportions: Record<SectionKey, number> = {
    attack: 0.20,
    decay: 0.20,
    sustain: 0.30,
    release: 0.30,
  };

  private lut = new Float32Array(LUT_SIZE);
  private dragging: DragTarget | null = null;
  private hovered: DragTarget | null = null;
  private onChangeCb: ((shape: EnvelopeShape, lut: Float32Array) => void) | null = null;

  constructor(container: HTMLElement) {
    this.dpr = window.devicePixelRatio || 1;
    this.cssW = 340;
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
    this.ctx2d = this.canvas.getContext("2d")!;
    this.ctx2d.scale(this.dpr, this.dpr);

    container.appendChild(this.canvas);

    this.canvas.addEventListener("mousedown", this.onMouseDown);
    this.canvas.addEventListener("mousemove", this.onMouseMove);
    window.addEventListener("mouseup", this.onMouseUp);
    this.canvas.addEventListener("contextmenu", this.onContextMenu);
    this.canvas.addEventListener("dblclick", this.onDblClick);

    this.resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) requestAnimationFrame(() => this.resize(w));
      }
    });
    this.resizeObserver.observe(container);

    this.enforceEndpointConstraints();
    this.buildLUT();
    this.render();
  }

  private resize(containerWidth: number) {
    const w = Math.round(containerWidth);
    const h = Math.round(w / EnvelopeEditor.ASPECT);
    if (w === this.cssW && h === this.cssH) return;

    this.cssW = w;
    this.cssH = h;
    this.drawW = w - MARGIN * 2;
    this.drawH = h - MARGIN * 2;

    this.canvas.width = w * this.dpr;
    this.canvas.height = h * this.dpr;
    this.canvas.style.height = `${h}px`;
    this.ctx2d = this.canvas.getContext("2d")!;
    this.ctx2d.scale(this.dpr, this.dpr);
    this.render();
  }

  onChange(cb: (shape: EnvelopeShape, lut: Float32Array) => void) {
    this.onChangeCb = cb;
  }

  getLUT(): Float32Array { return this.lut; }

  /** Get the sustain level (decay section's last node y). */
  private getSustainLevel(): number {
    const dec = this.sections.decay;
    return dec.length >= 2 ? dec[dec.length - 1].y : 0.6;
  }

  getShape(): EnvelopeShape {
    return {
      attack: { proportion: this.proportions.attack, nodes: this.sections.attack.map(n => ({ ...n })) },
      decay: { proportion: this.proportions.decay, nodes: this.sections.decay.map(n => ({ ...n })) },
      sustainLevel: this.getSustainLevel(),
      release: { proportion: this.proportions.release, nodes: this.sections.release.map(n => ({ ...n })) },
    };
  }

  toJSON(): string {
    return JSON.stringify({
      version: 2,
      proportions: this.proportions,
      attack: this.sections.attack,
      decay: this.sections.decay,
      release: this.sections.release,
    });
  }

  fromJSON(json: string) {
    try {
      const parsed = JSON.parse(json);

      if (parsed.version === 2 && parsed.proportions && parsed.attack && parsed.decay && parsed.release) {
        // New 4-section format
        this.proportions = { ...parsed.proportions };
        this.sections.attack = parsed.attack;
        this.sections.decay = parsed.decay;
        this.sections.release = parsed.release;
      } else if (parsed.proportions && parsed.attack && parsed.sustain && parsed.release) {
        // Old 3-section format: migrate sustain → decay
        const oldSustainProp = parsed.proportions.sustain ?? 0.45;
        this.proportions = {
          attack: parsed.proportions.attack ?? 0.20,
          decay: oldSustainProp * 0.5,
          sustain: oldSustainProp * 0.5,
          release: parsed.proportions.release ?? 0.30,
        };
        // Normalize
        const total = this.proportions.attack + this.proportions.decay + this.proportions.sustain + this.proportions.release;
        this.proportions.attack /= total;
        this.proportions.decay /= total;
        this.proportions.sustain /= total;
        this.proportions.release /= total;

        this.sections.attack = parsed.attack;
        this.sections.decay = parsed.sustain; // old sustain → new decay
        this.sections.release = parsed.release;
      } else {
        return; // unrecognized format
      }

      this.enforceEndpointConstraints();
      this.buildLUT();
      this.render();
    } catch { /* ignore bad data */ }
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

  /* ── Coordinate transforms ──────────────────────────────────────── */

  /** Convert global curve coords (x in [0,1], y in [0,1]) to canvas pixel coords. */
  private curveToCanvas(x: number, y: number): [number, number] {
    return [MARGIN + x * this.drawW, MARGIN + (1 - y) * this.drawH];
  }

  private canvasToCurve(cx: number, cy: number): [number, number] {
    return [(cx - MARGIN) / this.drawW, 1 - (cy - MARGIN) / this.drawH];
  }

  private toInternal(e: MouseEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (this.cssW / rect.width),
      (e.clientY - rect.top) * (this.cssH / rect.height),
    ];
  }

  /** Convert section-local x [0,1] to global x [0,1]. */
  private sectionLocalToGlobal(section: SectionKey, localX: number): number {
    let offset = 0;
    for (const key of SECTION_KEYS) {
      if (key === section) return offset + localX * this.proportions[key];
      offset += this.proportions[key];
    }
    return offset;
  }

  /** Convert global x [0,1] to { section, localX }. */
  private globalToSection(globalX: number): { section: SectionKey; localX: number } {
    let offset = 0;
    for (const key of SECTION_KEYS) {
      const end = offset + this.proportions[key];
      if (globalX <= end || key === "release") {
        const localX = this.proportions[key] > 0
          ? clamp((globalX - offset) / this.proportions[key], 0, 1)
          : 0;
        return { section: key, localX };
      }
      offset = end;
    }
    return { section: "release", localX: 1 };
  }

  /** Get the global x position of a section divider. 0=A|D, 1=D|S, 2=S|R. */
  private dividerGlobalX(index: number): number {
    let x = 0;
    for (let i = 0; i <= index; i++) x += this.proportions[SECTION_KEYS[i]];
    return x;
  }

  /* ── Endpoint constraints ──────────────────────────────────────── */

  /**
   * Enforce ADSR constraints:
   * - Attack: first y=0, last y=free (peak level)
   * - Decay: first y = attack last y (peak), last y=free (sustain level)
   * - Release: first y = decay last y (sustain level), last y=0
   * All first nodes x=0, all last nodes x=1.
   */
  private enforceEndpointConstraints() {
    const atk = this.sections.attack;
    const dec = this.sections.decay;
    const rel = this.sections.release;

    // Lock x endpoints
    if (atk.length >= 2) { atk[0] = { ...atk[0], x: 0 }; atk[atk.length - 1] = { ...atk[atk.length - 1], x: 1 }; }
    if (dec.length >= 2) { dec[0] = { ...dec[0], x: 0 }; dec[dec.length - 1] = { ...dec[dec.length - 1], x: 1 }; }
    if (rel.length >= 2) { rel[0] = { ...rel[0], x: 0 }; rel[rel.length - 1] = { ...rel[rel.length - 1], x: 1 }; }

    // Attack first y = 0
    if (atk.length >= 1) atk[0] = { ...atk[0], y: 0 };

    // Decay first y = attack last y (peak)
    const peakY = atk.length >= 2 ? atk[atk.length - 1].y : 0.8;
    if (dec.length >= 1) dec[0] = { ...dec[0], y: peakY };

    // Release first y = decay last y (sustain level), release last y = 0
    const sustainLevel = dec.length >= 2 ? dec[dec.length - 1].y : peakY;
    if (rel.length >= 1) rel[0] = { ...rel[0], y: sustainLevel };
    if (rel.length >= 2) rel[rel.length - 1] = { ...rel[rel.length - 1], y: 0 };
  }

  /* ── Hit testing ─────────────────────────────────────────────────── */

  private hitTest(cx: number, cy: number): DragTarget | null {
    // Test dividers first (3 dividers: A|D, D|S, S|R)
    for (let d = 0; d < 3; d++) {
      const gx = this.dividerGlobalX(d);
      const [divCx] = this.curveToCanvas(gx, 0);
      if (Math.abs(cx - divCx) < DIVIDER_HIT_W) {
        return { type: "divider", dividerIndex: d };
      }
    }

    // Test handles, then nodes (only editable sections — not sustain)
    for (const key of EDITABLE_SECTION_KEYS) {
      const nodes = this.sections[key];
      // Handles
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const ngx = this.sectionLocalToGlobal(key, n.x);
        if (i < nodes.length - 1) {
          const hgx = ngx + n.handleOutDx * this.proportions[key];
          const [hcx, hcy] = this.curveToCanvas(hgx, n.y + n.handleOutDy);
          if (Math.hypot(cx - hcx, cy - hcy) < HIT_RADIUS) {
            return { type: "handleOut", section: key, index: i };
          }
        }
        if (i > 0) {
          const hgx = ngx + n.handleInDx * this.proportions[key];
          const [hcx, hcy] = this.curveToCanvas(hgx, n.y + n.handleInDy);
          if (Math.hypot(cx - hcx, cy - hcy) < HIT_RADIUS) {
            return { type: "handleIn", section: key, index: i };
          }
        }
      }
      // Nodes
      for (let i = 0; i < nodes.length; i++) {
        const [nx, ny] = this.curveToCanvas(
          this.sectionLocalToGlobal(key, nodes[i].x),
          nodes[i].y,
        );
        if (Math.hypot(cx - nx, cy - ny) < HIT_RADIUS) {
          return { type: "node", section: key, index: i };
        }
      }
    }

    return null;
  }

  /* ── Mouse handlers ────────────────────────────────────────────── */

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

    if (!this.dragging) {
      const prev = this.hovered;
      this.hovered = this.hitTest(cx, cy);
      if (this.hovered) {
        this.canvas.style.cursor =
          this.hovered.type === "divider" ? "move" :
          this.hovered.type === "node" ? "grab" : "pointer";
      } else {
        this.canvas.style.cursor = "crosshair";
      }
      if (this.hovered?.type !== prev?.type ||
        (this.hovered as NodeDrag)?.index !== (prev as NodeDrag)?.index ||
        (this.hovered as NodeDrag)?.section !== (prev as NodeDrag)?.section) {
        this.render();
      }
      return;
    }

    const [curveX, curveY] = this.canvasToCurve(cx, cy);

    if (this.dragging.type === "divider") {
      this.handleDividerDrag(this.dragging.dividerIndex, curveX, curveY);
    } else {
      this.handleNodeDrag(this.dragging, curveX, curveY);
    }

    this.canvas.style.cursor = this.dragging.type === "divider" ? "move" : "grabbing";
    this.enforceEndpointConstraints();
    this.buildLUT();
    this.render();
    this.onChangeCb?.(this.getShape(), this.lut);
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
    if (hit && hit.type === "node") {
      const nodes = this.sections[hit.section];
      if (hit.index > 0 && hit.index < nodes.length - 1) {
        nodes.splice(hit.index, 1);
        this.enforceEndpointConstraints();
        this.buildLUT();
        this.render();
        this.onChangeCb?.(this.getShape(), this.lut);
      }
    }
  };

  private onDblClick = (e: MouseEvent) => {
    const [cx, cy] = this.toInternal(e);

    // Double-click on existing interior node → delete it
    const hit = this.hitTest(cx, cy);
    if (hit && hit.type === "node") {
      const nodes = this.sections[hit.section];
      if (hit.index > 0 && hit.index < nodes.length - 1) {
        nodes.splice(hit.index, 1);
        this.enforceEndpointConstraints();
        this.buildLUT();
        this.render();
        this.onChangeCb?.(this.getShape(), this.lut);
      }
      return;
    }

    // Double-click empty space → add node in the appropriate section
    const [curveX, curveY] = this.canvasToCurve(cx, cy);
    const { section, localX } = this.globalToSection(curveX);

    // No node adding in sustain section
    if (section === "sustain") return;
    if (localX <= 0.02 || localX >= 0.98) return;

    const nodes = this.sections[section as EditableSectionKey];
    let insertIdx = 1;
    for (let i = 1; i < nodes.length; i++) {
      if (nodes[i].x > localX) { insertIdx = i; break; }
    }

    const prevX = nodes[insertIdx - 1].x;
    const nextX = nodes[insertIdx].x;
    const hLen = Math.min((localX - prevX) / 3, (nextX - localX) / 3, 0.15);

    nodes.splice(insertIdx, 0, makeNode(
      localX, clamp(curveY, 0, 1),
      -hLen, 0, hLen, 0,
    ));

    this.enforceEndpointConstraints();
    this.buildLUT();
    this.render();
    this.onChangeCb?.(this.getShape(), this.lut);
  };

  /* ── Drag logic ─────────────────────────────────────────────────── */

  private handleDividerDrag(dividerIndex: number, globalX: number, curveY: number) {
    const MIN_PROP = 0.05;
    const keys = SECTION_KEYS;

    // ── Horizontal: adjust section proportions ──────────────────────
    const leftKey = keys[dividerIndex];
    const rightKey = keys[dividerIndex + 1];

    let leftStart = 0;
    for (let i = 0; i < dividerIndex; i++) leftStart += this.proportions[keys[i]];

    let rightEnd = leftStart + this.proportions[leftKey] + this.proportions[rightKey];

    const newLeft = clamp(globalX - leftStart, MIN_PROP, rightEnd - leftStart - MIN_PROP);
    this.proportions[leftKey] = newLeft;
    this.proportions[rightKey] = rightEnd - leftStart - newLeft;

    const total = this.proportions.attack + this.proportions.decay + this.proportions.sustain + this.proportions.release;
    for (const k of keys) this.proportions[k] /= total;

    // ── Vertical: adjust boundary height ────────────────────────────
    const clampedY = clamp(curveY, 0, 1);

    if (dividerIndex === 0) {
      // A|D boundary → peak level
      const atk = this.sections.attack;
      if (atk.length >= 2) atk[atk.length - 1] = { ...atk[atk.length - 1], y: clampedY };
      // Decay first node gets synced by enforceEndpointConstraints
    } else if (dividerIndex === 1) {
      // D|S boundary → sustain level
      const dec = this.sections.decay;
      if (dec.length >= 2) dec[dec.length - 1] = { ...dec[dec.length - 1], y: clampedY };
      // Release first node gets synced by enforceEndpointConstraints
    }
    // Divider 2 (S|R): no boundary height to adjust (sustain level already set by divider 1)
  }

  private handleNodeDrag(drag: NodeDrag, globalCurveX: number, curveY: number) {
    const { section, index, type } = drag;
    const nodes = this.sections[section];
    const node = nodes[index];
    const prop = this.proportions[section];

    // Convert global x to section-local x
    let sectionStart = 0;
    for (const key of SECTION_KEYS) {
      if (key === section) break;
      sectionStart += this.proportions[key];
    }
    const localX = prop > 0 ? (globalCurveX - sectionStart) / prop : 0;

    if (type === "node") {
      const isFirst = index === 0;
      const isLast = index === nodes.length - 1;
      if (isFirst || isLast) {
        // Endpoints: x is locked. y is free for most endpoints.
        if (section === "attack" && isFirst) return; // fully locked (0,0)
        if (section === "release" && isLast) return;  // fully locked (1,0)
        // Attack last (peak), decay endpoints, release first: y is adjustable
        nodes[index] = { ...node, y: clamp(curveY, 0, 1) };
      } else {
        // Interior node: constrain x between neighbors
        const minX = nodes[index - 1].x + 0.005;
        const maxX = nodes[index + 1].x - 0.005;
        nodes[index] = { ...node, x: clamp(localX, minX, maxX), y: clamp(curveY, 0, 1) };
      }
    } else if (type === "handleIn") {
      const dx = localX - node.x;
      nodes[index] = { ...node, handleInDx: Math.min(0, dx), handleInDy: curveY - node.y };
    } else if (type === "handleOut") {
      const dx = localX - node.x;
      nodes[index] = { ...node, handleOutDx: Math.max(0, dx), handleOutDy: curveY - node.y };
    }
  }

  /* ── Bezier math ────────────────────────────────────────────────── */

  private findYForXInSection(section: EditableSectionKey, targetLocalX: number): number {
    const nodes = this.sections[section];
    if (nodes.length < 2) return 0;
    if (targetLocalX <= 0) return nodes[0].y;
    if (targetLocalX >= 1) return nodes[nodes.length - 1].y;

    for (let i = 0; i < nodes.length - 1; i++) {
      const n0 = nodes[i];
      const n1 = nodes[i + 1];
      if (targetLocalX < n0.x || targetLocalX > n1.x) continue;

      const p0x = n0.x, p0y = n0.y;
      const p1x = n0.x + n0.handleOutDx, p1y = n0.y + n0.handleOutDy;
      const p2x = n1.x + n1.handleInDx, p2y = n1.y + n1.handleInDy;
      const p3x = n1.x, p3y = n1.y;

      let lo = 0, hi = 1;
      for (let iter = 0; iter < 32; iter++) {
        const mid = (lo + hi) / 2;
        const [bx] = evalCubicBezier(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, mid);
        if (bx < targetLocalX) lo = mid; else hi = mid;
      }
      const [, by] = evalCubicBezier(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, (lo + hi) / 2);
      return by;
    }
    return targetLocalX;
  }

  /** Evaluate the full envelope at a global x position [0,1] → y [0,1]. */
  private evalGlobal(globalX: number): number {
    const { section, localX } = this.globalToSection(globalX);
    if (section === "sustain") return this.getSustainLevel();
    return this.findYForXInSection(section, localX);
  }

  private buildLUT() {
    for (let i = 0; i < LUT_SIZE; i++) {
      const x = i / (LUT_SIZE - 1);
      this.lut[i] = clamp(this.evalGlobal(x), 0, 1);
    }
  }

  /* ── Rendering ──────────────────────────────────────────────────── */

  private render() {
    const ctx = this.ctx2d;
    const w = this.cssW;
    const h = this.cssH;

    // Background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, w, h);

    // Section background tints
    let xOffset = 0;
    for (let s = 0; s < SECTION_KEYS.length; s++) {
      const key = SECTION_KEYS[s];
      const prop = this.proportions[key];
      const [x0] = this.curveToCanvas(xOffset, 1);
      const [x1] = this.curveToCanvas(xOffset + prop, 0);
      ctx.fillStyle = SECTION_COLORS[key];
      ctx.fillRect(x0, MARGIN, x1 - x0, this.drawH);
      xOffset += prop;
    }

    // Border
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // Grid lines (4x4)
    ctx.strokeStyle = "#262626";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gy = MARGIN + (i / 4) * this.drawH;
      ctx.beginPath();
      ctx.moveTo(MARGIN, gy);
      ctx.lineTo(MARGIN + this.drawW, gy);
      ctx.stroke();
    }

    // Section dividers (3 dividers: A|D, D|S, S|R)
    for (let d = 0; d < 3; d++) {
      const gx = this.dividerGlobalX(d);
      const [divCx] = this.curveToCanvas(gx, 0);
      const isHovered = this.hovered?.type === "divider" &&
        (this.hovered as DividerDrag).dividerIndex === d;
      ctx.strokeStyle = isHovered ? "#888" : "#444";
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(divCx, MARGIN);
      ctx.lineTo(divCx, MARGIN + this.drawH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Section labels
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    xOffset = 0;
    for (let s = 0; s < SECTION_KEYS.length; s++) {
      const key = SECTION_KEYS[s];
      const prop = this.proportions[key];
      const midX = xOffset + prop / 2;
      const [labelCx] = this.curveToCanvas(midX, 0);
      ctx.fillStyle = "#555";
      ctx.fillText(SECTION_LABELS[s], labelCx, MARGIN + this.drawH + 11);
      xOffset += prop;
    }

    // Curve per section (draw from LUT for smooth composite)
    xOffset = 0;
    for (let s = 0; s < SECTION_KEYS.length; s++) {
      const key = SECTION_KEYS[s];
      const prop = this.proportions[key];
      const startGX = xOffset;
      const endGX = xOffset + prop;

      ctx.strokeStyle = SECTION_CURVE_COLORS[key];
      ctx.lineWidth = 2;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < LUT_SIZE; i++) {
        const gx = i / (LUT_SIZE - 1);
        if (gx < startGX - 0.002 || gx > endGX + 0.002) continue;
        const [px, py] = this.curveToCanvas(gx, this.lut[i]);
        if (first) { ctx.moveTo(px, py); first = false; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      xOffset += prop;
    }

    // Handles and nodes (only for editable sections — not sustain)
    for (const key of EDITABLE_SECTION_KEYS) {
      const nodes = this.sections[key];
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const ngx = this.sectionLocalToGlobal(key, n.x);
        const [nx, ny] = this.curveToCanvas(ngx, n.y);

        // Handle-in
        if (i > 0) {
          const hgx = ngx + n.handleInDx * this.proportions[key];
          const [hx, hy] = this.curveToCanvas(hgx, n.y + n.handleInDy);
          ctx.strokeStyle = "#556";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(nx, ny);
          ctx.lineTo(hx, hy);
          ctx.stroke();
          const isH = this.hovered?.type === "handleIn" &&
            (this.hovered as NodeDrag).section === key &&
            (this.hovered as NodeDrag).index === i;
          ctx.fillStyle = isH ? "#aab" : "#778";
          ctx.beginPath();
          ctx.arc(hx, hy, HANDLE_RADIUS, 0, Math.PI * 2);
          ctx.fill();
        }

        // Handle-out
        if (i < nodes.length - 1) {
          const hgx = ngx + n.handleOutDx * this.proportions[key];
          const [hx, hy] = this.curveToCanvas(hgx, n.y + n.handleOutDy);
          ctx.strokeStyle = "#556";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(nx, ny);
          ctx.lineTo(hx, hy);
          ctx.stroke();
          const isH = this.hovered?.type === "handleOut" &&
            (this.hovered as NodeDrag).section === key &&
            (this.hovered as NodeDrag).index === i;
          ctx.fillStyle = isH ? "#aab" : "#778";
          ctx.beginPath();
          ctx.arc(hx, hy, HANDLE_RADIUS, 0, Math.PI * 2);
          ctx.fill();
        }

        // Node circle
        const isNodeH = this.hovered?.type === "node" &&
          (this.hovered as NodeDrag).section === key &&
          (this.hovered as NodeDrag).index === i;
        ctx.fillStyle = isNodeH ? "#fff" : "#ddd";
        ctx.beginPath();
        ctx.arc(nx, ny, NODE_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = SECTION_CURVE_COLORS[key];
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/*  LUT builders (standalone, no editor needed)                        */
/* ------------------------------------------------------------------ */

/**
 * Build a Float32Array LUT from an EnvelopeShape.
 * Uses the editor's proportions for time allocation (for display).
 * Values are raw amplitude [0, 1]. Caller scales by volume.
 */
export function envelopeShapeToLUT(shape: EnvelopeShape, size: number = LUT_SIZE): Float32Array {
  const lut = new Float32Array(size);
  const sections: { key: string; section: EnvelopeSection | null; level?: number }[] = [
    { key: "attack", section: shape.attack },
    { key: "decay", section: shape.decay },
    { key: "sustain", section: null, level: shape.sustainLevel },
    { key: "release", section: shape.release },
  ];
  const totalProp = shape.attack.proportion + (shape.decay.proportion) +
    // Sustain proportion: infer from remainder (since it's not stored in EnvelopeSection)
    // For display LUT, use a default sustain proportion
    0.3 + shape.release.proportion;
  // Actually, let's compute proportion from shape sections that exist
  // Attack + Decay + Release have proportions; sustain fills the rest to make display nice
  const adrProp = shape.attack.proportion + shape.decay.proportion + shape.release.proportion;
  // Give sustain 30% of adr total for display purposes
  const sustainDisplayProp = adrProp * 0.4;
  const totalWithSustain = adrProp + sustainDisplayProp;

  const props = [
    shape.attack.proportion / totalWithSustain,
    shape.decay.proportion / totalWithSustain,
    sustainDisplayProp / totalWithSustain,
    shape.release.proportion / totalWithSustain,
  ];

  for (let i = 0; i < size; i++) {
    const globalX = i / (size - 1);

    let offset = 0;
    let value = 0;
    for (let s = 0; s < 4; s++) {
      const p = props[s];
      if (globalX <= offset + p || s === 3) {
        const localX = p > 0 ? clamp((globalX - offset) / p, 0, 1) : 0;
        const sec = sections[s];
        if (sec.section && sec.section.nodes.length >= 2) {
          value = findYInNodes(sec.section.nodes, localX);
        } else {
          value = sec.level ?? 0;
        }
        break;
      }
      offset += p;
    }

    lut[i] = clamp(value, 0, 1);
  }
  return lut;
}

/**
 * Build a gate-aware LUT from an EnvelopeShape with explicit A/D/S/R durations.
 * The sustain section is a flat line at shape.sustainLevel for sustainDur.
 * A/D/R sections use the editor's bezier curves.
 * Values are raw amplitude [0, 1]. Caller scales by volume.
 */
export function buildGateAwareLUT(
  shape: EnvelopeShape,
  attackDur: number,
  decayDur: number,
  sustainDur: number,
  releaseDur: number,
  size: number = LUT_SIZE,
): Float32Array {
  const lut = new Float32Array(size);
  const totalDur = attackDur + decayDur + sustainDur + releaseDur;
  if (totalDur <= 0) return lut;

  const aFrac = attackDur / totalDur;
  const dFrac = decayDur / totalDur;
  const sFrac = sustainDur / totalDur;
  // rFrac = remainder

  for (let i = 0; i < size; i++) {
    const x = i / (size - 1);

    if (x <= aFrac) {
      // Attack section
      const localX = aFrac > 0 ? x / aFrac : 1;
      lut[i] = shape.attack.nodes.length >= 2
        ? clamp(findYInNodes(shape.attack.nodes, localX), 0, 1)
        : 0;
    } else if (x <= aFrac + dFrac) {
      // Decay section
      const localX = dFrac > 0 ? (x - aFrac) / dFrac : 1;
      lut[i] = shape.decay.nodes.length >= 2
        ? clamp(findYInNodes(shape.decay.nodes, localX), 0, 1)
        : shape.sustainLevel;
    } else if (x <= aFrac + dFrac + sFrac) {
      // Sustain section: flat line
      lut[i] = clamp(shape.sustainLevel, 0, 1);
    } else {
      // Release section
      const rStart = aFrac + dFrac + sFrac;
      const rFrac = 1 - rStart;
      const localX = rFrac > 0 ? (x - rStart) / rFrac : 1;
      lut[i] = shape.release.nodes.length >= 2
        ? clamp(findYInNodes(shape.release.nodes, localX), 0, 1)
        : 0;
    }
  }
  return lut;
}

function findYInNodes(nodes: readonly EnvelopeNode[], targetX: number): number {
  if (targetX <= 0) return nodes[0].y;
  if (targetX >= 1) return nodes[nodes.length - 1].y;

  for (let i = 0; i < nodes.length - 1; i++) {
    const n0 = nodes[i];
    const n1 = nodes[i + 1];
    if (targetX < n0.x || targetX > n1.x) continue;

    const p0x = n0.x, p0y = n0.y;
    const p1x = n0.x + n0.handleOutDx, p1y = n0.y + n0.handleOutDy;
    const p2x = n1.x + n1.handleInDx, p2y = n1.y + n1.handleInDy;
    const p3x = n1.x, p3y = n1.y;

    let lo = 0, hi = 1;
    for (let iter = 0; iter < 32; iter++) {
      const mid = (lo + hi) / 2;
      const [bx] = evalCubicBezier(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, mid);
      if (bx < targetX) lo = mid; else hi = mid;
    }
    const [, by] = evalCubicBezier(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, (lo + hi) / 2);
    return by;
  }
  return targetX;
}
