/**
 * Wire types for detection worker communication.
 *
 * Maps / trees are serialized to plain arrays/objects so they can be
 * transferred via structured clone without overhead.
 */

import type { DetectionConfig } from "./index";
import type { ForceMatrix } from "../particles/particle";

/* ── Serialized (wire) representations ─────────────────────────────── */

export interface OrganelleStateWire {
  readonly id: number;
  readonly typeId: number;
  readonly particleIndices: Uint32Array;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly avgVelX: number;
  readonly avgVelY: number;
  readonly minCol: number;
  readonly maxCol: number;
  readonly minRow: number;
  readonly maxRow: number;
}

export interface OrganelleTreeNodeWire {
  readonly organelleId: number;
  readonly typeId: number;
  readonly children: ReadonlyArray<OrganelleTreeNodeWire>;
}

export interface OrganismStateWire {
  readonly id: number;
  readonly organelleIds: ReadonlyArray<number>;
  readonly colorSignature: string;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly tree: OrganelleTreeNodeWire;
  readonly crossTypeLinks: ReadonlyArray<[number, number]>;
}

export interface LedgerStateWire {
  readonly organellesByType: ReadonlyArray<[number, number]>;
  readonly organismsBySignature: ReadonlyArray<[string, number]>;
  readonly organelleStability: ReadonlyArray<[number, number]>;
  readonly organismStability: ReadonlyArray<[string, number]>;
}

export interface DetectionFrameWire {
  readonly organelles: ReadonlyArray<OrganelleStateWire>;
  readonly organisms: ReadonlyArray<OrganismStateWire>;
  readonly holdTimers: ReadonlyArray<[number, number]>;
  readonly ledger: LedgerStateWire;
}

/* ── Worker message types ──────────────────────────────────────────── */

export interface DetectionWorkerRequest {
  readonly id: number;
  /** ReadbackData fields — typed arrays are transferred. */
  readonly n: number;
  readonly posX: Float32Array;
  readonly posY: Float32Array;
  readonly velX: Float32Array;
  readonly velY: Float32Array;
  readonly particleTypes: Uint32Array;
  readonly particleCells: Uint32Array;
  readonly cellHeads: Int32Array;
  readonly cellNext: Int32Array;
  readonly cols: number;
  readonly rows: number;
  readonly cellSize: number;
  readonly width: number;
  readonly height: number;

  readonly config: DetectionConfig;
  readonly dt: number;
  readonly forceMatrix: ForceMatrix;
  readonly typeKeys: ReadonlyArray<string>;
  readonly prevFrame: DetectionFrameWire | null;
}

export interface DetectionWorkerResponse {
  readonly id: number;
  readonly frame: DetectionFrameWire;
}
