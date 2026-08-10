/**
 * Serialize / deserialize DetectionFrame ↔ DetectionFrameWire.
 *
 * OrganelleState and OrganelleTreeNode are already plain-ish objects,
 * but organisms contain Maps which need conversion to arrays.
 */

import type {
  DetectionFrame,
  LedgerState,
  OrganelleState,
  OrganismState,
} from "./detection";

import type {
  DetectionFrameWire,
  LedgerStateWire,
  OrganelleStateWire,
  OrganismStateWire,
} from "./detection-worker-types";

/* ── Serialize (main → worker, or worker → main) ─────────────────── */

function serializeOrganelle(o: OrganelleState): OrganelleStateWire {
  return {
    id: o.id,
    typeId: o.typeId,
    particleIndices: o.particleIndices,
    centroidX: o.centroidX,
    centroidY: o.centroidY,
    avgVelX: o.avgVelX,
    avgVelY: o.avgVelY,
    minCol: o.minCol,
    maxCol: o.maxCol,
    minRow: o.minRow,
    maxRow: o.maxRow,
  };
}

function serializeOrganism(o: OrganismState): OrganismStateWire {
  return {
    id: o.id,
    organelleIds: Array.from(o.organelleIds),
    colorSignature: o.colorSignature,
    centroidX: o.centroidX,
    centroidY: o.centroidY,
    tree: o.tree, // already plain objects
    crossTypeLinks: Array.from(o.crossTypeLinks),
  };
}

function serializeLedger(l: LedgerState): LedgerStateWire {
  return {
    organellesByType: Array.from(l.organellesByType),
    organismsBySignature: Array.from(l.organismsBySignature),
    organelleStability: Array.from(l.organelleStability),
    organismStability: Array.from(l.organismStability),
  };
}

export function serializeDetectionFrame(f: DetectionFrame): DetectionFrameWire {
  return {
    organelles: f.organelles.map(serializeOrganelle),
    organisms: f.organisms.map(serializeOrganism),
    holdTimers: Array.from(f.holdTimers),
    ledger: serializeLedger(f.ledger),
  };
}

/* ── Deserialize (worker → main, or main → worker) ───────────────── */

function deserializeOrganelle(w: OrganelleStateWire): OrganelleState {
  return {
    id: w.id,
    typeId: w.typeId,
    particleIndices: w.particleIndices instanceof Uint32Array
      ? w.particleIndices
      : new Uint32Array(w.particleIndices),
    centroidX: w.centroidX,
    centroidY: w.centroidY,
    avgVelX: w.avgVelX,
    avgVelY: w.avgVelY,
    minCol: w.minCol,
    maxCol: w.maxCol,
    minRow: w.minRow,
    maxRow: w.maxRow,
  };
}

function deserializeOrganism(w: OrganismStateWire): OrganismState {
  return {
    id: w.id,
    organelleIds: w.organelleIds,
    colorSignature: w.colorSignature,
    centroidX: w.centroidX,
    centroidY: w.centroidY,
    tree: w.tree,
    crossTypeLinks: new Map(w.crossTypeLinks),
  };
}

function deserializeLedger(w: LedgerStateWire): LedgerState {
  return {
    organellesByType: new Map(w.organellesByType),
    organismsBySignature: new Map(w.organismsBySignature),
    organelleStability: new Map(w.organelleStability),
    organismStability: new Map(w.organismStability),
  };
}

export function deserializeDetectionFrame(w: DetectionFrameWire): DetectionFrame {
  return {
    organelles: w.organelles.map(deserializeOrganelle),
    organisms: w.organisms.map(deserializeOrganism),
    holdTimers: new Map(w.holdTimers),
    ledger: deserializeLedger(w.ledger),
  };
}
