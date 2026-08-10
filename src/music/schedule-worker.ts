/**
 * Web Worker for bar scheduling.
 *
 * Streams results: bar-meta first, then per-organism slot-fills, then done.
 * This allows the main thread to start playing notes as soon as they arrive
 * rather than waiting for the entire bar computation to finish.
 */

import { computeBarContext, computeSpeciesSlots, type ScheduleConfig } from "./hit-scheduler";
import type { ScheduleWorkerSlotFill } from "./types";
import {
  deserializeBarSnapshot,
  deserializeMusicState,
  serializeBarMeta,
  type BarSnapshotWire,
  type MusicStateWire,
  type ScheduleWorkerMsgWire,
} from "./worker-serialization";

/* ── Message types ───────────────────────────────────────────────────── */

export interface ScheduleWorkerRequest {
  readonly id: number;
  readonly snapshot: BarSnapshotWire;
  readonly barNumber: number;
  readonly barStartTime: number;
  readonly barDur: number;
  readonly prevState: MusicStateWire | null;
  readonly config: ScheduleConfig;
}

/* ── Handler ─────────────────────────────────────────────────────────── */

function post(msg: ScheduleWorkerMsgWire): void {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = (e: MessageEvent<ScheduleWorkerRequest>) => {
  const req = e.data;

  // The whole computation is guarded so `done` ALWAYS reaches the main
  // thread — pendingBarRequest is only cleared by `done`, so a bare throw
  // here would permanently stall all future bar scheduling.
  try {
    const snapshot = deserializeBarSnapshot(req.snapshot);
    const prevState = deserializeMusicState(req.prevState);

    // 1. Compute bar context (fast — mode, root, bass, ranges)
    const ctx = computeBarContext(
      snapshot,
      req.barStartTime,
      req.barDur,
      prevState,
      req.config,
    );

    // 2. Stream bar-meta immediately
    post(serializeBarMeta(ctx, req.id, req.barNumber));

    // 3. Stream per-species slot fills (one voice per type per species)
    const speciesResults = computeSpeciesSlots(
      snapshot, ctx, req.barStartTime, req.barDur, req.config, req.barNumber,
    );
    for (const result of speciesResults) {
      for (const { slotIndex, note } of result.slots) {
        const fill: ScheduleWorkerSlotFill = {
          kind: "slot-fill",
          id: req.id,
          tierIndex: result.tierIndex,
          slotIndex,
          notes: [note],
        };
        post(fill);
      }
    }
  } catch (err) {
    console.error("schedule-worker: bar computation failed", err);
  } finally {
    // 4. Signal completion
    post({ kind: "done" as const, id: req.id });
  }
};
