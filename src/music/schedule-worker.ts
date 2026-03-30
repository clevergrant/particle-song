/**
 * Web Worker for scheduleBar + applyVoiceBudget.
 *
 * Runs the heavy music scheduling computation off the main thread
 * so the animation loop stays smooth at bar boundaries.
 */

import { scheduleBar, type ScheduleConfig } from "./hit-scheduler";
import { applyVoiceBudget } from "./voice-budget";
import {
  deserializeBarSnapshot,
  deserializeMusicState,
  serializeScheduledBar,
  type BarSnapshotWire,
  type MusicStateWire,
  type ScheduledBarWire,
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
  readonly voiceBudget: number;
}

export interface ScheduleWorkerResponse {
  readonly id: number;
  /** Raw output — used for musicState/bass updates. */
  readonly scheduled: ScheduledBarWire;
  /** After voice-budget culling — used for playback. */
  readonly culled: ScheduledBarWire;
}

/* ── Handler ─────────────────────────────────────────────────────────── */

self.onmessage = (e: MessageEvent<ScheduleWorkerRequest>) => {
  const req = e.data;

  const snapshot = deserializeBarSnapshot(req.snapshot);
  const prevState = deserializeMusicState(req.prevState);

  const scheduled = scheduleBar(
    snapshot,
    req.barNumber,
    req.barStartTime,
    req.barDur,
    prevState,
    req.config,
  );

  const culled = applyVoiceBudget(scheduled, req.voiceBudget);

  const response: ScheduleWorkerResponse = {
    id: req.id,
    scheduled: serializeScheduledBar(scheduled),
    culled: serializeScheduledBar(culled),
  };

  (self as unknown as Worker).postMessage(response);
};
