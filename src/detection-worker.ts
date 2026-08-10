/**
 * Web Worker that runs the detection pipeline + global metrics off the main thread.
 *
 * Receives ReadbackData + config, returns a serialized DetectionFrame and GlobalMetrics.
 */

import { runDetection, type ReadbackData } from "./detection";
import { computeGlobalMetrics } from "./music/global-metrics";
import { serializeDetectionFrame, deserializeDetectionFrame } from "./detection-serialization";
import type {
  DetectionWorkerRequest,
  DetectionWorkerResponse,
  GlobalMetricsWire,
} from "./detection-worker-types";

self.onmessage = (e: MessageEvent<DetectionWorkerRequest>) => {
  const req = e.data;

  const readbackData: ReadbackData = {
    n: req.n,
    posX: req.posX,
    posY: req.posY,
    velX: req.velX,
    velY: req.velY,
    particleTypes: req.particleTypes,
    particleCells: req.particleCells,
    cellHeads: req.cellHeads,
    cellNext: req.cellNext,
    cols: req.cols,
    rows: req.rows,
    cellSize: req.cellSize,
    width: req.width,
    height: req.height,
  };

  const prevFrame = req.prevFrame ? deserializeDetectionFrame(req.prevFrame) : null;

  const frame = runDetection(
    readbackData,
    prevFrame,
    req.config,
    req.dt,
    req.bpm,
    req.forceMatrix,
    req.typeKeys,
  );

  // Compute global metrics while we still have readbackData
  const typeCounts = new Map(req.typeCounts);
  const metrics = computeGlobalMetrics(
    readbackData,
    frame,
    typeCounts,
    prevFrame,
    req.organismPrediction,
    req.typeKeys,
  );

  const metricsWire: GlobalMetricsWire = {
    freeParticleCount: metrics.freeParticleCount,
    freeParticlePercentByType: Array.from(metrics.freeParticlePercentByType),
    avgVelocity: metrics.avgVelocity,
    avgOrganelleDensity: metrics.avgOrganelleDensity,
    speciesCount: metrics.speciesCount,
    organismCount: metrics.organismCount,
    spatialEntropy: metrics.spatialEntropy,
    events: metrics.events,
    organismFulfillment: metrics.organismFulfillment,
  };

  const wire = serializeDetectionFrame(frame);

  // Collect transferable typed arrays from the result
  const transferables: Transferable[] = [];
  for (const org of wire.organelles) {
    if (org.particleIndices instanceof Uint32Array) {
      transferables.push(org.particleIndices.buffer);
    }
  }

  const resp: DetectionWorkerResponse = { id: req.id, frame: wire, metrics: metricsWire };
  (self as unknown as Worker).postMessage(resp, transferables);
};
