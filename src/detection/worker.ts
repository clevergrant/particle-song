/**
 * Web Worker that runs the detection pipeline off the main thread.
 *
 * Receives ReadbackData + config, returns a serialized DetectionFrame.
 */

import { runDetection, type ReadbackData } from "./index";
import { serializeDetectionFrame, deserializeDetectionFrame } from "./serialization";
import type {
  DetectionWorkerRequest,
  DetectionWorkerResponse,
} from "./worker-types";

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
    req.forceMatrix,
    req.typeKeys,
  );

  const wire = serializeDetectionFrame(frame);

  // Collect transferable typed arrays from the result
  const transferables: Transferable[] = [];
  for (const org of wire.organelles) {
    if (org.particleIndices instanceof Uint32Array) {
      transferables.push(org.particleIndices.buffer);
    }
  }

  const resp: DetectionWorkerResponse = { id: req.id, frame: wire };
  (self as unknown as Worker).postMessage(resp, transferables);
};
