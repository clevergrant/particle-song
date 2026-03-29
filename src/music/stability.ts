/**
 * Net stability computation from global metrics (§4.2).
 *
 * Uses organism fulfillment — the fraction of organism-capable particles
 * that are actually in organisms — as the primary stability signal.
 */

import type { GlobalMetrics } from "./types";

/* ------------------------------------------------------------------ */
/*  Net stability                                                      */
/* ------------------------------------------------------------------ */

/**
 * Compute net stability from global metrics.
 * Returns a value in [0, 1] where 1 = maximally stable.
 *
 * Uses organismFulfillment directly, compressed into [0.10, 0.85]
 * so peak stability lands in Ionian (major) and the floor stays
 * above Chromatic.
 */
export function computeNetStability(metrics: GlobalMetrics): number {
  return metrics.organismFulfillment;
}
