/**
 * Pure detection pipeline: particles → organelles → organisms → ledger.
 *
 * All functions are stateless and operate on typed arrays from the readback
 * callback. The only mutable state is the latch timers carried forward between
 * detection ticks via DetectionFrame.
 */

import type { ForceMatrix } from "./particles/particle";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface RegisteredOrganism {
  readonly registryId: number;
  readonly colorSignature: string;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly velX: number;
  readonly velY: number;
  readonly organelleIds: ReadonlyArray<number>;
  readonly tree: OrganelleTreeNode;
  readonly creationTime: number;
  /** Number of direct cross-type bonds per organelle (organelle id → count). */
  readonly crossTypeLinks: ReadonlyMap<number, number>;
}

export interface OrganismRegistry {
  readonly organisms: ReadonlyArray<RegisteredOrganism>;
  readonly nextId: number;
}

export interface DetectionConfig {
  readonly proximityRadius: number;
  readonly coherenceThreshold: number;
  readonly minOrganelleSize: number;
  readonly organelleLatchBeats: number;
  readonly organismProximityRadius: number;
  readonly organismCoherenceThreshold: number;
}

export const DEFAULT_DETECTION_CONFIG: DetectionConfig = {
  proximityRadius: 18,
  coherenceThreshold: 45,
  minOrganelleSize: 3,
  organelleLatchBeats: 1,
  organismProximityRadius: 60,
  organismCoherenceThreshold: 80,
};

export interface ReadbackData {
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
}

export interface OrganelleState {
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

export interface OrganelleTreeNode {
  readonly organelleId: number;
  readonly typeId: number;
  readonly children: ReadonlyArray<OrganelleTreeNode>;
}

export interface OrganismState {
  readonly id: number;
  readonly organelleIds: ReadonlyArray<number>;
  readonly colorSignature: string;
  readonly centroidX: number;
  readonly centroidY: number;
  readonly tree: OrganelleTreeNode;
  /** Number of direct cross-type bonds per organelle (organelle id → count). */
  readonly crossTypeLinks: ReadonlyMap<number, number>;
}

export interface LedgerState {
  readonly organellesByType: ReadonlyMap<number, number>;
  readonly organismsBySignature: ReadonlyMap<string, number>;
  readonly organelleStability: ReadonlyMap<number, number>;   // typeId → EMA score [0,1]
  readonly organismStability: ReadonlyMap<string, number>;    // signature → EMA score [0,1]
}

export interface DetectionFrame {
  readonly organelles: ReadonlyArray<OrganelleState>;
  readonly organisms: ReadonlyArray<OrganismState>;
  readonly holdTimers: ReadonlyMap<number, number>;  // particleIdx → seconds remaining
  readonly ledger: LedgerState;
}

/* ------------------------------------------------------------------ */
/*  Union-Find (path-splitting, rank-union)                            */
/* ------------------------------------------------------------------ */

function ufCreate(n: number): { parent: Uint32Array; rank: Uint8Array } {
  const parent = new Uint32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  return { parent, rank: new Uint8Array(n) };
}

function ufFind(parent: Uint32Array, i: number): number {
  while (parent[i] !== i) {
    const next = parent[parent[i]];
    parent[i] = next;
    i = next;
  }
  return i;
}

function ufUnion(parent: Uint32Array, rank: Uint8Array, a: number, b: number): void {
  const ra = ufFind(parent, a);
  const rb = ufFind(parent, b);
  if (ra === rb) return;
  if (rank[ra] < rank[rb]) { parent[ra] = rb; }
  else if (rank[ra] > rank[rb]) { parent[rb] = ra; }
  else { parent[rb] = ra; rank[ra]++; }
}

/* ------------------------------------------------------------------ */
/*  Level 1: Particles → Organelles                                    */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Organelle Tree (MST-based)                                         */
/* ------------------------------------------------------------------ */

/**
 * Build an organelle tree via Prim's MST from centroid distances.
 * The first organelle in the list becomes the root.
 * O(n^2) — fine for typically <10 organelles per organism.
 */
function buildOrganelleTree(
  organelleIds: ReadonlyArray<number>,
  organelleMap: ReadonlyMap<number, OrganelleState>,
): OrganelleTreeNode {
  const n = organelleIds.length;
  if (n === 0) throw new Error("Cannot build tree from empty organelle list");

  if (n === 1) {
    const org = organelleMap.get(organelleIds[0])!;
    return { organelleId: org.id, typeId: org.typeId, children: [] };
  }

  // Prim's MST: parent[i] = index into organelleIds of this node's parent
  const inMST = new Uint8Array(n);
  const parent = new Int32Array(n).fill(-1);
  const minEdge = new Float64Array(n).fill(Infinity);
  minEdge[0] = 0;

  for (let step = 0; step < n; step++) {
    // Find the unvisited node with smallest edge weight
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (!inMST[i] && minEdge[i] < best) {
        best = minEdge[i];
        u = i;
      }
    }
    if (u === -1) break;
    inMST[u] = 1;

    const ou = organelleMap.get(organelleIds[u])!;
    for (let v = 0; v < n; v++) {
      if (inMST[v]) continue;
      const ov = organelleMap.get(organelleIds[v])!;
      const dx = ou.centroidX - ov.centroidX;
      const dy = ou.centroidY - ov.centroidY;
      const distSq = dx * dx + dy * dy;
      if (distSq < minEdge[v]) {
        minEdge[v] = distSq;
        parent[v] = u;
      }
    }
  }

  // Build adjacency list from parent array
  const children: number[][] = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) {
    const p = parent[i];
    if (p >= 0) children[p].push(i);
  }

  // Recursively construct immutable tree
  function buildNode(idx: number): OrganelleTreeNode {
    const org = organelleMap.get(organelleIds[idx])!;
    return {
      organelleId: org.id,
      typeId: org.typeId,
      children: children[idx].map(buildNode),
    };
  }

  return buildNode(0);
}

/**
 * Attempt to reuse a previous tree, updating only what changed.
 * Returns null if the tree must be rebuilt from scratch.
 */
function updateOrganelleTree(
  prevTree: OrganelleTreeNode,
  currentIds: ReadonlySet<number>,
  newIds: ReadonlyArray<number>,
  organelleMap: ReadonlyMap<number, OrganelleState>,
): OrganelleTreeNode | null {
  // Check if the tree is unchanged (all previous IDs still present, no new ones)
  const prevIds = collectTreeIds(prevTree);
  const allPresent = prevIds.every(id => currentIds.has(id));
  const noNewIds = newIds.length === 0;
  if (allPresent && noNewIds) return prevTree;

  // Prune missing nodes
  let tree: OrganelleTreeNode | null = pruneTree(prevTree, currentIds);
  if (!tree) return null; // root was removed, rebuild

  // Attach new organelles to nearest existing node
  for (const newId of newIds) {
    const newOrg = organelleMap.get(newId);
    if (!newOrg) continue;
    tree = attachToNearest(tree, newOrg, organelleMap);
  }

  return tree;
}

function collectTreeIds(node: OrganelleTreeNode): number[] {
  const ids = [node.organelleId];
  for (const child of node.children) {
    ids.push(...collectTreeIds(child));
  }
  return ids;
}


/** Prune nodes not in currentIds; reparent their children to the parent. */
function pruneTree(
  node: OrganelleTreeNode,
  currentIds: ReadonlySet<number>,
): OrganelleTreeNode | null {
  // Recursively prune children first
  const keptChildren: OrganelleTreeNode[] = [];
  for (const child of node.children) {
    const pruned = pruneTree(child, currentIds);
    if (pruned) {
      keptChildren.push(pruned);
    } else {
      // Child was removed — adopt its children (reparent)
      for (const grandchild of child.children) {
        const prunedGrandchild = pruneTree(grandchild, currentIds);
        if (prunedGrandchild) keptChildren.push(prunedGrandchild);
      }
    }
  }

  if (!currentIds.has(node.organelleId)) {
    // This node is removed; its kept children become orphans for the caller to adopt
    // Return null but the caller handles reparenting via the pruneTree logic above
    return null;
  }

  return { ...node, children: keptChildren };
}

/** Attach a new organelle to the nearest existing node in the tree. */
function attachToNearest(
  tree: OrganelleTreeNode,
  newOrg: OrganelleState,
  organelleMap: ReadonlyMap<number, OrganelleState>,
): OrganelleTreeNode {
  // Find the closest existing node by centroid distance
  let bestId = tree.organelleId;
  let bestDistSq = Infinity;

  function search(node: OrganelleTreeNode) {
    const org = organelleMap.get(node.organelleId);
    if (org) {
      const dx = org.centroidX - newOrg.centroidX;
      const dy = org.centroidY - newOrg.centroidY;
      const distSq = dx * dx + dy * dy;
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        bestId = node.organelleId;
      }
    }
    for (const child of node.children) search(child);
  }
  search(tree);

  const newNode: OrganelleTreeNode = {
    organelleId: newOrg.id,
    typeId: newOrg.typeId,
    children: [],
  };

  // Insert the new node as a child of the closest node
  function insertAt(node: OrganelleTreeNode): OrganelleTreeNode {
    if (node.organelleId === bestId) {
      return { ...node, children: [...node.children, newNode] };
    }
    return { ...node, children: node.children.map(insertAt) };
  }

  return insertAt(tree);
}

/**
 * Depth-first traversal of the organelle tree with type-diversity heuristic.
 * When choosing which child to visit next, prefer children with a different
 * typeId than the previously visited node.
 */
export function traverseOrganelleTree(
  root: OrganelleTreeNode,
): ReadonlyArray<{ readonly organelleId: number; readonly typeId: number }> {
  const result: { organelleId: number; typeId: number }[] = [];

  function visit(node: OrganelleTreeNode, prevTypeId: number) {
    result.push({ organelleId: node.organelleId, typeId: node.typeId });

    if (node.children.length === 0) return;

    // Sort children: different typeId from current node first, then same
    const sorted = [...node.children].sort((a, b) => {
      const aMatch = a.typeId === node.typeId ? 1 : 0;
      const bMatch = b.typeId === node.typeId ? 1 : 0;
      return aMatch - bMatch;
    });

    for (const child of sorted) {
      visit(child, node.typeId);
    }
  }

  visit(root, -1);
  return result;
}

/* ------------------------------------------------------------------ */
/*  Level 2: Organelles → Organisms                                    */
/* ------------------------------------------------------------------ */

function detectOrganisms(
  organelles: ReadonlyArray<OrganelleState>,
  prevFrame: DetectionFrame | null,
  config: DetectionConfig,
  forceMatrix: ForceMatrix,
  typeKeys: ReadonlyArray<string>,
): OrganismState[] {
  const orgCount = organelles.length;
  if (orgCount === 0) return [];

  const proxRadSq = config.organismProximityRadius * config.organismProximityRadius;
  const cohThreshSq = config.organismCoherenceThreshold * config.organismCoherenceThreshold;

  // Union-find on organelle centroids, tracking cross-type adjacency
  const { parent, rank } = ufCreate(orgCount);
  // Per-organelle set of directly bonded cross-type typeIds (by array index)
  const crossTypeNeighborTypes: Set<number>[] = Array.from({ length: orgCount }, () => new Set());

  for (let i = 0; i < orgCount; i++) {
    const a = organelles[i];
    for (let j = i + 1; j < orgCount; j++) {
      const b = organelles[j];
      // Same-type organelles don't connect — organisms require cross-type bonds
      if (a.typeId === b.typeId) continue;
      const dx = a.centroidX - b.centroidX;
      const dy = a.centroidY - b.centroidY;
      const distSq = dx * dx + dy * dy;
      if (distSq >= proxRadSq) continue;

      const dvx = a.avgVelX - b.avgVelX;
      const dvy = a.avgVelY - b.avgVelY;
      const velDiffSq = dvx * dvx + dvy * dvy;
      if (velDiffSq >= cohThreshSq) continue;

      ufUnion(parent, rank, i, j);
      crossTypeNeighborTypes[i].add(b.typeId);
      crossTypeNeighborTypes[j].add(a.typeId);
    }
  }

  // Extract connected components
  const compMap = new Map<number, number[]>();
  for (let i = 0; i < orgCount; i++) {
    const root = ufFind(parent, i);
    let comp = compMap.get(root);
    if (!comp) { comp = []; compMap.set(root, comp); }
    comp.push(i);
  }

  // Build organelle lookup map for tree construction
  const organelleMap = new Map<number, OrganelleState>();
  for (const org of organelles) organelleMap.set(org.id, org);

  // Index previous trees by color signature for reuse
  const prevTreeBySig = new Map<string, OrganelleTreeNode>();
  if (prevFrame) {
    for (const osm of prevFrame.organisms) {
      prevTreeBySig.set(osm.colorSignature, osm.tree);
    }
  }

  // Build organisms from components with 2+ organelles,
  // but only if the component's types have net positive affinity.
  const organisms: OrganismState[] = [];
  let nextId = 0;
  for (const comp of compMap.values()) {
    if (comp.length < 2) continue;

    // Compute net affinity across all cross-type organelle pairs in this
    // component.  For each pair (a, b) of different types, look up both
    // force directions and weight by the product of particle counts.
    // Same-type pairs are inherently cohesive — only gate on cross-type.
    let affinity = 0;
    let hasCrossType = false;
    for (let ci = 0; ci < comp.length; ci++) {
      const oa = organelles[comp[ci]];
      const aKey = typeKeys[oa.typeId] ?? "";
      const aCount = oa.particleIndices.length;
      const aRow = forceMatrix[aKey];
      for (let cj = ci + 1; cj < comp.length; cj++) {
        const ob = organelles[comp[cj]];
        if (ob.typeId === oa.typeId) continue;
        hasCrossType = true;
        const bKey = typeKeys[ob.typeId] ?? "";
        const bCount = ob.particleIndices.length;
        const bRow = forceMatrix[bKey];
        affinity += ((aRow?.[bKey] ?? 0) + (bRow?.[aKey] ?? 0)) * aCount * bCount;
      }
    }
    if (hasCrossType && affinity <= 0) continue;

    const typeSet = new Set<number>();
    let cx = 0, cy = 0, total = 0;
    const organelleIds: number[] = [];
    const crossTypeLinks = new Map<number, number>();

    for (const oi of comp) {
      const org = organelles[oi];
      typeSet.add(org.typeId);
      const w = org.particleIndices.length;
      cx += org.centroidX * w;
      cy += org.centroidY * w;
      total += w;
      organelleIds.push(org.id);
      crossTypeLinks.set(org.id, crossTypeNeighborTypes[oi].size);
    }

    const sig = Array.from(typeSet).sort((a, b) => a - b).join("+");

    // Build or reuse organelle tree
    const currentIdSet = new Set(organelleIds);
    const prevTree = prevTreeBySig.get(sig);
    let tree: OrganelleTreeNode;

    if (prevTree) {
      const prevNodeIds = collectTreeIds(prevTree);
      const newIds = organelleIds.filter(id => !new Set(prevNodeIds).has(id));
      const updated = updateOrganelleTree(prevTree, currentIdSet, newIds, organelleMap);
      tree = updated ?? buildOrganelleTree(organelleIds, organelleMap);
    } else {
      tree = buildOrganelleTree(organelleIds, organelleMap);
    }

    organisms.push({
      id: nextId++,
      organelleIds,
      colorSignature: sig,
      centroidX: cx / total,
      centroidY: cy / total,
      tree,
      crossTypeLinks,
    });
  }

  return organisms;
}

/* ------------------------------------------------------------------ */
/*  Ledger                                                             */
/* ------------------------------------------------------------------ */

const STABILITY_ALPHA = 0.02;
const STABILITY_EPSILON = 0.001;

function buildLedger(
  organelles: ReadonlyArray<OrganelleState>,
  organisms: ReadonlyArray<OrganismState>,
  prevFrame: DetectionFrame | null,
): LedgerState {
  const organellesByType = new Map<number, number>();
  for (const org of organelles) {
    organellesByType.set(org.typeId, (organellesByType.get(org.typeId) ?? 0) + 1);
  }

  const organismsBySignature = new Map<string, number>();
  for (const osm of organisms) {
    organismsBySignature.set(osm.colorSignature, (organismsBySignature.get(osm.colorSignature) ?? 0) + 1);
  }

  // EMA stability scores for organelle types
  const prevOrgStability = prevFrame?.ledger.organelleStability;
  const organelleStability = new Map<number, number>();
  const oneMinusAlpha = 1 - STABILITY_ALPHA;

  for (const [typeId] of organellesByType) {
    const prev = prevOrgStability?.get(typeId) ?? 0;
    organelleStability.set(typeId, STABILITY_ALPHA + oneMinusAlpha * prev);
  }
  if (prevOrgStability) {
    for (const [typeId, prev] of prevOrgStability) {
      if (organellesByType.has(typeId)) continue;
      const decayed = oneMinusAlpha * prev;
      if (decayed > STABILITY_EPSILON) organelleStability.set(typeId, decayed);
    }
  }

  // EMA stability scores for organism signatures
  const prevOsmStability = prevFrame?.ledger.organismStability;
  const organismStability = new Map<string, number>();

  for (const [sig] of organismsBySignature) {
    const prev = prevOsmStability?.get(sig) ?? 0;
    organismStability.set(sig, STABILITY_ALPHA + oneMinusAlpha * prev);
  }
  if (prevOsmStability) {
    for (const [sig, prev] of prevOsmStability) {
      if (organismsBySignature.has(sig)) continue;
      const decayed = oneMinusAlpha * prev;
      if (decayed > STABILITY_EPSILON) organismStability.set(sig, decayed);
    }
  }

  return { organellesByType, organismsBySignature, organelleStability, organismStability };
}

/* ------------------------------------------------------------------ */
/*  Top-level pipeline                                                 */
/* ------------------------------------------------------------------ */

export function runDetection(
  data: ReadbackData,
  prevFrame: DetectionFrame | null,
  config: DetectionConfig,
  dt: number,
  bpm: number,
  forceMatrix: ForceMatrix,
  typeKeys: ReadonlyArray<string>,
): DetectionFrame {
  const beatDuration = 60 / Math.max(20, bpm);
  const organelleHoldDuration = config.organelleLatchBeats * beatDuration;

  const { organelles, holdTimers } = detectOrganelles(data, prevFrame, config, dt, organelleHoldDuration);
  const organisms = detectOrganisms(organelles, prevFrame, config, forceMatrix, typeKeys);
  const ledger = buildLedger(organelles, organisms, prevFrame);

  return { organelles, organisms, holdTimers, ledger };
}

function detectOrganelles(
  data: ReadbackData,
  prevFrame: DetectionFrame | null,
  config: DetectionConfig,
  dt: number,
  holdDuration: number,
): { organelles: OrganelleState[]; holdTimers: Map<number, number> } {
  const { n, posX, posY, velX, velY, particleTypes, particleCells, cellHeads, cellNext, cols, rows } = data;
  const proxSq = config.proximityRadius * config.proximityRadius;
  const cohSq = config.coherenceThreshold * config.coherenceThreshold;

  const { parent, rank } = ufCreate(n);

  // Union same-type neighbors within proximity and coherence thresholds
  // (fresh each tick — no carry-forward from previous frame)
  for (let i = 0; i < n; i++) {
    const cellIdx = particleCells[i];
    const col = cellIdx % cols;
    const row = (cellIdx - col) / cols;
    const ti = particleTypes[i];
    const px = posX[i], py = posY[i];
    const vxi = velX[i], vyi = velY[i];

    for (let dr = -1; dr <= 1; dr++) {
      const nr = row + dr;
      if (nr < 0 || nr >= rows) continue;
      for (let dc = -1; dc <= 1; dc++) {
        const nc = col + dc;
        if (nc < 0 || nc >= cols) continue;
        let j = cellHeads[nr * cols + nc];
        while (j !== -1) {
          if (j > i && particleTypes[j] === ti) {
            const dx = posX[j] - px;
            const dy = posY[j] - py;
            if (dx * dx + dy * dy < proxSq) {
              const dvx = velX[j] - vxi;
              const dvy = velY[j] - vyi;
              if (dvx * dvx + dvy * dvy < cohSq) {
                ufUnion(parent, rank, i, j);
              }
            }
          }
          j = cellNext[j];
        }
      }
    }
  }

  // Count component sizes by root
  const rootSize = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const root = ufFind(parent, i);
    rootSize.set(root, (rootSize.get(root) ?? 0) + 1);
  }

  // Absorption pass: particles in sub-threshold components try to join
  // nearby large components (same type, within proximity, no coherence check).
  // This prevents small clusters from fragmenting off when they're adjacent
  // to an established organelle.
  const minSize = config.minOrganelleSize;
  for (let i = 0; i < n; i++) {
    const ri = ufFind(parent, i);
    if ((rootSize.get(ri) ?? 0) >= minSize) continue; // already in a large component

    const cellIdx = particleCells[i];
    const col = cellIdx % cols;
    const row = (cellIdx - col) / cols;
    const ti = particleTypes[i];
    const px = posX[i], py = posY[i];

    for (let dr = -1; dr <= 1; dr++) {
      const nr = row + dr;
      if (nr < 0 || nr >= rows) continue;
      for (let dc = -1; dc <= 1; dc++) {
        const nc = col + dc;
        if (nc < 0 || nc >= cols) continue;
        let j = cellHeads[nr * cols + nc];
        while (j !== -1) {
          if (j !== i && particleTypes[j] === ti) {
            const rj = ufFind(parent, j);
            if ((rootSize.get(rj) ?? 0) >= minSize) {
              const dx = posX[j] - px;
              const dy = posY[j] - py;
              if (dx * dx + dy * dy < proxSq) {
                // Merge small component into the large one
                const oldRoot = ufFind(parent, i);
                const oldSize = rootSize.get(oldRoot) ?? 0;
                ufUnion(parent, rank, i, j);
                const newRoot = ufFind(parent, i);
                // Update size counts
                if (newRoot !== oldRoot) {
                  rootSize.set(newRoot, (rootSize.get(newRoot) ?? 0) + oldSize);
                  rootSize.set(oldRoot, 0);
                }
              }
            }
          }
          j = cellNext[j];
        }
      }
    }
  }

  // Extract components grouped by root
  const components = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = ufFind(parent, i);
    let arr = components.get(root);
    if (!arr) { arr = []; components.set(root, arr); }
    arr.push(i);
  }

  // Track which particles are freshly detected this tick
  const detectedNow = new Set<number>();

  // Build organelles from components above minimum size
  const organelles: OrganelleState[] = [];
  let nextId = 0;

  for (const [, indices] of components) {
    if (indices.length < config.minOrganelleSize) continue;

    const typeId = particleTypes[indices[0]];
    let cx = 0, cy = 0, vx = 0, vy = 0;
    let minCol = cols, maxCol = 0, minRow = rows, maxRow = 0;

    for (const idx of indices) {
      cx += posX[idx];
      cy += posY[idx];
      vx += velX[idx];
      vy += velY[idx];
      detectedNow.add(idx);
      const cellIdx = particleCells[idx];
      const c = cellIdx % cols;
      const r = (cellIdx - c) / cols;
      if (c < minCol) minCol = c;
      if (c > maxCol) maxCol = c;
      if (r < minRow) minRow = r;
      if (r > maxRow) maxRow = r;
    }

    const len = indices.length;
    organelles.push({
      id: nextId++,
      typeId,
      particleIndices: new Uint32Array(indices),
      centroidX: cx / len,
      centroidY: cy / len,
      avgVelX: vx / len,
      avgVelY: vy / len,
      minCol, maxCol, minRow, maxRow,
    });
  }

  // Hold timers: particles detected now get full hold duration.
  // Particles not detected but with remaining hold time keep counting down.
  const newTimers = new Map<number, number>();
  const prevTimers = prevFrame?.holdTimers;

  for (const pi of detectedNow) {
    newTimers.set(pi, holdDuration);
  }

  if (prevTimers) {
    for (const [pi, timer] of prevTimers) {
      if (pi >= n || detectedNow.has(pi)) continue;
      const remaining = timer - dt;
      if (remaining > 0) newTimers.set(pi, remaining);
    }
  }

  return { organelles, holdTimers: newTimers };
}

/* ------------------------------------------------------------------ */
/*  Organism Registry (stable identity across frames)                  */
/* ------------------------------------------------------------------ */

/**
 * Match current-frame organisms to previous registered organisms by
 * centroid proximity (within same colorSignature). Extrapolates previous
 * centroids forward by dt for more accurate matching.
 *
 * Greedy nearest-neighbor: for each signature, sort candidate pairs by
 * distance and assign 1:1, closest first.
 */
export function updateRegistry(
  prev: OrganismRegistry | null,
  frame: DetectionFrame,
  organelleMap: ReadonlyMap<number, OrganelleState>,
  dt: number,
  matchRadius: number = 80,
  currentTime: number = 0,
): OrganismRegistry {
  const matchRadiusSq = matchRadius * matchRadius;
  let nextId = prev?.nextId ?? 0;

  const prevBySig = new Map<string, RegisteredOrganism[]>();
  if (prev) {
    for (const ro of prev.organisms) {
      let arr = prevBySig.get(ro.colorSignature);
      if (!arr) { arr = []; prevBySig.set(ro.colorSignature, arr); }
      arr.push(ro);
    }
  }

  const currBySig = new Map<string, OrganismState[]>();
  for (const osm of frame.organisms) {
    let arr = currBySig.get(osm.colorSignature);
    if (!arr) { arr = []; currBySig.set(osm.colorSignature, arr); }
    arr.push(osm);
  }

  const result: RegisteredOrganism[] = [];

  const allSigs = new Set([...prevBySig.keys(), ...currBySig.keys()]);
  for (const sig of allSigs) {
    const prevOrgs = prevBySig.get(sig) ?? [];
    const currOrgs = currBySig.get(sig) ?? [];

    const pairs: { pi: number; ci: number; distSq: number }[] = [];
    for (let pi = 0; pi < prevOrgs.length; pi++) {
      const p = prevOrgs[pi];
      const px = p.centroidX + p.velX * dt;
      const py = p.centroidY + p.velY * dt;
      for (let ci = 0; ci < currOrgs.length; ci++) {
        const c = currOrgs[ci];
        const dx = px - c.centroidX;
        const dy = py - c.centroidY;
        const distSq = dx * dx + dy * dy;
        if (distSq <= matchRadiusSq) {
          pairs.push({ pi, ci, distSq });
        }
      }
    }

    pairs.sort((a, b) => a.distSq - b.distSq);
    const usedPrev = new Set<number>();
    const usedCurr = new Set<number>();
    const matched = new Map<number, number>();

    for (const pair of pairs) {
      if (usedPrev.has(pair.pi) || usedCurr.has(pair.ci)) continue;
      usedPrev.add(pair.pi);
      usedCurr.add(pair.ci);
      matched.set(pair.ci, prevOrgs[pair.pi].registryId);
    }

    for (let ci = 0; ci < currOrgs.length; ci++) {
      const osm = currOrgs[ci];
      const prevId = matched.get(ci);
      const matchedPrev = prevId !== undefined
        ? prevOrgs.find(p => p.registryId === prevId)
        : undefined;
      const registryId = prevId ?? nextId++;

      let vx = 0, vy = 0, n = 0;
      for (const orgId of osm.organelleIds) {
        const org = organelleMap.get(orgId);
        if (org) { vx += org.avgVelX; vy += org.avgVelY; n++; }
      }
      if (n > 0) { vx /= n; vy /= n; }

      result.push({
        registryId,
        colorSignature: osm.colorSignature,
        centroidX: osm.centroidX,
        centroidY: osm.centroidY,
        velX: vx,
        velY: vy,
        organelleIds: osm.organelleIds,
        tree: osm.tree,
        creationTime: matchedPrev?.creationTime ?? currentTime,
        crossTypeLinks: osm.crossTypeLinks,
      });
    }
  }

  return { organisms: result, nextId };
}
