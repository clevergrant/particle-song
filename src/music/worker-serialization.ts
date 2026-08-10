/**
 * Serialization helpers for the schedule-bar Web Worker.
 *
 * postMessage uses the structured-clone algorithm which cannot handle
 * Map or Set.  These helpers convert the five Map/Set fields in the
 * schedule-worker data flow to plain arrays and back.
 */

import type {
  BarSnapshot,
  GlobalMetrics,
  MusicState,
  ScheduleWorkerBarMeta,
  ScheduleWorkerSlotFill,
  ScheduleWorkerDone,
  ScheduleWorkerMsg,
  SlotNote,
  SnapshotOrganism,
  SpeciesCycle,
  TransitionChord,
} from "./types";
import type { BarContext } from "./hit-scheduler";

/* ── Wire-format type overrides ──────────────────────────────────────── */

type MapWire<K, V> = readonly (readonly [K, V])[];
type SetWire<T> = readonly T[];

export type GlobalMetricsWire = Omit<GlobalMetrics, "freeParticlePercentByType"> & {
  readonly freeParticlePercentByType: MapWire<number, number>;
};

export type SnapshotOrganismWire = Omit<SnapshotOrganism, "composition" | "typeAdjacency"> & {
  readonly composition: MapWire<number, number>;
  readonly typeAdjacency: SetWire<number>;
};

export type BarSnapshotWire = Omit<BarSnapshot, "organisms" | "globalMetrics"> & {
  readonly organisms: readonly SnapshotOrganismWire[];
  readonly globalMetrics: GlobalMetricsWire;
};

export type SpeciesCycleWire = {
  readonly played: MapWire<string, SetWire<number>>;
};

export type TransitionChordWire = Omit<TransitionChord, "pitchClasses"> & {
  readonly pitchClasses: SetWire<number>;
};

export type MusicStateWire = Omit<MusicState, "bufferChord" | "speciesCycle"> & {
  readonly bufferChord: TransitionChordWire | null;
  readonly speciesCycle: SpeciesCycleWire;
};

/* ── Serialize (main → worker) ───────────────────────────────────────── */

function serializeMap<K, V>(m: ReadonlyMap<K, V>): MapWire<K, V> {
  return [...m.entries()];
}

function serializeSet<T>(s: ReadonlySet<T>): SetWire<T> {
  return [...s];
}

function serializeSpeciesCycle(sc: SpeciesCycle): SpeciesCycleWire {
  return {
    played: [...sc.played.entries()].map(([k, v]) => [k, [...v]] as const),
  };
}

function serializeTransitionChord(tc: TransitionChord): TransitionChordWire {
  return { ...tc, pitchClasses: serializeSet(tc.pitchClasses) };
}

export function serializeBarSnapshot(snapshot: BarSnapshot): BarSnapshotWire {
  return {
    ...snapshot,
    organisms: snapshot.organisms.map(org => ({
      ...org,
      composition: serializeMap(org.composition),
      typeAdjacency: serializeSet(org.typeAdjacency),
    })),
    globalMetrics: {
      ...snapshot.globalMetrics,
      freeParticlePercentByType: serializeMap(snapshot.globalMetrics.freeParticlePercentByType),
    },
  };
}

export function serializeMusicState(state: MusicState | null): MusicStateWire | null {
  if (!state) return null;
  return {
    ...state,
    bufferChord: state.bufferChord
      ? serializeTransitionChord(state.bufferChord)
      : null,
    speciesCycle: serializeSpeciesCycle(state.speciesCycle),
  };
}

/* ── Deserialize (worker → main, or inside worker) ───────────────────── */

function deserializeMap<K, V>(arr: MapWire<K, V>): ReadonlyMap<K, V> {
  return new Map(arr);
}

function deserializeSet<T>(arr: SetWire<T>): ReadonlySet<T> {
  return new Set(arr);
}

function deserializeSpeciesCycle(sc: SpeciesCycleWire): SpeciesCycle {
  return {
    played: new Map(sc.played.map(([k, v]) => [k, new Set(v)])),
  };
}

function deserializeTransitionChord(tc: TransitionChordWire): TransitionChord {
  return { ...tc, pitchClasses: deserializeSet(tc.pitchClasses) };
}

export function deserializeBarSnapshot(data: BarSnapshotWire): BarSnapshot {
  return {
    ...data,
    organisms: data.organisms.map(org => ({
      ...org,
      composition: deserializeMap(org.composition),
      typeAdjacency: deserializeSet(org.typeAdjacency),
    })),
    globalMetrics: {
      ...data.globalMetrics,
      freeParticlePercentByType: deserializeMap(data.globalMetrics.freeParticlePercentByType),
    },
  };
}

export function deserializeMusicState(data: MusicStateWire | null): MusicState | null {
  if (!data) return null;
  return {
    ...data,
    bufferChord: data.bufferChord
      ? deserializeTransitionChord(data.bufferChord)
      : null,
    speciesCycle: deserializeSpeciesCycle(data.speciesCycle),
  };
}

/* ── Streaming worker messages ─────────────────────────────────────────── */

/** Wire format for bar-meta (Maps/Sets → arrays). */
export type BarMetaWire = Omit<ScheduleWorkerBarMeta, "bufferChord" | "speciesCycle" | "chordPitchClasses"> & {
  readonly bufferChord: TransitionChordWire | null;
  readonly speciesCycle: SpeciesCycleWire;
  readonly chordPitchClasses: SetWire<number>;
};

/** Union of all wire-format worker messages. */
export type ScheduleWorkerMsgWire =
  | BarMetaWire
  | ScheduleWorkerSlotFill
  | ScheduleWorkerDone;

/** Serialize a bar-meta message for postMessage. */
export function serializeBarMeta(
  ctx: BarContext,
  id: number,
  barNumber: number,
): BarMetaWire {
  return {
    kind: "bar-meta",
    id,
    barNumber,
    mode: ctx.mode,
    rootMidi: ctx.rootMidi,
    isBufferBar: ctx.isBufferBar,
    bufferChord: ctx.bufferChord ? serializeTransitionChord(ctx.bufferChord) : null,
    netStability: ctx.netStability,
    spatialEntropy: ctx.spatialEntropy,
    envelopeRanges: ctx.envelopeRanges,
    speciesCycle: serializeSpeciesCycle(ctx.speciesCycle),
    chordPitchClasses: serializeSet(ctx.chordPitchClasses),
  };
}

/** Deserialize a bar-meta wire message on the main thread. */
export function deserializeBarMeta(wire: BarMetaWire): ScheduleWorkerBarMeta {
  return {
    ...wire,
    bufferChord: wire.bufferChord ? deserializeTransitionChord(wire.bufferChord) : null,
    speciesCycle: deserializeSpeciesCycle(wire.speciesCycle),
    chordPitchClasses: deserializeSet(wire.chordPitchClasses),
  };
}
