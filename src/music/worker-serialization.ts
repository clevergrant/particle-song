/**
 * Serialization helpers for the schedule-bar Web Worker.
 *
 * postMessage uses the structured-clone algorithm which cannot handle
 * Map or Set.  These helpers convert the five Map/Set fields in the
 * scheduleBar data flow to plain arrays and back.
 */

import type {
  BarSnapshot,
  BassUpdate,
  GlobalMetrics,
  MusicState,
  ScheduledBar,
  SnapshotOrganism,
  SpeciesCycle,
  TransitionChord,
} from "./types";

/* ── Wire-format type overrides ──────────────────────────────────────── */

type MapWire<K, V> = readonly (readonly [K, V])[];
type SetWire<T> = readonly T[];

export type GlobalMetricsWire = Omit<GlobalMetrics, "freeParticlePercentByType"> & {
  readonly freeParticlePercentByType: MapWire<number, number>;
};

export type SnapshotOrganismWire = Omit<SnapshotOrganism, "composition"> & {
  readonly composition: MapWire<number, number>;
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

export type BassUpdateWire = Omit<BassUpdate, "freeParticlePercentByType"> & {
  readonly freeParticlePercentByType: MapWire<number, number>;
};

export type ScheduledBarWire = Omit<ScheduledBar, "bassUpdate" | "bufferChord" | "speciesCycle"> & {
  readonly bassUpdate: BassUpdateWire;
  readonly bufferChord: TransitionChordWire | null;
  readonly speciesCycle: SpeciesCycleWire;
};

export type MusicStateWire = Omit<MusicState, "prevScheduledBar" | "bufferChord" | "speciesCycle"> & {
  readonly prevScheduledBar: ScheduledBarWire | null;
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

function serializeBassUpdate(bu: BassUpdate): BassUpdateWire {
  return { ...bu, freeParticlePercentByType: serializeMap(bu.freeParticlePercentByType) };
}

export function serializeScheduledBar(bar: ScheduledBar): ScheduledBarWire {
  return {
    ...bar,
    bassUpdate: serializeBassUpdate(bar.bassUpdate),
    bufferChord: bar.bufferChord ? serializeTransitionChord(bar.bufferChord) : null,
    speciesCycle: serializeSpeciesCycle(bar.speciesCycle),
  };
}

export function serializeBarSnapshot(snapshot: BarSnapshot): BarSnapshotWire {
  return {
    ...snapshot,
    organisms: snapshot.organisms.map(org => ({
      ...org,
      composition: serializeMap(org.composition),
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
    prevScheduledBar: state.prevScheduledBar
      ? serializeScheduledBar(state.prevScheduledBar)
      : null,
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

function deserializeBassUpdate(bu: BassUpdateWire): BassUpdate {
  return { ...bu, freeParticlePercentByType: deserializeMap(bu.freeParticlePercentByType) };
}

export function deserializeScheduledBar(bar: ScheduledBarWire): ScheduledBar {
  return {
    ...bar,
    bassUpdate: deserializeBassUpdate(bar.bassUpdate),
    bufferChord: bar.bufferChord ? deserializeTransitionChord(bar.bufferChord) : null,
    speciesCycle: deserializeSpeciesCycle(bar.speciesCycle),
  };
}

export function deserializeBarSnapshot(data: BarSnapshotWire): BarSnapshot {
  return {
    ...data,
    organisms: data.organisms.map(org => ({
      ...org,
      composition: deserializeMap(org.composition),
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
    prevScheduledBar: data.prevScheduledBar
      ? deserializeScheduledBar(data.prevScheduledBar)
      : null,
    bufferChord: data.bufferChord
      ? deserializeTransitionChord(data.bufferChord)
      : null,
    speciesCycle: deserializeSpeciesCycle(data.speciesCycle),
  };
}
