/**
 * Tiny pure helpers used across the music2 engine.
 * Kept dependency-free so anything can pull from here without cycles.
 */

import { A4_FREQ_HZ, A4_MIDI, MIDI_RANGE } from "../constants"

export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v)

export const clamp = (v: number, lo: number, hi: number): number =>
	v < lo ? lo : v > hi ? hi : v

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

export const midiToFreq = (midi: number): number =>
	A4_FREQ_HZ * Math.pow(2, (midi - A4_MIDI) / 12)

export const clampMidi = (m: number): number =>
	clamp(Math.round(m), MIDI_RANGE.min, MIDI_RANGE.max)
