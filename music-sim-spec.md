# Particle Simulation Music Engine — Specification

## 1. Overview

A music generation engine driven by a 2D particle simulation with attraction/repulsion force dynamics. The simulation produces emergent structures (organelles, organisms) whose properties are mapped to musical parameters. Each organism becomes a self-contained polyrhythmic instrument. The overall music is the superposition of all living organisms' grooves, layered over a continuous drone that anchors the tonal center.

---

## 2. Simulation Data Model

### 2.1 Particles

- Position (x, y)
- Velocity vector
- Speed (scalar magnitude of velocity)
- Type (discrete category, e.g. A, B, C, D...)

### 2.2 Organelles

A collection of particles of the same type that satisfy:
- All particles fall within a defined spatial **position** radius of each other
- All particles fall within a defined **velocity** difference threshold of each other

Derived properties:
- **Type** — inherited from particle type
- **Centroid** — mean position of constituent particles
- **Centroid velocity** — mean velocity of constituent particles
- **Particle count** — number of particles in the organelle
- **Spatial radius** — bounding radius or variance of particle positions
- **Density** — particle count / area
- **Creation time** — timestamp at formation (age = `current_time - creation_time`)

Constraints:
- Organelles cannot bond with other organelles of the same type
- An organelle belongs to at most one organism at a time

### 2.3 Organisms

A collection of different types of organelles whose centroids satisfy:
- All organelle centroids fall within a defined spatial **position** radius of each other
- All organelle centroids fall within a defined **velocity** difference threshold of each other

Derived properties:
- **Centroid** — mean of organelle centroids
- **Centroid velocity** — mean of organelle centroid velocities
- **Composition** — map of organelle type → count (e.g. {A: 1, B: 5, C: 3})
- **Organelle count** — total number of organelles
- **Total particle count** — sum of all organelle particle counts
- **Spatial spread** — variance of organelle centroid positions
- **Complexity** — number of distinct organelle types
- **Creation time** — timestamp at formation (age = `current_time - creation_time`)

### 2.4 Detection

Organelle and organism detection use the same generic algorithm (cluster by position proximity + velocity similarity) at two scales:

| | Organelle Detection | Organism Detection |
|---|---|---|
| **Input entities** | Particles | Organelle centroids |
| **Position threshold** | Particle radius | Organism radius |
| **Velocity threshold** | Particle velocity | Centroid velocity |
| **Type constraint** | Same type only | Different types only |

A single clustering function serves both, parameterized by threshold values and type constraint direction.

### 2.5 Free Particles

All particles not currently bound to any organelle.
- **Free particle count** — global scalar
- **Free particle percentage per type** — fraction of each type's particles not in organelles

### 2.6 Global Metrics

Computed from the full simulation state:
- **Average velocity** — mean speed across all particles
- **Average organelle density** — mean density across all organelles
- **Species count** — number of distinct organism species (unique compositions) currently alive
- **Organism count** — total number of living organisms
- **Free particle count** — particles not bound to any organelle
- **Spatial entropy** — how evenly distributed particles are (high = spread out, low = clustered)

### 2.7 Events

Discrete moments detected per tick:
- Organelle formation / dissolution
- Organism formation / dissolution
- Organelle joining / leaving an organism

---

## 3. Musical Architecture

### 3.1 Core Principle — Music-First Timing

**The music engine is the authority on timing, not the simulation.**

The simulation runs in continuous time. The music engine operates on a bar/beat grid. Simulation state is **sampled at bar boundaries** — the music engine reads a snapshot once per bar and uses it to schedule the next bar. Between bar boundaries, the music is locked and stable regardless of simulation activity.

This means:
- No note, voice, rhythm, scale, or key change occurs mid-bar unless already scheduled at the bar boundary.
- If an organism forms mid-bar, it doesn't sound until the next bar.
- If an organelle joins or leaves mid-bar, the subdivision doesn't change until the next bar.
- If an organism dissolves mid-bar, its current bar plays to completion, then the voice stops.
- If an organism crosses an age threshold into a new overtone phase, the new partial becomes available at the next bar.
- Scale and key changes are evaluated at bar boundaries and may trigger a transition buffer bar (§3.6).

**The bar is the atom of musical change.**

Exception: the drone layer (§9) operates independently of the bar grid.

### 3.2 Fire-and-Forget Scheduling

The music engine operates as a two-phase loop:

**Phase A — Sample (at bar boundary):**
1. Snapshot all simulation state: organism list, compositions, all organelle properties (speed, size, density, spatial radius, angular position relative to organism velocity vector), and global metrics (§2.6).
2. Compute every hit for the upcoming bar: which voices play, which subdivision each hit falls on, and each hit's pitch, octave, volume, filter cutoff, envelope parameters, and vibrato depth.
3. Schedule all computed hits.

**Phase B — Play (during the bar):**
1. Execute scheduled hits at their designated times.
2. Do not read simulation state.
3. For each hit, light up the corresponding organelle on screen. The **brightness follows the envelope curve** (§5.4) — ramping up with attack, holding at sustain, fading with release. If the organelle no longer exists, the visual callback fails silently. The note still plays.

### 3.3 Species as Rhythm

An organism's composition map determines its polyrhythm. Each organelle type present is a rhythmic voice. The count of that type determines its subdivision.

Example — composition {Yellow: 1, Red: 5, Green: 3}:
- Yellow: 1 hit per bar (whole note)
- Red: 5 evenly spaced hits (quintuplet)
- Green: 3 evenly spaced hits (triplet)

Each individual organelle gets its own hit within the subdivision. Two organisms with identical composition play the same polyrhythmic pattern — they are the same species.

#### Rhythmic Phase Offset (Anchor + Fill)

To prevent all types from piling onto the downbeat, types within an organism are assigned **rhythmic roles**. Types are sorted by ID; the lowest ID is the **anchor** and plays on the downbeat (zero offset). Remaining types phase-offset their subdivision grids so their first hits interleave into the spaces between anchor hits:

```text
offset(typeIndex) = (typeIndex / numTypes) × (barDuration / subdivision)
```

- **Anchor (typeIndex 0):** offset = 0 — grounds the rhythm on beat 1.
- **Fill types (typeIndex 1…N−1):** offset increases linearly, spreading first hits across the anchor's first subdivision cell.

Example — composition {Yellow: 1, Red: 5, Green: 3} (sorted by type ID):

- Yellow (anchor): hits at [0/1] of bar — downbeat
- Red (fill 1): 5 hits offset by 1/3 × (barDur/5) — first hit lands ~6.7% into the bar
- Green (fill 2): 3 hits offset by 2/3 × (barDur/3) — first hit lands ~22.2% into the bar

### 3.4 Visual Ordering Within a Type

Organelles within a type are ordered by **alternating left-right from the organism's velocity vector** for **visual light-up purposes only**. This ordering does not affect pitch, volume, or any other sonic parameter — temporal hit ordering within a subdivision uses a stable index (e.g., organelle ID order).

The velocity vector defines forward; organelles are sorted by angular offset, then interleaved: closest right, closest left, next right, next left, etc.

Example with 5 organelles at angles -10°, -35°, +15°, +40°, +60° from the velocity vector:
- Visual light-up order: +15° (R1), -10° (L1), +40° (R2), -35° (L2), +60° (R3)

This produces a visible oscillating pulse swinging outward from the organism's direction of motion. Different types oscillate independently at their own subdivision rates. The velocity vector direction is captured in the bar-boundary snapshot, so ordering is locked for the entire bar.

### 3.5 High Count Handling

Maximum subdivision per type is **16 hits per bar**. If a type has more than 16 members, hits wrap using modulo: organelle N plays on hit `N % 16`. Multiple organelles light up simultaneously on the same hit, but the rhythm never exceeds 16th note density.

### 3.6 Transition Buffer System

At every bar boundary, the music engine compares the incoming state (new scale + key) against the outgoing state (current scale + key). If the change exceeds a **dissonance threshold**, a transition buffer bar is automatically inserted.

**Dissonance evaluation:**
The dissonance score is the number of differing pitch classes between outgoing and incoming sets (0–12). Examples:
- Same key, Ionian → Mixolydian: 1 note differs → no buffer
- Key change by a fifth, same mode: 1 note differs → no buffer
- Key change by a half step, same mode: 5+ notes differ → buffer
- Key change by a tritone + mode shift → buffer

**Buffer bar behavior:**
- Plays only pitch classes common to both outgoing and incoming sets (intersection)
- All organisms are constrained to this intersection for the bar
- Non-committal — belongs to both keys simultaneously
- If the change reverses next bar, the music drops back seamlessly
- If the change persists, the next bar lands in the new state
- If the target changes during a buffer, the next bar evaluates the buffer's intersection against the new target. If still too dissonant, another buffer is inserted using the intersection of the current buffer and the new target. **Consecutive buffers converge** — the pitch set can only narrow or stay the same, never grow, so the music always approaches resolution

Buffer bars are inherently harmonically safe. Back-to-back buffers represent genuine instability. Protection against unnecessary flickering comes from scale change hysteresis (§4.2.1), not from limiting buffers.

**Dissonance threshold: 3 differing pitch classes.** This aligns with the music theory distinction between closely related keys (1–2 differences, no preparation needed) and distantly related keys (3+, traditionally require a pivot chord — which is what the buffer bar is).

---

## 4. Pitch

### 4.1 Type-to-Pitch Mapping

Each particle/organelle type maps to a scale degree. This mapping is fixed for the simulation lifetime (or configured at startup).

Example:
- Type A → degree 1 (root)
- Type B → degree 3 (third)
- Type C → degree 5 (fifth)
- Type D → degree 7 (seventh)

All organelles of a given type play the same pitch class.

### 4.2 Scale/Mode Selection ← Global Net Stability

The scale is an interval set — semitone offsets from root. Examples:
- Major pentatonic: [0, 2, 4, 7, 9]
- Natural minor: [0, 2, 3, 5, 7, 8, 10]
- Harmonic minor: [0, 2, 3, 5, 7, 8, 11]
- Whole tone: [0, 2, 4, 6, 8, 10]

**Global net stability** drives scale selection, derived from three factors:

1. **Species diversity** (positive — more species = more stable)
2. **Average velocity** (inverse — lower velocity = more stable)
3. **Average organelle density** (positive — denser = more stable)

Formula:
```
net_stability = (sigmoid(species_count) + sigmoid(1 / avg_velocity) + sigmoid(avg_density)) / 3
```
Where `sigmoid(x) = 1 / (1 + e^(-steepness * (x - midpoint)))`. Each signal has its own adjustable midpoint and steepness. No adaptation — sustained chaos stays mapped to low stability indefinitely. Midpoint and steepness are tuned per simulation based on typical observed values.

All organisms share one scale:

| Net Stability | Mode |
|---|---|
| 0.89 – 1.00 | Lydian (brightest) |
| 0.78 – 0.89 | Ionian / Major |
| 0.67 – 0.78 | Mixolydian |
| 0.56 – 0.67 | Dorian |
| 0.44 – 0.56 | Aeolian / Natural Minor |
| 0.33 – 0.44 | Phrygian |
| 0.22 – 0.33 | Locrian |
| 0.11 – 0.22 | Whole tone |
| 0.00 – 0.11 | Chromatic |

Scale changes are evaluated at bar boundaries and may trigger a buffer (§3.6).

### 4.2.1 Scale Change Hysteresis

Scale changes require stability to cross a band threshold by a margin of **0.03 (adjustable)**. Example: in Dorian (0.56–0.67), stability must rise above 0.70 for Mixolydian or fall below 0.53 for Aeolian. Hovering between 0.53 and 0.70 keeps the mode locked. Increase the margin if modes change too often; decrease if the music feels unresponsive.

### 4.3 Register & Harmonic Complexity — Overtone Series Progression

An organism's age determines its harmonic vocabulary by recapitulating the overtone series (octaves → fifths → thirds → sevenths → chromaticism).

#### 4.3.1 Register Width ← Organism Count

- 1 organism: octave 4
- 2–3 organisms: octaves 3–5
- 4–6 organisms: octaves 2–6
- 7+ organisms: octaves 1–7

#### 4.3.2 Harmonic Phases ← Organism Age

Each phase unlocks new intervals. The vocabulary accumulates.

**Phase 1 — Birth: Fundamental only.** All types play root.

**Phase 2 — Young: Octave (2:1).** Octave above/below available. Bass opens.

**Phase 3 — Establishing: Fifth (3:2).** First moment the organism sounds like more than a single tone.

**Phase 4 — Maturing: Third (5:4).** Emotional color. Major/minor determined by global scale (§4.2).

**Phase 5 — Complex: Seventh (7:4).** Tension, harmonic pull.

**Phase 6 — Ancient: Upper partials (9ths, 11ths, 13ths).** Dense, chromatic.

#### 4.3.3 Interaction with Type-to-Pitch Mapping

Types mapped to unavailable scale degrees collapse to the nearest available pitch:
- Phase 1: all types → root
- Phase 2: root or octave-displaced root
- Phase 3: 5th degree becomes active
- Phase 4: 3rd degree becomes active
- Phase 5+: 7th and extensions become active

An organism's rhythm is present from birth. Its harmonic identity unfolds over time.

#### 4.3.4 Pitch Resolution Order

For any hit, pitch is determined in sequence:
1. **Type** → pitch class (§4.1)
2. **Overtone phase** → is that pitch class available yet? (§4.3.2)
3. **Size** → which octave, constrained by register width (§4.3.1, §5.2)

#### 4.3.5 Phase Timing

Each phase lasts a configurable number of bars (**phase rate**, user-controlled):
- **1 bar per phase** — fast aging (Phase 6 in 6 bars)
- **2 bars per phase** — slow aging (Phase 6 in 12 bars)

Computed at bar boundaries: `phase = floor((current_time - creation_time) / (bar_duration * phase_rate)) + 1`, capped at 6.

### 4.4 Root/Key Center ← Species Supremacy

The root note is determined by which species holds the **oldest living organism**. Key changes are evaluated at bar boundaries and may trigger a buffer (§3.6).

#### 4.4.1 Root Assignment — Derived from Force Matrix

Computed at startup (recomputed on force matrix reshuffle). Physical relationships between particle types map to harmonic distances on the **circle of fifths**:

- Types that **attract** → roots **close** on the circle (smooth key changes)
- Types that **repel** → roots **far apart** (dramatic key changes)

Gentle modulation = an ally took over. Harsh modulation = a competitor displaced.

**Computation:**
1. Extract pairwise affinity scores from the force matrix
2. Normalize to distance (attraction = small, repulsion = large)
3. Map species to circle-of-fifths positions that approximate these distances (constraint-satisfaction / graph embedding)

TBD: exact optimization algorithm.

---

## 5. Per-Hit Expression

Each hit is shaped by properties of the specific organelle, sampled at the bar boundary (§3.2):

### 5.1 Volume ← Particle Count

More particles = louder. Linear or logarithmic mapping to amplitude (0.0–1.0).

### 5.2 Octave Offset ← Particle Count

Larger organelles play lower octaves. Smaller play higher. Constrained by register width (§4.3.1) and overtone phase (§4.3.2). Two organelles of the same type can play different octaves if they differ in size.

### 5.3 Bandpass Filter Cutoff ← Particle Count

Larger organelles = lower cutoff (warm, muffled). Smaller = higher cutoff (bright, thin). Applied per-hit.

### 5.4 Envelope ← Physics Signals

Each envelope parameter is driven directly by a physics signal:

- **Attack duration ← centroid speed.** Fast = short attack (percussive). Slow = long attack (swell).
- **Decay duration ← density.** Dense = short decay. Diffuse = long decay.
- **Sustain level ← particle count.** Big = high sustain. Small = low sustain.
- **Release duration ← spatial radius.** Tight = short release (clean cutoff). Spread = long release (lingering tail).

Curve shapes per segment are configurable per type (§8). Durations and levels are computed from the snapshot.

The envelope also drives the **visual brightness** of the organelle light-up (§3.2).

### 5.5 Vibrato Depth ← Centroid Speed

Stationary = clean tone. Fast = pitch wobble.

- **Depth**: 0 cents (stationary) to ±100 cents (fast).
- **Rate**: adjustable parameter (default 5–6 Hz).

---

## 6. Spatial Audio

### 6.1 Stereo Panning ← Organism Centroid X Position

Left edge = hard left. Right edge = hard right. Center = center. All organelles share the organism's panning. Screen wrapping may cause abrupt pan shifts; this is acceptable.

### 6.2 Reverb ← Global Spatial Entropy

High spatial entropy = large reverb. Low = dry. Applied globally.

---

## 7. Tempo and Timing

### 7.1 Global Tempo — User Controlled

BPM is set by the user and remains constant unless manually changed. A **time multiplier** radio selection allows switching between:
- **Half time** (0.5×) — subdivisions play at half speed
- **Normal time** (1×) — default
- **Double time** (2×) — subdivisions play at double speed

### 7.2 Timing via Timestamps

All timing is derived from timestamps, never from frame counters:

- **Sound enabled time** (`t_sound_start`) — bar N starts at `t_sound_start + N * bar_duration`. Deterministic from this single value.
- **Organism / organelle creation time** — age = `current_time - creation_time`.

Bar duration = `(60 / BPM) * beats_per_bar`. If tempo changes, `t_sound_start` and bar count are recalculated to maintain continuity.

### 7.3 Bar-Boundary Sampling

The music engine samples simulation state once per bar boundary (§3.1). Subdivisions are fixed for the bar once it begins.

### 7.4 Formation Timing

New organisms begin playing at the bar boundary following their detection, provided they have satisfied all detection criteria continuously for one full bar (§10.1).

---

## 8. Timbre — Derived from Force Matrix

Computed at startup (recomputed on force matrix reshuffle), consistent with root assignment (§4.4.1).

### 8.1 Sociability Score

For each particle type, compute the average of its attraction/repulsion values against all other types. Normalize to 0.0–1.0 across all types.

### 8.2 Continuous Waveform Blend

Sociability maps to a continuous waveform spectrum:
- **0.0 (antisocial)** → sawtooth (all harmonics, harsh)
- **0.33** → square (odd harmonics, buzzy)
- **0.66** → triangle (few harmonics, soft)
- **1.0 (sociable)** → sine (fundamental only, warm)

With 8 types, each sits at a different point on the spectrum.

### 8.3 Envelope Curve Shapes

Each type gets configurable curve shapes per envelope segment (attack, decay, release): linear, exponential, logarithmic, ease-in, ease-out, etc. Durations and sustain level come from physics (§5.4); curves come from type configuration.

TBD: whether curve shapes should also derive from the force matrix.

---

## 9. Background Texture Layer — Drone

A continuous drone layer independent of the bar/beat grid. Each voice uses its type's waveform (§8.2).

### 9.1 Two Phases

**Before any organisms exist:** A **stack of perfect fourths**, one pitch per particle type, constrained to the current scale (§4.2) and key (§4.4). Dense, ambiguous — the "orchestra warming up."

**Once organisms exist:** The quartal stack collapses into a **root + fifth pedal**. All types contribute to either root or fifth, with octave placement determined by the expanding spiral (§9.2). The drone anchors the tonal center while organisms handle melody and harmony.

### 9.2 Octave Placement — Expanding Spiral

Types are sorted by **average repulsion force** (highest first). Starting from center (octave 4), voices spiral outward, alternating low and high:

| Drone # | Offset | Octave | Pitch |
|---|---|---|---|
| 1 | -1 | 3 | Root |
| 2 | +1 | 5 | Fifth |
| 3 | -2 | 2 | Root |
| 4 | +2 | 6 | Fifth |
| 5 | -3 | 1 | Root |
| 6 | +3 | 7 | Fifth |
| 7+ | ... | wraps | ... |

Odd drones go low (root), even go high (fifth). Each pair expands one octave further. If the octave exceeds A1–C8, it wraps via modulo. The middle octaves remain clear for organism voices.

### 9.3 Volume per Type ← Free Particle Percentage

Each drone voice's volume = percentage of that type's particles not currently in organelles. 100% free = full volume. 0% free = silence. Volume changes are continuous and smoothed, not quantized to bar boundaries.

### 9.4 Independence from BPM

The drone is not tied to the bar grid. Pitches sustain indefinitely. Volume drifts in real time.

When a scale or key change occurs, drone pitches transition using one of two modes (**user-togglable**):
- **Crossfade** — old pitch releases while new pitch attacks. Brief overlap.
- **Glide** — portamento to the new pitch (default ~1–2 seconds). No new note triggered.

### 9.5 Envelope Behavior

Drone voices use a **forced soft, slow attack** regardless of physics signals. The drone is always ambient and non-percussive. Physics-driven envelopes (§5.4) apply only to organism hits.

### 9.6 Lifecycle

- No organisms: full quartal stack
- First organism forms: stack collapses to root + fifth pedal
- Mature ecosystem: drone thins as particles bind, organisms dominate
- Force matrix reshuffle: organisms dissolve, drone swells to quartal stack, then collapses again as new organisms form

---

## 10. Lifecycle Events

All events are governed by bar-boundary sampling (§3.1). The simulation detects events at any moment; the music engine acts at the next bar line.

### 10.1 Organism Formation

An organism must continuously satisfy all detection criteria (§2.3) for a **configurable fraction of a bar** before being recognized. This threshold is a float quantized to the tenths place (e.g., 0.5, 1.0, 1.5 bars). Default: **0.5 bars**. Transient clusters that briefly meet thresholds are never registered.

Once qualified:
- Voices introduced at the next bar boundary
- Polyrhythmic pattern begins
- Creation time is set to the start of the qualification bar
- Starts in Phase 1 (fundamental only)

### 10.2 Organism Dissolution

- Current bar plays to completion
- Voice removed at next bar boundary
- If the organism re-stabilizes before the bar boundary, dissolution is cancelled

### 10.3 Organelle Joining an Organism

- No immediate musical effect
- At next bar boundary: subdivision recalculated, angular ordering reshuffled

### 10.4 Organelle Leaving an Organism

- No immediate musical effect
- At next bar boundary: subdivision shrinks, organelles reshuffle
- If last of a type leaves, that voice drops out

### 10.5 Organelle Formation/Dissolution

- Formation: no musical effect unless the organelle joins an organism at a subsequent bar boundary
- Dissolution: if in an organism, treated as leaving (§10.4)

---

## 11. Open Questions

1. Root assignment optimization algorithm — deferred to implementation (§4.4.1)
2. Whether envelope curve shapes should derive from force matrix (§8.3) — deferred
