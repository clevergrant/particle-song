struct Particle {
  pos:    vec2<f32>,  // 0
  vel:    vec2<f32>,  // 8
  color:  vec4<f32>,  // 16  (rgb + unused alpha)
  typeId: u32,        // 32
  stress: f32,        // 36  sum of individual force magnitudes
  _pad1:  u32,        // 40
  _pad2:  u32,        // 44
};                    // total: 48 bytes

struct SimParams {
  width:              f32,
  height:             f32,
  interactionRadius:  f32,
  baseStrength:       f32,
  damping:            f32,
  dt:                 f32,
  numParticles:       u32,
  numTypes:           u32,
  mouseX:             f32,
  mouseY:             f32,
  mouseForceRadius:   f32,
  mouseForceStrength: f32,
  mouseActive:        u32,  // 0=none, 1=repel, 2=attract
  densityRadius:      f32,
  densityThreshold:   f32,
  densityRepulsion:   f32,
  repelStrength:      f32,
  repelRadius:        f32,
  maxSpeed:           f32,
};

const MAX_TYPES = 32u; // must match TypeScript MAX_TYPES

struct PeakStats {
  stressFrameMax: atomic<u32>,  // current frame's max stress (bitcast f32)
  stressPeak:     atomic<u32>,  // running EMA peak stress (bitcast f32)
  speedFrameMax:  atomic<u32>,  // current frame's max speed (bitcast f32)
  speedPeak:      atomic<u32>,  // running EMA peak speed (bitcast f32)
};

@group(0) @binding(0) var<storage, read>       particlesIn:  array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform>             params:       SimParams;
@group(0) @binding(3) var<storage, read>       forceMatrix:  array<f32>;
@group(0) @binding(4) var<storage, read_write> peakStats: PeakStats;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  var p = particlesIn[i];
  var fx = 0.0;
  var fy = 0.0;
  var stress = 0.0; // sum of individual force magnitudes (not net)
  // Attractive forces tracked separately so density regulation can dampen
  // only attraction without weakening repulsion (which causes clumping).
  var attractX = 0.0;
  var attractY = 0.0;
  var neighborCount = 0.0;
  // Accumulate direction away from nearby neighbors (for density push)
  var pushX = 0.0;
  var pushY = 0.0;

  let radiusSq = params.interactionRadius * params.interactionRadius;
  let densityRadiusSq = params.densityRadius * params.densityRadius;
  let repelRadiusSq = params.repelRadius * params.repelRadius;

  // O(n^2) pairwise force accumulation + density counting
  for (var j = 0u; j < params.numParticles; j++) {
    if (i == j) { continue; }
    let other = particlesIn[j];

    // Shortest toroidal distance
    var dx = p.pos.x - other.pos.x;
    var dy = p.pos.y - other.pos.y;
    if (dx > params.width * 0.5) { dx -= params.width; }
    else if (dx < -params.width * 0.5) { dx += params.width; }
    if (dy > params.height * 0.5) { dy -= params.height; }
    else if (dy < -params.height * 0.5) { dy += params.height; }

    let rawDistSq = dx * dx + dy * dy;

    // When particles sit at the exact same position, give dx/dy a tiny
    // deterministic nudge so the force has a nonzero direction.
    if (rawDistSq < 0.001) {
      // Use particle indices to pick a unique-ish angle
      let angle = f32(i ^ j) * 2.399;          // golden angle
      dx = cos(angle);
      dy = sin(angle);
    }

    // Clamp distance to a minimum of 1px to avoid division by zero
    // while still applying forces to overlapping particles.
    let distSq = max(rawDistSq, 1.0);

    // Count neighbors within density radius and accumulate push direction
    if (densityRadiusSq > 0.0 && distSq < densityRadiusSq) {
      neighborCount += 1.0;
      let dist = sqrt(distSq);
      let t = 1.0 - dist / params.densityRadius;
      // Push away from neighbor (dx points from other to self)
      pushX += (dx / dist) * t;
      pushY += (dy / dist) * t;
    }

    // Universal short-range repulsion between all particles
    if (params.repelStrength > 0.0 && distSq < repelRadiusSq) {
      let dist = sqrt(distSq);
      let t = 1.0 - dist / params.repelRadius;
      let f = params.repelStrength * t;
      fx += (dx / dist) * f;
      fy += (dy / dist) * f;
      stress += f;
    }

    if (distSq > radiusSq) { continue; }

    let dist = sqrt(distSq);
    let r = dist / params.interactionRadius;
    let a = forceMatrix[p.typeId * MAX_TYPES + other.typeId];
    let beta = 0.3;
    var f: f32;
    if (r < beta) {
      f = r / beta - 1.0;
    } else if (r < 1.0) {
      f = a * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
    } else {
      f = 0.0;
    }
    f *= params.baseStrength;
    stress += abs(f);
    if (f > 0.0) {
      // Positive f = attraction → pull toward other (negate dx which points away)
      attractX -= (dx / dist) * f;
      attractY -= (dy / dist) * f;
    } else {
      // Negative f = repulsion → push away (negate dx, negate f → net away)
      fx -= (dx / dist) * f;
      fy -= (dy / dist) * f;
    }
  }

  // Density regulation: dampen attraction only and push outward when crowded.
  // Repulsive forces (fx, fy) are NOT dampened — dampening them caused a
  // feedback loop where crowding weakened repulsion, leading to more crowding.
  if (params.densityThreshold > 0.0 && neighborCount > params.densityThreshold) {
    let excess = (neighborCount - params.densityThreshold) / params.densityThreshold;
    let dampen = 1.0 / (1.0 + excess * params.densityRepulsion);
    attractX *= dampen;
    attractY *= dampen;
    // Gentle outward push — normalize the accumulated push direction,
    // then scale by repulsion strength (independent of baseStrength/neighbor count)
    let pushLen = sqrt(pushX * pushX + pushY * pushY);
    if (pushLen > 0.001) {
      let pushScale = params.densityRepulsion * min(excess, 2.0);
      fx += (pushX / pushLen) * pushScale;
      fy += (pushY / pushLen) * pushScale;
    }
  }

  // Merge attractive forces into final force
  fx += attractX;
  fy += attractY;

  // Mouse force
  if (params.mouseActive > 0u) {
    let sign = select(-1.0, 1.0, params.mouseActive == 1u);
    var dx = p.pos.x - params.mouseX;
    var dy = p.pos.y - params.mouseY;
    if (dx > params.width * 0.5) { dx -= params.width; }
    else if (dx < -params.width * 0.5) { dx += params.width; }
    if (dy > params.height * 0.5) { dy -= params.height; }
    else if (dy < -params.height * 0.5) { dy += params.height; }
    let dist = sqrt(dx * dx + dy * dy);
    if (dist < params.mouseForceRadius && dist > 1.0) {
      let t = 1.0 - dist / params.mouseForceRadius;
      let f = sign * params.mouseForceStrength * t * t / dist;
      fx += f * dx;
      fy += f * dy;
    }
  }

  p.stress = stress;
  atomicMax(&peakStats.stressFrameMax, bitcast<u32>(stress));

  // Integration
  p.vel = p.vel * params.damping + vec2<f32>(fx, fy) * params.dt;

  // Soft speed limiter — tanh curve that asymptotically approaches maxSpeed
  let spd = length(p.vel);
  let cap = params.maxSpeed;
  if (spd > 0.001 && cap > 0.001) {
    let limited = cap * tanh(spd / cap);
    p.vel = p.vel * (limited / spd);
  }

  p.pos += p.vel * params.dt;

  atomicMax(&peakStats.speedFrameMax, bitcast<u32>(length(p.vel)));

  // Toroidal wrap
  p.pos.x = ((p.pos.x % params.width) + params.width) % params.width;
  p.pos.y = ((p.pos.y % params.height) + params.height) % params.height;

  particlesOut[i] = p;
}

// Single-invocation pass: lerp running peaks toward frame max (EMA)
@compute @workgroup_size(1)
fn updatePeaks() {
  let rate = 0.03;

  let stressMax = bitcast<f32>(atomicLoad(&peakStats.stressFrameMax));
  let prevStress = bitcast<f32>(atomicLoad(&peakStats.stressPeak));
  atomicStore(&peakStats.stressPeak, bitcast<u32>(max(mix(prevStress, stressMax, rate), 1.0)));
  atomicStore(&peakStats.stressFrameMax, 0u);

  let speedMax = bitcast<f32>(atomicLoad(&peakStats.speedFrameMax));
  let prevSpeed = bitcast<f32>(atomicLoad(&peakStats.speedPeak));
  atomicStore(&peakStats.speedPeak, bitcast<u32>(max(mix(prevSpeed, speedMax, rate), 1.0)));
  atomicStore(&peakStats.speedFrameMax, 0u);
}
