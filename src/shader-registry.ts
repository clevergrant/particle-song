/* ------------------------------------------------------------------ */
/*  Shader effect registry                                             */
/*  Each effect provides only the @fragment fn fs_main(...) body.      */
/*  The registry concatenates it with the shared prefix (structs,      */
/*  bindings, vertex shader) to produce the full WGSL source.          */
/* ------------------------------------------------------------------ */

// --- Particle effect fragments ---
import gradientFrag from "./shaders/particle-effects/gradient.wgsl?raw";
import solidFrag from "./shaders/particle-effects/solid.wgsl?raw";
import speedColorFrag from "./shaders/particle-effects/speed-color.wgsl?raw";
import stressColorFrag from "./shaders/particle-effects/stress-color.wgsl?raw";

// --- Post-process effect fragments ---
import normalizeFrag from "./shaders/quad-effects/normalize.wgsl?raw";
import chromaticFrag from "./shaders/quad-effects/chromatic.wgsl?raw";
import crtFrag from "./shaders/quad-effects/crt.wgsl?raw";
import paletteFrag from "./shaders/quad-effects/palette.wgsl?raw";
import metaballFrag from "./shaders/quad-effects/metaball.wgsl?raw";
import duotoneFrag from "./shaders/quad-effects/duotone.wgsl?raw";
import edgeGlowFrag from "./shaders/quad-effects/edge-glow.wgsl?raw";
import stainFrag from "./shaders/quad-effects/stain.wgsl?raw";
import topographicFrag from "./shaders/quad-effects/topographic.wgsl?raw";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface ShaderEffectParam {
  slot: number;     // maps to param0..param3 in the WGSL struct
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface ShaderEffect {
  id: string;
  name: string;
  category: "particle" | "postprocess";
  fragmentSrc: string;
  params?: ShaderEffectParam[];
}

/* ------------------------------------------------------------------ */
/*  Shared shader prefixes                                             */
/*  Everything EXCEPT the @fragment fn fs_main(...)                    */
/* ------------------------------------------------------------------ */

/** Particle render prefix: structs, bindings, vertex shader */
export const PARTICLE_PREFIX = /* wgsl */ `
// Shared particle struct — must match compute shader layout
struct Particle {
  pos:    vec2<f32>,
  vel:    vec2<f32>,
  color:  vec4<f32>,
  typeId: u32,
  stress: f32,
  _pad1:  u32,
  _pad2:  u32,
};

struct RenderParams {
  resolution: vec2<f32>,
  pointSize:  f32,
  mode:       u32,
  time:       f32,
  param0:     f32,
  param1:     f32,
  param2:     f32,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv:    vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) speed: f32,
  @location(3) stress: f32,
  @location(4) detection: vec2<f32>,  // x=organelle(0/1), y=organism(0/1)
  @location(5) organelleId: f32,     // raw organelle ID (1-based, 0=none)
  @location(6) organismId: f32,     // raw organism ID (1-based, 0=none)
};

struct PeakInfo {
  _stressFrameMax: u32,
  stressPeak:      f32,
  _speedFrameMax:  u32,
  speedPeak:       f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform>       renderParams: RenderParams;
@group(0) @binding(2) var               falloffLUT: texture_2d<f32>;
@group(0) @binding(3) var               falloffSampler: sampler;
@group(0) @binding(4) var<storage, read> peakInfo: PeakInfo;
@group(0) @binding(5) var<storage, read> detectionData: array<u32>;
@group(0) @binding(6) var<storage, read> radiusScales: array<f32>;

// Detection overlay: adds a light ring around detected particles
fn detectionRing(uv: vec2<f32>, det: vec2<f32>, baseColor: vec3<f32>) -> vec3<f32> {
  let dist = length(uv);
  let inOrganelle = det.x > 0.5;
  let inOrganism = det.y > 0.5;
  if (!inOrganelle) { return vec3<f32>(0.0); }
  // Thin ring at the edge of the particle, lightened toward white
  let ring = smoothstep(0.75, 0.85, dist) * smoothstep(1.0, 0.9, dist);
  let light = mix(baseColor, vec3<f32>(1.0), 0.6); // shift toward white
  if (inOrganism) {
    return light * ring * 2.5;
  }
  return light * ring * 1.8;
}

// Quad corners: two triangles forming a quad
const CORNERS = array<vec2<f32>, 6>(
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
  vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corner = CORNERS[vertexIndex];
  let p = particles[instanceIndex];

  var out: VertexOutput;
  out.uv = corner;
  out.color = p.color.rgb;
  out.speed = length(p.vel);
  out.stress = p.stress;
  let det = detectionData[instanceIndex];
  out.detection = vec2<f32>(
    select(0.0, 1.0, (det & 0xFFFFu) > 0u),
    select(0.0, 1.0, (det >> 16u) > 0u),
  );
  out.organelleId = f32(det & 0xFFFFu);
  out.organismId = f32((det >> 16u) & 0xFFu);

  // Offset corner by particle center, scaled by point size (in pixels)
  let scale = radiusScales[instanceIndex];
  let pos = p.pos + corner * renderParams.pointSize * scale * 0.5;
  var clip = (pos / renderParams.resolution) * 2.0 - 1.0;
  clip.y = -clip.y;
  out.position = vec4<f32>(clip, 0.0, 1.0);

  return out;
}

// --- HSL conversion ---

fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
  var t = t_in;
  if (t < 0.0) { t += 1.0; }
  if (t > 1.0) { t -= 1.0; }
  if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
  if (t < 1.0 / 2.0) { return q; }
  if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
  return p;
}

fn hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
  if (hsl.y == 0.0) { return vec3<f32>(hsl.z); }
  var q: f32;
  if (hsl.z < 0.5) { q = hsl.z * (1.0 + hsl.y); } else { q = hsl.z + hsl.y - hsl.z * hsl.y; }
  let p = 2.0 * hsl.z - q;
  return vec3<f32>(
    hue2rgb(p, q, hsl.x + 1.0 / 3.0),
    hue2rgb(p, q, hsl.x),
    hue2rgb(p, q, hsl.x - 1.0 / 3.0),
  );
}
`;

/** Quad post-process prefix: structs, bindings, vertex shader, HSL helpers */
export const QUAD_PREFIX = /* wgsl */ `
struct QuadParams {
  time:   f32,
  param0: f32,
  param1: f32,
  param2: f32,
  param3: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Fullscreen triangle-strip quad
const POSITIONS = array<vec2<f32>, 4>(
  vec2(-1.0, -1.0),
  vec2( 1.0, -1.0),
  vec2(-1.0,  1.0),
  vec2( 1.0,  1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let pos = POSITIONS[vi];
  var out: VertexOutput;
  out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
  out.position = vec4<f32>(pos, 0.0, 1.0);
  return out;
}

// --- HSL conversion ---

fn rgb2hsl(c: vec3<f32>) -> vec3<f32> {
  let maxC = max(c.r, max(c.g, c.b));
  let minC = min(c.r, min(c.g, c.b));
  let l = (maxC + minC) * 0.5;
  if (maxC == minC) { return vec3<f32>(0.0, 0.0, l); }
  let d = maxC - minC;
  var s: f32;
  if (l > 0.5) { s = d / (2.0 - maxC - minC); } else { s = d / (maxC + minC); }
  var h: f32;
  if (maxC == c.r) {
    h = (c.g - c.b) / d;
    if (c.g < c.b) { h += 6.0; }
  } else if (maxC == c.g) {
    h = (c.b - c.r) / d + 2.0;
  } else {
    h = (c.r - c.g) / d + 4.0;
  }
  h /= 6.0;
  return vec3<f32>(h, s, l);
}

fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
  var t = t_in;
  if (t < 0.0) { t += 1.0; }
  if (t > 1.0) { t -= 1.0; }
  if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
  if (t < 1.0 / 2.0) { return q; }
  if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
  return p;
}

fn hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
  if (hsl.y == 0.0) { return vec3<f32>(hsl.z); }
  var q: f32;
  if (hsl.z < 0.5) { q = hsl.z * (1.0 + hsl.y); } else { q = hsl.z + hsl.y - hsl.z * hsl.y; }
  let p = 2.0 * hsl.z - q;
  return vec3<f32>(
    hue2rgb(p, q, hsl.x + 1.0 / 3.0),
    hue2rgb(p, q, hsl.x),
    hue2rgb(p, q, hsl.x - 1.0 / 3.0),
  );
}

@group(0) @binding(0) var sceneTex:     texture_2d<f32>;
@group(0) @binding(1) var sceneSampler: sampler;
@group(0) @binding(2) var<uniform>     quadParams: QuadParams;

// --- Tone mapping ---

fn toneMap(texel: vec4<f32>) -> vec3<f32> {
  if (texel.a < 0.001) { return vec3<f32>(0.0); }
  var hsl = rgb2hsl(texel.rgb / max(texel.a, 0.001));
  hsl.y = mix(hsl.y, quadParams.param1, hsl.y);
  let intensity = clamp(texel.a, 0.0, 1.0);
  hsl.z = intensity * quadParams.param0;
  return hsl2rgb(hsl);
}
`;

/* ------------------------------------------------------------------ */
/*  Effect registries                                                  */
/* ------------------------------------------------------------------ */

export const particleEffects: ShaderEffect[] = [
  {
    id: "gradient", name: "Gradient", category: "particle",
    fragmentSrc: gradientFrag,
    params: [
      { slot: 0, label: "Intensity", default: 1.0, min: 0.1, max: 5, step: 0.1 },
    ],
  },
  {
    id: "solid", name: "Solid", category: "particle",
    fragmentSrc: solidFrag,
    params: [
      { slot: 0, label: "Intensity", default: 1.0, min: 0.1, max: 5, step: 0.1 },
      { slot: 1, label: "Core Size", default: 0.4, min: 0, max: 0.95, step: 0.05 },
    ],
  },
  {
    id: "speed-color", name: "Speed Color", category: "particle",
    fragmentSrc: speedColorFrag,
    params: [
      { slot: 0, label: "Hue Range", default: 0.66, min: 0, max: 1, step: 0.01 },
      { slot: 1, label: "Brightness", default: 0.5, min: 0.1, max: 1, step: 0.05 },
    ],
  },
  {
    id: "stress-color", name: "Stress Color", category: "particle",
    fragmentSrc: stressColorFrag,
    params: [
      { slot: 0, label: "Hue Range", default: 0.66, min: 0, max: 1, step: 0.01 },
      { slot: 1, label: "Brightness", default: 0.5, min: 0.1, max: 1, step: 0.05 },
    ],
  },
];

export const postEffects: ShaderEffect[] = [
  {
    id: "normalize", name: "Normalize", category: "postprocess",
    fragmentSrc: normalizeFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
    ],
  },
  {
    id: "chromatic", name: "Chromatic", category: "postprocess",
    fragmentSrc: chromaticFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
      { slot: 2, label: "Aberration", default: 0.004, min: 0, max: 0.03, step: 0.001 },
    ],
  },
  {
    id: "crt", name: "CRT", category: "postprocess",
    fragmentSrc: crtFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
      { slot: 2, label: "Distortion", default: 0.15, min: 0, max: 0.5, step: 0.01 },
      { slot: 3, label: "Scanlines", default: 800, min: 50, max: 2000, step: 10 },
    ],
  },
  {
    id: "palette", name: "Fire Palette", category: "postprocess",
    fragmentSrc: paletteFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
    ],
  },
  {
    id: "metaball", name: "Metaball", category: "postprocess",
    fragmentSrc: metaballFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
      { slot: 2, label: "Threshold", default: 0.3, min: 0.05, max: 0.8, step: 0.01 },
    ],
  },
  {
    id: "duotone", name: "Duotone", category: "postprocess",
    fragmentSrc: duotoneFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
    ],
  },
  {
    id: "edge-glow", name: "Edge Glow", category: "postprocess",
    fragmentSrc: edgeGlowFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
      { slot: 2, label: "Glow", default: 3.0, min: 0.5, max: 10, step: 0.1 },
      { slot: 3, label: "Fill", default: 0.3, min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    id: "topographic", name: "Topographic", category: "postprocess",
    fragmentSrc: topographicFrag,
    params: [
      { slot: 0, label: "Lightness", default: 1.0, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Contours", default: 12, min: 1, max: 40, step: 1 },
      { slot: 2, label: "Line Width", default: 0.4, min: 0.01, max: 1.5, step: 0.01 },
      { slot: 3, label: "Smoothing", default: 3.0, min: 0.5, max: 8, step: 0.5 },
    ],
  },
  {
    id: "stain", name: "Stain", category: "postprocess",
    fragmentSrc: stainFrag,
    params: [
      { slot: 0, label: "Lightness", default: 0.6, min: 0, max: 2, step: 0.05 },
      { slot: 1, label: "Saturation", default: 0.8, min: 0, max: 2, step: 0.05 },
      { slot: 2, label: "Decay", default: 0.012, min: 0.001, max: 0.1, step: 0.001 },
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  Shader builders — concatenate prefix + effect fragment             */
/* ------------------------------------------------------------------ */

export function buildParticleShader(effect: ShaderEffect): string {
  return PARTICLE_PREFIX + "\n" + effect.fragmentSrc;
}

export function buildQuadShader(effect: ShaderEffect): string {
  return QUAD_PREFIX + "\n" + effect.fragmentSrc;
}

export function findParticleEffect(id: string): ShaderEffect {
  return particleEffects.find(e => e.id === id) ?? particleEffects[0];
}

export function findPostEffect(id: string): ShaderEffect {
  return postEffects.find(e => e.id === id) ?? postEffects[0];
}

/** Returns default param values [param0, param1, ...] for an effect */
export function effectDefaults(effect: ShaderEffect): number[] {
  const vals = [0, 0, 0, 0];
  for (const p of effect.params ?? []) vals[p.slot] = p.default;
  return vals;
}
