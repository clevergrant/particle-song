// Stain update — decay old stain toward transparent, stamp particle colors
// Reads: old stain texture + offscreen particle accumulation texture
// Writes: new stain texture (RGBA — alpha tracks trail intensity)

struct StainParams {
  _pad0:     f32,
  _pad1:     f32,
  _pad2:     f32,
  decayRate: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

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

@group(0) @binding(0) var oldStain:      texture_2d<f32>;
@group(0) @binding(1) var stainSampler:  sampler;
@group(0) @binding(2) var particleTex:   texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: StainParams;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let old = textureSample(oldStain, stainSampler, in.uv);
  let particle = textureSample(particleTex, stainSampler, in.uv);

  // Decay toward transparent — subtract at least 1/255 so 8-bit values always shrink
  var color = old.rgb;
  let step = max(old.a * params.decayRate, 1.0 / 255.0);
  var alpha = max(old.a - step, 0.0);

  // Where particles exist, stamp their normalized color
  if (particle.a > 0.01) {
    let particleColor = particle.rgb / max(particle.a, 0.001);
    let t = clamp(particle.a, 0.0, 1.0);
    color = mix(color, particleColor, t);
    alpha = mix(alpha, 1.0, t);
  }

  return vec4<f32>(color, alpha);
}
