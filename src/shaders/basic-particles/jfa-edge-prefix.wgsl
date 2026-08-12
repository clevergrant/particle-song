// Shared prefix for JFA edge shaders (organelle + organism variants).
@group(0) @binding(0) var jfaTex: texture_2d<u32>;
@group(0) @binding(1) var detColorTex: texture_2d<f32>;

struct BubbleParams {
  threshold: f32,
  edgeWidth: f32,
  organelleThreshold: f32,
  _pad1: f32,
};
@group(0) @binding(2) var<uniform> bubbleParams: BubbleParams;
@group(0) @binding(3) var idTex: texture_2d<u32>;

const SENTINEL = 0xFFFFFFFFu;

fn unpackXY(packed: u32) -> vec2<f32> {
  return vec2<f32>(f32(packed >> 16u), f32(packed & 0xFFFFu));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}
