// Fullscreen quad that edge-detects on the detection ID texture.
@group(0) @binding(0) var detIdTex: texture_2d<u32>;
@group(0) @binding(1) var detColorTex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));
  let center = textureLoad(detIdTex, coord, 0).r;
  let up    = textureLoad(detIdTex, coord + vec2(0, -1), 0).r;
  let down  = textureLoad(detIdTex, coord + vec2(0,  1), 0).r;
  let left  = textureLoad(detIdTex, coord + vec2(-1, 0), 0).r;
  let right = textureLoad(detIdTex, coord + vec2( 1, 0), 0).r;

  let isEdge = (up != center) || (down != center) || (left != center) || (right != center);
  if (!isEdge) { discard; }

  let color = textureLoad(detColorTex, coord, 0).rgb;
  return vec4<f32>(color, 1.0);
}
