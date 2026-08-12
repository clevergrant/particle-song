// Organism edge pipeline: fullscreen edge detection, white outlines.
@group(0) @binding(0) var osmIdTex: texture_2d<u32>;

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
  let center = textureLoad(osmIdTex, coord, 0).r;
  let up    = textureLoad(osmIdTex, coord + vec2(0, -1), 0).r;
  let down  = textureLoad(osmIdTex, coord + vec2(0,  1), 0).r;
  let left  = textureLoad(osmIdTex, coord + vec2(-1, 0), 0).r;
  let right = textureLoad(osmIdTex, coord + vec2( 1, 0), 0).r;
  let isEdge = (up != center) || (down != center) || (left != center) || (right != center);
  if (!isEdge) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
