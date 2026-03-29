struct JfaParams {
  stepSize: u32,
  width:    u32,
  height:   u32,
  _pad:     u32,
};

@group(0) @binding(0) var jfaIn:  texture_2d<u32>;
@group(0) @binding(1) var jfaOut: texture_storage_2d<rg32uint, write>;
@group(0) @binding(2) var<uniform> params: JfaParams;

const SENTINEL = 0xFFFFFFFFu;

fn unpackXY(packed: u32) -> vec2<f32> {
  return vec2<f32>(f32(packed >> 16u), f32(packed & 0xFFFFu));
}

fn seedDistSq(packed: u32, pos: vec2<f32>) -> f32 {
  if (packed == SENTINEL) { return 1e18; }
  let seed = unpackXY(packed);
  let d = seed - pos;
  return dot(d, d);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pos = gid.xy;
  if (pos.x >= params.width || pos.y >= params.height) { return; }

  let coord = vec2<i32>(pos);
  let step = i32(params.stepSize);
  let fpos = vec2<f32>(pos);

  let cur = textureLoad(jfaIn, coord, 0).rg;
  var bestPacked = cur.x;
  var bestId = cur.y;
  var bestDist = seedDistSq(bestPacked, fpos);

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nc = coord + vec2<i32>(dx, dy) * step;
      if (nc.x < 0 || nc.y < 0 || nc.x >= i32(params.width) || nc.y >= i32(params.height)) {
        continue;
      }
      let neighbor = textureLoad(jfaIn, nc, 0).rg;
      let d = seedDistSq(neighbor.x, fpos);
      if (d < bestDist) {
        bestDist = d;
        bestPacked = neighbor.x;
        bestId = neighbor.y;
      }
    }
  }

  textureStore(jfaOut, coord, vec4<u32>(bestPacked, bestId, 0u, 0u));
}
