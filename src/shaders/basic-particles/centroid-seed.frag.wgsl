// Organism centroid seed: writes packed coords + osmId to rg32uint (inflated radius).
// Appended to centroid-fill-prefix.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  return vec2<u32>(packed, in.osmId);
}
