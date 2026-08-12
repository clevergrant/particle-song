// Organism JFA seed: writes packed (x,y) + organismId to rg32uint.
// Appended to the depth-modified PARTICLE_PREFIX (same prefix as organism-fill).
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<u32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.y < 0.5) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  return vec2<u32>(packed, u32(in.organismId));
}
