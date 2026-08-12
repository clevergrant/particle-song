// Appended to the depth-modified PARTICLE_PREFIX. Writes organism ID to R8Uint.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.y < 0.5) { discard; }
  return u32(in.organismId);
}
