// Writes organism ID to r8uint (inflated radius). Appended to centroid-fill-prefix.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  return in.osmId;
}
