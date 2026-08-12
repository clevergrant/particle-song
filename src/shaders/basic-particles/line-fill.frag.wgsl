// Writes organism ID to r8uint (inflated width). Appended to line-fill-prefix.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
  return in.osmId;
}
