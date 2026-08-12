// Thin white ring for organism-level centroids, appended to centroid-prefix.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let ring = smoothstep(0.82, 0.88, dist) * smoothstep(1.0, 0.94, dist);
  if (ring < 0.01) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, ring);
}
