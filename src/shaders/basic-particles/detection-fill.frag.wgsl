// Appended to PARTICLE_PREFIX. Writes organelle ID to R8Uint + color to RGBA8.
struct FillOutput {
  @location(0) id: u32,
  @location(1) color: vec4<f32>,
};
@fragment
fn fs_main(in: VertexOutput) -> FillOutput {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.x < 0.5) { discard; }
  var out: FillOutput;
  out.id = u32(in.organelleId);
  out.color = vec4<f32>(in.color, 1.0);
  return out;
}
