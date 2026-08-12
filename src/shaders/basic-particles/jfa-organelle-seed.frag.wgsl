// Organelle JFA seed: writes packed (x,y) + organelleId to rg32uint.
// Appended to PARTICLE_PREFIX.
struct SeedOutput {
  @location(0) seed: vec2<u32>,
  @location(1) color: vec4<f32>,
};
@fragment
fn fs_main(in: VertexOutput) -> SeedOutput {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  if (in.detection.x < 0.5) { discard; }
  let px = vec2<u32>(u32(in.position.x), u32(in.position.y));
  let packed = (px.x << 16u) | px.y;
  var out: SeedOutput;
  out.seed = vec2<u32>(packed, u32(in.organelleId));
  out.color = vec4<f32>(in.color, 1.0);
  return out;
}
