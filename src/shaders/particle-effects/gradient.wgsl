// Default gradient mode — uses falloff LUT for soft glow
// param0 = intensity
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let falloff = textureSample(falloffLUT, falloffSampler, vec2<f32>(dist, 0.5)).r;
  let intensity = falloff * renderParams.param0;
  // Detection ring only in circle-overlay pass (mode=1) to avoid additive accumulation
  let det = select(vec3<f32>(0.0), detectionRing(in.uv, in.detection, in.color), renderParams.mode == 1u);
  return vec4<f32>(in.color * intensity + det, intensity);
}
