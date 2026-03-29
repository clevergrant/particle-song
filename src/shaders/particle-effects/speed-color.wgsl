// Speed-based color — maps particle speed to a rainbow hue gradient
// Slow particles are blue/cool, fast particles are red/hot
// param0 = hue range (0-1), param1 = brightness (0-1)
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }

  // Normalize speed relative to running EMA peak
  let t = clamp(in.speed / peakInfo.speedPeak, 0.0, 1.0);

  // Map speed to hue: hueRange (blue, slow) → 0.0 (red, fast)
  let hue = (1.0 - t) * renderParams.param0;
  let sat = 1.0;
  let lit = renderParams.param1;

  let color = hsl2rgb(vec3<f32>(hue, sat, lit));

  // Soft falloff using the falloff LUT
  let falloff = textureSample(falloffLUT, falloffSampler, vec2<f32>(dist, 0.5)).r;

  let det = select(vec3<f32>(0.0), detectionRing(in.uv, in.detection, in.color), renderParams.mode == 1u);
  return vec4<f32>(color * falloff + det, falloff);
}
