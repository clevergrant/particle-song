// Solid circle with soft glow halo
// param0 = intensity, param1 = core size (0-0.95)
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }

  // Solid core, then blend into falloff glow
  let coreSize = renderParams.param1;
  let core = smoothstep(coreSize + 0.05, coreSize - 0.05, dist);
  let glow = textureSample(falloffLUT, falloffSampler, vec2<f32>(dist, 0.5)).r;
  let intensity = max(core, glow) * renderParams.param0;

  // Detection ring in circle-overlay passes (mode >= 1)
  let det = select(vec3<f32>(0.0), detectionRing(in.uv, in.detection, in.color), renderParams.mode >= 1u);
  // Detection-only mode (mode=2): discard particles not in any organelle
  let hasDet = in.detection.x > 0.5 || in.detection.y > 0.5;
  if (renderParams.mode == 2u && !hasDet) { discard; }
  // In overlay mode, use particle albedo directly for fill; otherwise use intensity-modulated color
  let fill = select(in.color * intensity, in.color * intensity + det, renderParams.mode >= 1u);
  let alpha = select(intensity, max(intensity, max(det.r, max(det.g, det.b))), renderParams.mode >= 1u);
  return vec4<f32>(fill, alpha);
}
