// CRT monitor — scanlines + barrel distortion + vignette
// param0 = lightness, param1 = saturation, param2 = distortion, param3 = scanline freq
fn barrelDistort(uv: vec2<f32>, strength: f32) -> vec2<f32> {
  let centered = uv - 0.5;
  let r2 = dot(centered, centered);
  return uv + centered * r2 * strength;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let distorted = barrelDistort(in.uv, quadParams.param2);

  // Sample before branching — textureSample requires uniform control flow
  let texel = textureSample(sceneTex, sceneSampler, clamp(distorted, vec2(0.0), vec2(1.0)));

  // Out of bounds → black
  if (distorted.x < 0.0 || distorted.x > 1.0 || distorted.y < 0.0 || distorted.y > 1.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  if (texel.a < 0.001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  var color = toneMap(texel);

  // Scanlines
  let scanline = sin(distorted.y * quadParams.param3) * 0.5 + 0.5;
  color *= 0.85 + 0.15 * scanline;

  // Vignette
  let centered = distorted - 0.5;
  let vignette = 1.0 - dot(centered, centered) * 1.5;
  color *= clamp(vignette, 0.0, 1.0);

  return vec4<f32>(color, 1.0);
}
