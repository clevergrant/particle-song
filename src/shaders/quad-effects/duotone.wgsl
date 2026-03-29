// Duotone — map low intensity to one color, high to another
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(sceneTex, sceneSampler, in.uv);

  if (texel.a < 0.001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  let intensity = clamp(texel.a, 0.0, 1.0) * quadParams.param0;

  // Deep blue → hot pink
  let colorLow = vec3<f32>(0.05, 0.05, 0.3);
  let colorHigh = vec3<f32>(1.0, 0.2, 0.6);
  let color = mix(colorLow, colorHigh, intensity);

  return vec4<f32>(color, 1.0);
}
