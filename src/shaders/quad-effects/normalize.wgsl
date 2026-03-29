// Default HSL normalization
// param0 = lightness, param1 = saturation
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(sceneTex, sceneSampler, in.uv);

  if (texel.a < 0.001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  return vec4<f32>(toneMap(texel), 1.0);
}
