// Color palette mapping — intensity mapped through a fire/plasma gradient
fn palette(t: f32) -> vec3<f32> {
  // Fire palette: black → red → orange → yellow → white
  let c0 = vec3<f32>(0.0, 0.0, 0.0);     // black
  let c1 = vec3<f32>(0.5, 0.0, 0.1);     // dark red
  let c2 = vec3<f32>(1.0, 0.3, 0.0);     // orange
  let c3 = vec3<f32>(1.0, 0.8, 0.2);     // yellow
  let c4 = vec3<f32>(1.0, 1.0, 0.9);     // white-hot

  if (t < 0.25) { return mix(c0, c1, t * 4.0); }
  if (t < 0.5)  { return mix(c1, c2, (t - 0.25) * 4.0); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.5) * 4.0); }
  return mix(c3, c4, (t - 0.75) * 4.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(sceneTex, sceneSampler, in.uv);

  if (texel.a < 0.001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  let intensity = clamp(texel.a, 0.0, 1.0) * quadParams.param0;
  let color = palette(intensity);
  return vec4<f32>(color, 1.0);
}
