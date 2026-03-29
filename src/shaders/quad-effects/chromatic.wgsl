// Chromatic aberration — offset R/G/B channels radially from screen center
// param0 = lightness, param1 = saturation, param2 = aberration strength
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let center = vec2<f32>(0.5, 0.5);
  let dir = in.uv - center;
  let strength = quadParams.param2;

  // Sample all three positions in uniform control flow
  let texR = textureSample(sceneTex, sceneSampler, in.uv + dir * strength);
  let texG = textureSample(sceneTex, sceneSampler, in.uv);
  let texB = textureSample(sceneTex, sceneSampler, in.uv - dir * strength);

  // Tone-map each sample independently with its own alpha
  let colorR = toneMap(texR);
  let colorG = toneMap(texG);
  let colorB = toneMap(texB);

  return vec4<f32>(colorR.r, colorG.g, colorB.b, 1.0);
}
