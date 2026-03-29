// Metaball threshold — smooth blobby isosurfaces from accumulated alpha
// param0 = lightness, param1 = saturation, param2 = threshold
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(sceneTex, sceneSampler, in.uv);

  if (texel.a < 0.001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Threshold the accumulated alpha to create blob boundaries
  let threshold = quadParams.param2;
  let edge = smoothstep(threshold - 0.05, threshold + 0.05, texel.a);

  var hsl = rgb2hsl(texel.rgb / max(texel.a, 0.001));
  hsl.y = mix(hsl.y, quadParams.param1, hsl.y);
  hsl.z = edge * quadParams.param0;

  // Add bright outline at the threshold boundary
  let outline = smoothstep(threshold - 0.06, threshold, texel.a)
              * (1.0 - smoothstep(threshold, threshold + 0.06, texel.a));
  let color = hsl2rgb(hsl) + outline * 0.5;

  return vec4<f32>(color, 1.0);
}
