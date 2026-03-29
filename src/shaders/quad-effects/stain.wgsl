// Stain — particles leave colored trails that decay to transparent.
// Uses a persistent stain texture (ping-pong) as the background.
// param0 = lightness, param1 = saturation, param2 = decay rate

@group(0) @binding(3) var stainTex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texel = textureSample(sceneTex, sceneSampler, in.uv);

  // Read stain (RGBA — alpha = trail intensity)
  let dims = textureDimensions(stainTex, 0);
  let coord = vec2<i32>(in.uv * vec2<f32>(dims));
  let stain = textureLoad(stainTex, coord, 0);

  // Stain trail composited over black
  var bg = stain.rgb * stain.a;

  if (texel.a < 0.001) {
    return vec4<f32>(bg, 1.0);
  }

  var hsl = rgb2hsl(texel.rgb / max(texel.a, 0.001));
  hsl.y = mix(hsl.y, quadParams.param1, hsl.y);
  let intensity = clamp(texel.a, 0.0, 1.0);
  hsl.z = intensity * quadParams.param0;
  let color = hsl2rgb(hsl);

  // Alpha-composite particles over the stain trail
  return vec4<f32>(mix(bg, color, intensity), 1.0);
}
