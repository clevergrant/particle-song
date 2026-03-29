// Edge detection with neon glow — Sobel filter on accumulated texture
// param0 = lightness, param1 = saturation, param2 = glow strength, param3 = fill dim
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(sceneTex, 0));
  let px = 1.0 / texSize;

  // Sample 3x3 neighborhood (alpha channel = intensity)
  let tl = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(-px.x,  px.y)).a;
  let tc = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( 0.0,   px.y)).a;
  let tr = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( px.x,  px.y)).a;
  let ml = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(-px.x,  0.0)).a;
  let mr = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( px.x,  0.0)).a;
  let bl = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(-px.x, -px.y)).a;
  let bc = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( 0.0,  -px.y)).a;
  let br = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( px.x, -px.y)).a;

  // Sobel operators
  let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
  let edge = sqrt(gx * gx + gy * gy);

  // Base color from the scene (dimmed by param3)
  let texel = textureSample(sceneTex, sceneSampler, in.uv);
  var baseColor = vec3<f32>(0.0);
  if (texel.a > 0.001) {
    var hsl = rgb2hsl(texel.rgb / max(texel.a, 0.001));
    hsl.y = mix(hsl.y, quadParams.param1, hsl.y);
    let intensity = clamp(texel.a, 0.0, 1.0);
    hsl.z = intensity * quadParams.param0 * quadParams.param3;
    baseColor = hsl2rgb(hsl);
  }

  // Add neon edge glow
  let neonColor = vec3<f32>(0.3, 0.9, 1.0); // cyan neon
  let glow = clamp(edge * quadParams.param2, 0.0, 1.0);
  let color = baseColor + neonColor * glow * quadParams.param0;

  return vec4<f32>(color, 1.0);
}
