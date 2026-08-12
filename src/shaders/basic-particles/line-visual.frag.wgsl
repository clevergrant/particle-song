// White lines onto canvas, appended to line-prefix.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let fade = smoothstep(0.0, 0.05, in.along) * smoothstep(1.0, 0.95, in.along);
  return vec4<f32>(1.0, 1.0, 1.0, 0.6 * fade);
}
