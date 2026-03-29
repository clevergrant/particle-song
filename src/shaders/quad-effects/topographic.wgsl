// Topographic contour lines — treats particle density as a height field
// and draws smooth contour lines at regular elevation intervals.
// param0 = lightness, param1 = line count, param2 = line width, param3 = smoothing
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(sceneTex, 0));
  let px = 1.0 / texSize;

  // Sample center and neighbors for Gaussian-blurred height field
  let sigma = quadParams.param3;
  let steps = 3;
  var height = 0.0;
  var weight = 0.0;
  for (var y = -steps; y <= steps; y++) {
    for (var x = -steps; x <= steps; x++) {
      let offset = vec2<f32>(f32(x), f32(y)) * px * sigma;
      let w = exp(-f32(x * x + y * y) / (2.0 * sigma * sigma + 0.001));
      height += textureSample(sceneTex, sceneSampler, in.uv + offset).a * w;
      weight += w;
    }
  }
  height /= weight;

  // Compute gradient of the height field for line width compensation
  let hL = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(-px.x, 0.0) * sigma).a;
  let hR = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>( px.x, 0.0) * sigma).a;
  let hD = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(0.0, -px.y) * sigma).a;
  let hU = textureSample(sceneTex, sceneSampler, in.uv + vec2<f32>(0.0,  px.y) * sigma).a;
  let grad = vec2<f32>(hR - hL, hU - hD) * 0.5;
  let gradMag = length(grad);

  // Number of contour levels and line thickness
  let levels = quadParams.param1;
  let lineWidth = quadParams.param2;

  // Map height to contour space
  let scaled = height * levels;
  let contourDist = abs(fract(scaled + 0.5) - 0.5);

  // Anti-aliased contour line using gradient magnitude for screen-space width
  let screenGrad = gradMag * levels;
  let aa = max(screenGrad, 0.001);
  let line = 1.0 - smoothstep(0.0, lineWidth * aa + 0.01, contourDist);

  // Emphasize every 5th contour as thicker "index" contour
  let indexScaled = height * levels * 0.2;
  let indexDist = abs(fract(indexScaled + 0.5) - 0.5);
  let indexLine = 1.0 - smoothstep(0.0, lineWidth * 1.8 * aa + 0.01, indexDist);

  let combinedLine = max(line * 0.6, indexLine);

  // Fade out where there's no particle data
  let presence = smoothstep(0.0, 0.05, height);

  let brightness = combinedLine * quadParams.param0 * presence;

  return vec4<f32>(vec3<f32>(brightness), 1.0);
}
