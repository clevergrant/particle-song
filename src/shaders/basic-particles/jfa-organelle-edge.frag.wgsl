// Smooth distance-based bubble + Voronoi where bubbles touch. Appended to jfa-edge-prefix.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));

  let jfaCenter = textureLoad(jfaTex, coord, 0).rg;
  if (jfaCenter.x == SENTINEL) { discard; }

  let seedPos = unpackXY(jfaCenter.x);
  let dist = distance(vec2<f32>(coord), seedPos);
  let groupId = jfaCenter.y;

  // Smooth distance-based bubble edge around each organelle (half organism threshold)
  let orgThreshold = bubbleParams.organelleThreshold;
  let halfEdge = bubbleParams.edgeWidth * 0.5;
  let bubble = dist > (orgThreshold - halfEdge) && dist < (orgThreshold + halfEdge);

  // Voronoi boundary where organelle bubbles overlap
  var voronoi = false;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nc = coord + vec2<i32>(dx, dy);
      let n = textureLoad(jfaTex, nc, 0).rg;
      if (n.x != SENTINEL && n.y != groupId && dist < orgThreshold) {
        voronoi = true;
      }
    }
  }

  if (!bubble && !voronoi) { discard; }
  // Sample color from nearest seed position (current pixel may be far from any particle)
  let seedCoord = vec2<i32>(unpackXY(jfaCenter.x));
  let color = textureLoad(detColorTex, seedCoord, 0).rgb;
  return vec4<f32>(color, 1.0);
}
