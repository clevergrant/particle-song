// Smooth distance-based bubble + Voronoi inter-group boundaries (white). Appended to jfa-edge-prefix.
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(i32(pos.x), i32(pos.y));
  let centerId = textureLoad(idTex, coord, 0).r;

  let jfaCenter = textureLoad(jfaTex, coord, 0).rg;
  if (jfaCenter.x == SENTINEL) { discard; }

  let seedPos = unpackXY(jfaCenter.x);
  let dist = distance(vec2<f32>(coord), seedPos);
  let groupId = jfaCenter.y;

  // Smooth distance-based bubble edge around each organism
  let halfEdge = bubbleParams.edgeWidth * 0.5;
  let bubble = dist > (bubbleParams.threshold - halfEdge) && dist < (bubbleParams.threshold + halfEdge);

  // JFA Voronoi inter-group boundary (organism vs organism)
  var voronoi = false;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nc = coord + vec2<i32>(dx, dy);
      let n = textureLoad(jfaTex, nc, 0).rg;
      let nId = textureLoad(idTex, nc, 0).r;
      if (n.x != SENTINEL && n.y != groupId && dist < bubbleParams.threshold) {
        voronoi = true;
      }
    }
  }

  if (!bubble && !voronoi) { discard; }
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
